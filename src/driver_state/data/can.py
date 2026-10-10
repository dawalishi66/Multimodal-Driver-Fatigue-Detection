"""Manifest-backed UL-DD CAN datasets and the shared batch contract.

Labels, identities, splits, and time ranges come only from the frozen CSV
manifest. NPZ files contain numeric features and quality arrays only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from driver_state.constants import KSS_CLASSES, SPLITS, ULDD_SPLIT_BY_SUBJECT, kss_label
from driver_state.preprocessing.fatigue_can.pipeline import CAN_FEATURE_COLUMNS

CAN_ARRAY_DTYPES = {
    "x": np.dtype("float32"),
    "time_s": np.dtype("float64"),
    "valid_mask": np.dtype("bool"),
    "support_s": np.dtype("float64"),
    "observed_fraction": np.dtype("float32"),
}
REQUIRED_WINDOW_FIELDS = {
    "sample_id", "modality", "subject_id", "session_id", "split",
    "window_index", "window_start_ms", "window_end_ms", "duration_ms",
    "label_start_ms", "label_end_ms", "kss_score", "label_class", "label_id",
    "valid", "valid_ratio", "feature_path", "feature_shape", "feature_dtype",
    "extractor_name", "extractor_version", "error", "parent_id",
    "feature_columns", "feature_sha256", "qc_status",
}
REQUIRED_PARENT_FIELDS = {
    "parent_id", "subject_id", "session_id", "split", "label_id",
    "window_count", "valid_window_count", "complete_for_240s",
}


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be true/false or 1/0, got {value!r}")


def _safe_relative_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute():
        raise ValueError("feature_path must be a non-empty relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("feature_path escapes feature_root") from exc
    return resolved


@dataclass(frozen=True)
class CanWindowRecord:
    sample_id: str
    parent_id: str
    subject_id: str
    session_id: str
    split: str
    window_index: int
    label_id: int
    label_class: str
    kss_score: float
    feature_path: Path
    feature_sha256: str
    valid_ratio: float
    extractor_name: str
    extractor_version: str


def load_complete_parent_ids(parent_manifest: Path | str, *, split: str) -> set[str]:
    path = Path(parent_manifest)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_PARENT_FIELDS - fields)
        if missing:
            raise ValueError(f"parent manifest is missing fields: {missing}")
        rows = list(reader)

    parent_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        if row["split"] != split:
            raise ValueError(
                f"parent row {row_number} uses split={row['split']!r}, expected {split!r}"
            )
        if not _strict_bool(row["complete_for_240s"], field="complete_for_240s"):
            raise ValueError(f"parent row {row_number} is not complete_for_240s")
        if int(row["window_count"]) != 8 or int(row["valid_window_count"]) != 8:
            raise ValueError(f"parent row {row_number} must contain eight valid windows")
        parent_id = row["parent_id"].strip()
        if not parent_id or parent_id in parent_ids:
            raise ValueError(f"invalid or duplicate parent_id at row {row_number}")
        parent_ids.add(parent_id)
    if not parent_ids:
        raise ValueError("complete-parent manifest is empty")
    return parent_ids


class CanWindowDataset(Dataset):
    """Lazy reader for one frozen split of valid CAN windows.

    Passing a parent manifest restricts the dataset to the conservative
    complete-8 cohort. Test loading is denied unless allow_test=True is
    deliberately supplied by a frozen evaluation entry point.
    """

    def __init__(
        self,
        manifest: Path | str,
        *,
        feature_root: Path | str,
        split: str,
        parent_manifest: Path | str | None = None,
        allow_test: bool = False,
        verify_feature_hash: bool = False,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError(
                "test manifest access is locked; a frozen evaluation must pass allow_test=True"
            )
        self.manifest_path = Path(manifest)
        self.feature_root = Path(feature_root)
        self.split = split
        self.verify_feature_hash = bool(verify_feature_hash)
        self.manifest_sha256 = file_sha256(self.manifest_path)
        parent_ids = (
            load_complete_parent_ids(parent_manifest, split=split)
            if parent_manifest is not None else None
        )

        with self.manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_WINDOW_FIELDS - fields)
            if missing:
                raise ValueError(f"window manifest is missing fields: {missing}")
            rows = list(reader)

        records: list[CanWindowRecord] = []
        sample_ids: set[str] = set()
        parent_windows: dict[str, list[tuple[int, int]]] = {}
        for row_number, row in enumerate(rows, start=2):
            if row["split"] != split:
                raise ValueError(
                    f"window row {row_number} uses split={row['split']!r}, expected {split!r}"
                )
            subject = row["subject_id"].strip()
            if ULDD_SPLIT_BY_SUBJECT.get(subject) != split:
                raise ValueError(f"subject {subject!r} does not belong to split {split!r}")
            if row["modality"] != "can":
                raise ValueError(f"row {row_number} must use modality=can")
            if not _strict_bool(row["valid"], field="valid") or row["qc_status"] != "PASS":
                raise ValueError("split window manifests may contain only valid PASS rows")
            if row["error"].strip():
                raise ValueError(f"valid row {row_number} must not contain an error")
            if not row["session_id"].strip():
                raise ValueError(f"session_id is empty at row {row_number}")
            if not row["extractor_name"].strip() or not row["extractor_version"].strip():
                raise ValueError(f"extractor identity is empty at row {row_number}")
            parent_id = row["parent_id"].strip()
            if parent_ids is not None and parent_id not in parent_ids:
                continue

            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"invalid or duplicate sample_id at row {row_number}")
            sample_ids.add(sample_id)
            label_id = int(row["label_id"])
            score = float(row["kss_score"])
            mapped_id, mapped_class = kss_label(score)
            if (label_id, row["label_class"]) != (mapped_id, mapped_class):
                raise ValueError(f"KSS mapping mismatch at row {row_number}")
            if not 0 <= label_id < len(KSS_CLASSES):
                raise ValueError(f"invalid label_id at row {row_number}")
            if int(row["duration_ms"]) != 30_000:
                raise ValueError(f"CAN row {row_number} is not a 30-second window")
            if int(row["window_end_ms"]) - int(row["window_start_ms"]) != 30_000:
                raise ValueError(f"CAN row {row_number} has inconsistent time bounds")
            if int(row["label_end_ms"]) - int(row["label_start_ms"]) != 240_000:
                raise ValueError(f"CAN row {row_number} has an invalid parent duration")
            window_index = int(row["window_index"])
            if window_index not in range(8):
                raise ValueError(f"window_index must be 0..7 at row {row_number}")
            expected_start = int(row["label_start_ms"]) + window_index * 30_000
            if (
                int(row["window_start_ms"]) != expected_start
                or int(row["window_end_ms"]) != expected_start + 30_000
            ):
                raise ValueError(
                    f"window time does not match its parent-relative index at row {row_number}"
                )
            if json.loads(row["feature_shape"]) != [300, 9]:
                raise ValueError(f"feature_shape must be [300,9] at row {row_number}")
            if row["feature_dtype"] != "float32":
                raise ValueError(f"feature_dtype must be float32 at row {row_number}")
            if json.loads(row["feature_columns"]) != list(CAN_FEATURE_COLUMNS):
                raise ValueError(f"feature column order mismatch at row {row_number}")
            valid_ratio = float(row["valid_ratio"])
            if not 0.95 <= valid_ratio <= 1.0:
                raise ValueError(f"valid_ratio is outside the approved range at row {row_number}")
            feature_path = _safe_relative_path(self.feature_root, row["feature_path"])
            if not feature_path.is_file():
                raise FileNotFoundError(f"feature file does not exist: {row['feature_path']}")
            record = CanWindowRecord(
                sample_id=sample_id,
                parent_id=parent_id,
                subject_id=subject,
                session_id=row["session_id"].strip(),
                split=split,
                window_index=window_index,
                label_id=label_id,
                label_class=row["label_class"],
                kss_score=score,
                feature_path=feature_path,
                feature_sha256=row["feature_sha256"],
                valid_ratio=valid_ratio,
                extractor_name=row["extractor_name"],
                extractor_version=row["extractor_version"],
            )
            records.append(record)
            parent_windows.setdefault(parent_id, []).append((window_index, label_id))

        if not records:
            raise ValueError(f"no eligible CAN windows found for split {split!r}")
        if parent_ids is not None:
            if set(parent_windows) != parent_ids:
                missing = sorted(parent_ids - set(parent_windows))
                raise ValueError(f"complete parents missing from window manifest: {missing[:5]}")
            for parent_id, windows in parent_windows.items():
                if sorted(index for index, _ in windows) != list(range(8)):
                    raise ValueError(f"parent {parent_id} does not contain windows 0..7 exactly once")
                if len({label for _, label in windows}) != 1:
                    raise ValueError(f"parent {parent_id} contains conflicting labels")
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.verify_feature_hash and file_sha256(record.feature_path) != record.feature_sha256:
            raise ValueError(f"feature hash mismatch for {record.sample_id}")
        with np.load(record.feature_path, allow_pickle=False) as archive:
            if set(archive.files) != set(CAN_ARRAY_DTYPES):
                raise ValueError(
                    f"{record.sample_id} must contain exactly {sorted(CAN_ARRAY_DTYPES)}"
                )
            arrays = {name: np.asarray(archive[name]) for name in CAN_ARRAY_DTYPES}

        expected_shapes = {
            "x": (300, 9), "time_s": (300,), "valid_mask": (300,),
            "support_s": (300, 2), "observed_fraction": (300,),
        }
        for name, expected_dtype in CAN_ARRAY_DTYPES.items():
            value = arrays[name]
            if value.dtype != expected_dtype or value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{record.sample_id}:{name} expected "
                    f"{expected_dtype}{expected_shapes[name]}, got {value.dtype}{value.shape}"
                )
            if value.dtype.kind in "f" and not np.isfinite(value).all():
                raise ValueError(f"{record.sample_id}:{name} contains NaN or Inf")
        if not arrays["valid_mask"].any():
            raise ValueError(f"{record.sample_id} has no valid CAN token")
        if not np.isclose(
            float(arrays["valid_mask"].mean()), record.valid_ratio, rtol=0, atol=1e-6
        ):
            raise ValueError(f"{record.sample_id}:valid_ratio differs from valid_mask")
        if not np.all((arrays["observed_fraction"] >= 0) & (arrays["observed_fraction"] <= 1)):
            raise ValueError(f"{record.sample_id}:observed_fraction must be within [0,1]")

        return {
            "inputs": {
                "can": {
                    "x": torch.from_numpy(arrays["x"].copy()),
                    "time_s": torch.from_numpy(arrays["time_s"].copy()),
                    "valid_mask": torch.from_numpy(arrays["valid_mask"].copy()),
                    "support_s": torch.from_numpy(arrays["support_s"].copy()),
                    "observed_fraction": torch.from_numpy(
                        arrays["observed_fraction"].copy()
                    ),
                }
            },
            "label": torch.tensor(record.label_id, dtype=torch.int64),
            "sample_id": record.sample_id,
            "parent_id": record.parent_id,
            "subject_id": record.subject_id,
            "session_id": record.session_id,
            "window_index": record.window_index,
            "split": record.split,
        }


def collate_can_batch(
    samples: Sequence[dict[str, Any]],
    *,
    standardizer: Any | None = None,
) -> dict[str, Any]:
    """Right-pad CAN samples while preserving internal invalid tokens."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    feature_dim = int(samples[0]["inputs"]["can"]["x"].shape[1])
    max_tokens = max(int(sample["inputs"]["can"]["x"].shape[0]) for sample in samples)
    batch_size = len(samples)
    x = torch.zeros((batch_size, max_tokens, feature_dim), dtype=torch.float32)
    time_s = torch.zeros((batch_size, max_tokens), dtype=torch.float64)
    valid_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    support_s = torch.zeros((batch_size, max_tokens, 2), dtype=torch.float64)
    observed_fraction = torch.zeros((batch_size, max_tokens), dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        can = sample["inputs"]["can"]
        tokens = int(can["x"].shape[0])
        if can["x"].shape != (tokens, feature_dim):
            raise ValueError("all CAN samples must share one feature dimension")
        x[batch_index, :tokens] = can["x"]
        time_s[batch_index, :tokens] = can["time_s"]
        valid_mask[batch_index, :tokens] = can["valid_mask"]
        support_s[batch_index, :tokens] = can["support_s"]
        observed_fraction[batch_index, :tokens] = can["observed_fraction"]
    if (~valid_mask.any(dim=1)).any():
        raise ValueError("all-invalid CAN samples are forbidden")
    if standardizer is not None:
        x = standardizer.transform_tensor(x, valid_mask)
    else:
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    return {
        "inputs": {
            "can": {
                "x": x,
                "valid_mask": valid_mask,
                "time_s": time_s,
                "support_s": support_s,
                "observed_fraction": observed_fraction,
            }
        },
        "labels": torch.stack([sample["label"] for sample in samples]),
        "sample_id": [sample["sample_id"] for sample in samples],
        "parent_id": [sample["parent_id"] for sample in samples],
        "subject_id": [sample["subject_id"] for sample in samples],
        "session_id": [sample["session_id"] for sample in samples],
        "window_index": torch.tensor(
            [sample["window_index"] for sample in samples], dtype=torch.int64
        ),
        "split": [sample["split"] for sample in samples],
    }


def make_can_collate(standardizer: Any | None = None) -> Callable:
    return partial(collate_can_batch, standardizer=standardizer)
