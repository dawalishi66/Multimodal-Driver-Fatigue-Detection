"""Manifest-backed UL-DD video/CAN data loading and collation.

The frozen pair manifest is the only source of labels and identities. Feature
archives contain numeric arrays only. Test access remains locked unless a
future frozen evaluation entry point opts in explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from driver_state.constants import (
    FATIGUE_PARENT_MS,
    FATIGUE_WINDOW_MS,
    KSS_CLASSES,
    SPLITS,
    ULDD_SPLIT_BY_SUBJECT,
    kss_label,
)
from driver_state.data.can import CAN_ARRAY_DTYPES, file_sha256
from driver_state.data.fatigue_pairing import (
    canonical_pair_id,
    normalize_video_time_arrays,
    parse_bool,
    validate_normalized_video_time,
)


MODALITIES = ("video", "can")
MODEL_INPUT_FIELDS = ("x", "valid_mask", "time_s")
FEATURE_ARRAY_FIELDS = tuple(CAN_ARRAY_DTYPES)
REQUIRED_PAIR_FIELDS = {
    "paired_sample_id",
    "parent_id",
    "subject_id",
    "session_id",
    "split",
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "duration_ms",
    "label_start_ms",
    "label_end_ms",
    "kss_score",
    "label_class",
    "label_id",
    "paired_valid",
    "exclude_reason",
    "complete8_eligible",
    "complete8_exclude_reason",
    "sync_status",
    "offset_ms",
    "hardware_clock_verified",
    "video_sample_id",
    "video_subwindow_ids",
    "video_feature_archive",
    "video_feature_member",
    "video_feature_shape",
    "video_feature_dtype",
    "video_feature_sha256",
    "video_source_time_reference",
    "video_time_transform",
    "can_sample_id",
    "can_feature_path",
    "can_feature_shape",
    "can_feature_dtype",
    "can_feature_sha256",
    "can_valid_ratio",
}


def _safe_relative_path(root: Path, value: str, *, field: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a safe, non-empty relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the UL-DD root") from exc
    return resolved


def _safe_zip_member(value: str) -> str:
    member = PurePosixPath(value)
    if not value or member.is_absolute() or ".." in member.parts:
        raise ValueError("video_feature_member must be a safe relative ZIP member")
    return member.as_posix()


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return normalized


def _parse_shape(value: str, *, field: str) -> tuple[int, int]:
    try:
        shape = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON") from exc
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise ValueError(f"{field} must contain two positive integers")
    return int(shape[0]), int(shape[1])


@dataclass(frozen=True)
class FatigueVideoCanRecord:
    sample_id: str
    parent_id: str
    subject_id: str
    session_id: str
    split: str
    window_index: int
    window_start_ms: int
    window_end_ms: int
    label_start_ms: int
    label_end_ms: int
    kss_score: float
    label_class: str
    label_id: int
    video_sample_id: str
    video_subwindow_ids: tuple[str, ...]
    video_archive_path: Path
    video_feature_member: str
    video_feature_shape: tuple[int, int]
    video_feature_sha256: str
    can_sample_id: str
    can_feature_path: Path
    can_feature_shape: tuple[int, int]
    can_feature_sha256: str
    can_valid_ratio: float


class FatigueVideoCanDataset(Dataset):
    """Read one split of the frozen complete-8 video/CAN cohort.

    Current v1 inputs are video ``[6,96]`` and CAN ``[300,9]``. The dimensions
    are also checked against every manifest row instead of being inferred from
    the first file.
    """

    def __init__(
        self,
        pair_manifest: Path | str,
        *,
        dataset_root: Path | str,
        split: str,
        allow_test: bool = False,
        verify_feature_hashes: bool = False,
        cache_features: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError(
                "test pair access is locked; a frozen evaluation must pass allow_test=True"
            )

        self.pair_manifest_path = Path(pair_manifest)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.verify_feature_hashes = bool(verify_feature_hashes)
        self.cache_features = bool(cache_features)
        self.manifest_sha256 = file_sha256(self.pair_manifest_path)
        self._feature_cache: dict[int, dict[str, dict[str, np.ndarray]]] = {}

        with self.pair_manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_PAIR_FIELDS - fields)
            if missing:
                raise ValueError(f"pair manifest is missing fields: {missing}")
            rows = list(reader)

        if not rows:
            raise ValueError("pair manifest is empty")
        if not allow_test and any(row["split"] == "test" for row in rows):
            raise PermissionError("a development pair manifest must not contain test rows")

        records: list[FatigueVideoCanRecord] = []
        sample_ids: set[str] = set()
        video_sample_ids: set[str] = set()
        can_sample_ids: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            if row["split"] != split:
                continue
            record = self._parse_record(row, row_number=row_number)
            for value, seen, field in (
                (record.sample_id, sample_ids, "paired_sample_id"),
                (record.video_sample_id, video_sample_ids, "video_sample_id"),
                (record.can_sample_id, can_sample_ids, "can_sample_id"),
            ):
                if value in seen:
                    raise ValueError(f"duplicate {field}: {value}")
                seen.add(value)
            records.append(record)

        if not records:
            raise ValueError(f"no complete paired windows found for split {split!r}")
        records.sort(
            key=lambda record: (
                record.subject_id,
                record.session_id,
                record.window_start_ms,
            )
        )
        self._validate_complete_parents(records)
        self._validate_zip_members(records)
        self.records = tuple(records)

    def _parse_record(
        self, row: Mapping[str, str], *, row_number: int
    ) -> FatigueVideoCanRecord:
        subject = row["subject_id"].strip()
        if ULDD_SPLIT_BY_SUBJECT.get(subject) != self.split:
            raise ValueError(
                f"subject {subject!r} does not belong to split {self.split!r}"
            )
        session = row["session_id"].strip()
        if not session or not session.startswith(f"{subject}_"):
            raise ValueError(f"invalid session_id at pair row {row_number}")
        if not parse_bool(row["paired_valid"]):
            raise ValueError(f"pair row {row_number} is not paired_valid")
        if not parse_bool(row["complete8_eligible"]):
            raise ValueError(f"pair row {row_number} is not complete8_eligible")
        if row["exclude_reason"].strip() or row["complete8_exclude_reason"].strip():
            raise ValueError(f"eligible pair row {row_number} contains an exclusion reason")
        if row["sync_status"] != "provider_documented_aligned":
            raise ValueError(f"pair row {row_number} has an unapproved sync status")
        if int(row["offset_ms"]) != 0:
            raise ValueError(f"pair row {row_number} must use the frozen zero offset")
        if parse_bool(row["hardware_clock_verified"]):
            raise ValueError(
                "UL-DD v1 must not claim hardware-clock verification"
            )

        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])
        label_start_ms = int(row["label_start_ms"])
        label_end_ms = int(row["label_end_ms"])
        window_index = int(row["window_index"])
        if int(row["duration_ms"]) != FATIGUE_WINDOW_MS or end_ms - start_ms != FATIGUE_WINDOW_MS:
            raise ValueError(f"pair row {row_number} is not a 30-second window")
        if label_end_ms - label_start_ms != FATIGUE_PARENT_MS:
            raise ValueError(f"pair row {row_number} is not inside a 240-second parent")
        if window_index not in range(8):
            raise ValueError(f"window_index must be 0..7 at pair row {row_number}")
        expected_start = label_start_ms + window_index * FATIGUE_WINDOW_MS
        if start_ms != expected_start or end_ms != expected_start + FATIGUE_WINDOW_MS:
            raise ValueError(f"window time/index mismatch at pair row {row_number}")

        score = float(row["kss_score"])
        label_id = int(row["label_id"])
        label_class = row["label_class"]
        if kss_label(score) != (label_id, label_class):
            raise ValueError(f"KSS mapping mismatch at pair row {row_number}")
        if label_id not in range(len(KSS_CLASSES)):
            raise ValueError(f"invalid label_id at pair row {row_number}")

        sample_id = row["paired_sample_id"].strip()
        if sample_id != canonical_pair_id(row):
            raise ValueError(f"non-canonical paired_sample_id at pair row {row_number}")
        if row["can_sample_id"].strip() != sample_id:
            raise ValueError(f"CAN and paired sample IDs differ at pair row {row_number}")
        condition = session.rsplit("_", 1)[-1]
        expected_parent = (
            f"ULDD_{subject}_{condition}_{label_start_ms:09d}_{label_end_ms:09d}"
        )
        if row["parent_id"].strip() != expected_parent:
            raise ValueError(f"non-canonical parent_id at pair row {row_number}")

        video_shape = _parse_shape(row["video_feature_shape"], field="video_feature_shape")
        can_shape = _parse_shape(row["can_feature_shape"], field="can_feature_shape")
        if video_shape != (6, 96):
            raise ValueError(f"video_feature_shape must be [6,96] at pair row {row_number}")
        if can_shape != (300, 9):
            raise ValueError(f"can_feature_shape must be [300,9] at pair row {row_number}")
        if row["video_feature_dtype"] != "float32" or row["can_feature_dtype"] != "float32":
            raise ValueError(f"feature dtype must be float32 at pair row {row_number}")
        if row["video_source_time_reference"] != "session_relative":
            raise ValueError(f"unexpected video time reference at pair row {row_number}")
        if row["video_time_transform"] != "subtract_window_start_ms_div_1000":
            raise ValueError(f"unexpected video time transform at pair row {row_number}")

        subwindow_ids = tuple(
            value for value in row["video_subwindow_ids"].split("|") if value
        )
        if len(subwindow_ids) != 6 or len(set(subwindow_ids)) != 6:
            raise ValueError(f"pair row {row_number} must name six unique video subwindows")
        valid_ratio = float(row["can_valid_ratio"])
        if not 0.95 <= valid_ratio <= 1.0:
            raise ValueError(f"CAN valid ratio is outside [0.95,1] at pair row {row_number}")

        video_archive = _safe_relative_path(
            self.dataset_root,
            row["video_feature_archive"],
            field="video_feature_archive",
        )
        can_feature = _safe_relative_path(
            self.dataset_root,
            row["can_feature_path"],
            field="can_feature_path",
        )
        if not video_archive.is_file():
            raise FileNotFoundError(
                f"video feature archive does not exist: {row['video_feature_archive']}"
            )
        if not can_feature.is_file():
            raise FileNotFoundError(
                f"CAN feature does not exist: {row['can_feature_path']}"
            )

        return FatigueVideoCanRecord(
            sample_id=sample_id,
            parent_id=expected_parent,
            subject_id=subject,
            session_id=session,
            split=self.split,
            window_index=window_index,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            label_start_ms=label_start_ms,
            label_end_ms=label_end_ms,
            kss_score=score,
            label_class=label_class,
            label_id=label_id,
            video_sample_id=row["video_sample_id"].strip(),
            video_subwindow_ids=subwindow_ids,
            video_archive_path=video_archive,
            video_feature_member=_safe_zip_member(row["video_feature_member"]),
            video_feature_shape=video_shape,
            video_feature_sha256=_validate_sha256(
                row["video_feature_sha256"], field="video_feature_sha256"
            ),
            can_sample_id=row["can_sample_id"].strip(),
            can_feature_path=can_feature,
            can_feature_shape=can_shape,
            can_feature_sha256=_validate_sha256(
                row["can_feature_sha256"], field="can_feature_sha256"
            ),
            can_valid_ratio=valid_ratio,
        )

    @staticmethod
    def _validate_complete_parents(records: Sequence[FatigueVideoCanRecord]) -> None:
        grouped: dict[str, list[FatigueVideoCanRecord]] = defaultdict(list)
        for record in records:
            grouped[record.parent_id].append(record)
        for parent_id, group in grouped.items():
            ordered = sorted(group, key=lambda record: record.window_index)
            if len(ordered) != 8 or [record.window_index for record in ordered] != list(range(8)):
                raise ValueError(f"parent {parent_id} must contain window indices 0..7 once")
            if len({record.label_id for record in ordered}) != 1:
                raise ValueError(f"parent {parent_id} contains conflicting labels")
            if len({record.subject_id for record in ordered}) != 1 or len(
                {record.session_id for record in ordered}
            ) != 1:
                raise ValueError(f"parent {parent_id} crosses subject or session")

    @staticmethod
    def _validate_zip_members(records: Sequence[FatigueVideoCanRecord]) -> None:
        expected: dict[Path, set[str]] = defaultdict(set)
        for record in records:
            expected[record.video_archive_path].add(record.video_feature_member)
        for archive_path, expected_members in expected.items():
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    actual_members = set(archive.namelist())
            except zipfile.BadZipFile as exc:
                raise ValueError(f"invalid video feature archive: {archive_path.name}") from exc
            missing = sorted(expected_members - actual_members)
            if missing:
                raise FileNotFoundError(
                    f"video archive is missing {len(missing)} required members; first={missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def parent_count(self) -> int:
        return len({record.parent_id for record in self.records})

    def _load_arrays(self, index: int) -> dict[str, dict[str, np.ndarray]]:
        if index in self._feature_cache:
            return self._feature_cache[index]
        record = self.records[index]
        with zipfile.ZipFile(record.video_archive_path) as archive:
            try:
                video_payload = archive.read(record.video_feature_member)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"video member disappeared: {record.video_feature_member}"
                ) from exc
        if self.verify_feature_hashes:
            digest = hashlib.sha256(video_payload).hexdigest()
            if digest != record.video_feature_sha256:
                raise ValueError(f"video feature hash mismatch for {record.sample_id}")
            if file_sha256(record.can_feature_path) != record.can_feature_sha256:
                raise ValueError(f"CAN feature hash mismatch for {record.sample_id}")

        video = self._read_npz(
            io.BytesIO(video_payload),
            sample_id=record.sample_id,
            modality="video",
            expected_x_shape=record.video_feature_shape,
        )
        can = self._read_npz(
            record.can_feature_path,
            sample_id=record.sample_id,
            modality="can",
            expected_x_shape=record.can_feature_shape,
        )
        video["time_s"], video["support_s"] = normalize_video_time_arrays(
            video["time_s"],
            video["support_s"],
            window_start_ms=record.window_start_ms,
        )
        self._validate_time_and_quality(video, record=record, modality="video")
        self._validate_time_and_quality(can, record=record, modality="can")
        if not np.isclose(
            float(can["valid_mask"].mean()),
            record.can_valid_ratio,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(f"CAN valid ratio differs from manifest for {record.sample_id}")

        result = {"video": video, "can": can}
        if self.cache_features:
            self._feature_cache[index] = result
        return result

    @staticmethod
    def _read_npz(
        source: Any,
        *,
        sample_id: str,
        modality: str,
        expected_x_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != set(FEATURE_ARRAY_FIELDS):
                raise ValueError(
                    f"{sample_id}:{modality} must contain exactly {sorted(FEATURE_ARRAY_FIELDS)}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy() for name in FEATURE_ARRAY_FIELDS
            }
        token_count = expected_x_shape[0]
        expected_shapes = {
            "x": expected_x_shape,
            "time_s": (token_count,),
            "valid_mask": (token_count,),
            "support_s": (token_count, 2),
            "observed_fraction": (token_count,),
        }
        for name, expected_dtype in CAN_ARRAY_DTYPES.items():
            value = arrays[name]
            if value.dtype != expected_dtype or value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{sample_id}:{modality}:{name} expected "
                    f"{expected_dtype}{expected_shapes[name]}, got {value.dtype}{value.shape}"
                )
            if value.dtype.kind in "f" and not np.isfinite(value).all():
                raise ValueError(f"{sample_id}:{modality}:{name} contains NaN or Inf")
        return arrays

    @staticmethod
    def _validate_time_and_quality(
        arrays: Mapping[str, np.ndarray],
        *,
        record: FatigueVideoCanRecord,
        modality: str,
    ) -> None:
        errors = validate_normalized_video_time(
            arrays["time_s"], arrays["support_s"], duration_s=30.0
        )
        if errors:
            raise ValueError(
                f"{record.sample_id}:{modality} invalid sample-relative time: {errors}"
            )
        if not arrays["valid_mask"].any():
            raise ValueError(f"{record.sample_id}:{modality} has no valid token")
        observed = arrays["observed_fraction"]
        if not np.all((observed >= 0.0) & (observed <= 1.0)):
            raise ValueError(
                f"{record.sample_id}:{modality} observed_fraction must be within [0,1]"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        arrays = self._load_arrays(index)
        inputs = {
            modality: {
                name: torch.from_numpy(arrays[modality][name].copy())
                for name in FEATURE_ARRAY_FIELDS
            }
            for modality in MODALITIES
        }
        return {
            "inputs": inputs,
            "label": torch.tensor(record.label_id, dtype=torch.int64),
            "sample_id": record.sample_id,
            "parent_id": record.parent_id,
            "subject_id": record.subject_id,
            "session_id": record.session_id,
            "window_index": record.window_index,
            "window_start_ms": record.window_start_ms,
            "window_end_ms": record.window_end_ms,
            "split": record.split,
            "video_sample_id": record.video_sample_id,
            "can_sample_id": record.can_sample_id,
        }


def _collate_stream(
    samples: Sequence[dict[str, Any]],
    *,
    modality: str,
    standardizer: Any | None,
) -> dict[str, torch.Tensor]:
    streams = [sample["inputs"][modality] for sample in samples]
    feature_dim = int(streams[0]["x"].shape[1])
    max_tokens = max(int(stream["x"].shape[0]) for stream in streams)
    batch_size = len(streams)
    x = torch.zeros((batch_size, max_tokens, feature_dim), dtype=torch.float32)
    time_s = torch.zeros((batch_size, max_tokens), dtype=torch.float64)
    valid_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    support_s = torch.zeros((batch_size, max_tokens, 2), dtype=torch.float64)
    observed_fraction = torch.zeros((batch_size, max_tokens), dtype=torch.float32)

    for batch_index, stream in enumerate(streams):
        if set(stream) != set(FEATURE_ARRAY_FIELDS):
            raise ValueError(f"{modality} stream must contain the five standard arrays")
        tokens = int(stream["x"].shape[0])
        expected_shapes = {
            "x": (tokens, feature_dim),
            "time_s": (tokens,),
            "valid_mask": (tokens,),
            "support_s": (tokens, 2),
            "observed_fraction": (tokens,),
        }
        expected_dtypes = {
            "x": torch.float32,
            "time_s": torch.float64,
            "valid_mask": torch.bool,
            "support_s": torch.float64,
            "observed_fraction": torch.float32,
        }
        for name in FEATURE_ARRAY_FIELDS:
            if stream[name].shape != expected_shapes[name] or stream[name].dtype != expected_dtypes[name]:
                raise ValueError(f"invalid {modality}.{name} shape or dtype")
        x[batch_index, :tokens] = stream["x"]
        time_s[batch_index, :tokens] = stream["time_s"]
        valid_mask[batch_index, :tokens] = stream["valid_mask"]
        support_s[batch_index, :tokens] = stream["support_s"]
        observed_fraction[batch_index, :tokens] = stream["observed_fraction"]

    if (~valid_mask.any(dim=1)).any():
        raise ValueError(f"all-invalid {modality} samples are forbidden")
    if standardizer is None:
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    else:
        x = standardizer.transform_tensor(x, valid_mask)
    return {
        "x": x,
        "time_s": time_s,
        "valid_mask": valid_mask,
        "support_s": support_s,
        "observed_fraction": observed_fraction,
    }


def collate_fatigue_video_can_batch(
    samples: Sequence[dict[str, Any]],
    *,
    standardizers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Right-pad modalities independently and retain all audit arrays."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    standardizers = {} if standardizers is None else dict(standardizers)
    unknown = set(standardizers) - set(MODALITIES)
    if unknown:
        raise ValueError(f"standardizers contain unknown modalities: {sorted(unknown)}")
    if any(set(sample.get("inputs", {})) != set(MODALITIES) for sample in samples):
        raise ValueError("each paired sample must contain exactly video and can")

    inputs = {
        modality: _collate_stream(
            samples,
            modality=modality,
            standardizer=standardizers.get(modality),
        )
        for modality in MODALITIES
    }
    return {
        "inputs": inputs,
        "labels": torch.stack([sample["label"] for sample in samples]),
        "sample_id": [sample["sample_id"] for sample in samples],
        "parent_id": [sample["parent_id"] for sample in samples],
        "subject_id": [sample["subject_id"] for sample in samples],
        "session_id": [sample["session_id"] for sample in samples],
        "window_index": torch.tensor(
            [sample["window_index"] for sample in samples], dtype=torch.int64
        ),
        "window_start_ms": torch.tensor(
            [sample["window_start_ms"] for sample in samples], dtype=torch.int64
        ),
        "window_end_ms": torch.tensor(
            [sample["window_end_ms"] for sample in samples], dtype=torch.int64
        ),
        "split": [sample["split"] for sample in samples],
        "video_sample_id": [sample["video_sample_id"] for sample in samples],
        "can_sample_id": [sample["can_sample_id"] for sample in samples],
    }


def make_fatigue_video_can_collate(
    standardizers: Mapping[str, Any] | None = None,
) -> Callable:
    return partial(
        collate_fatigue_video_can_batch,
        standardizers=standardizers,
    )


def make_fusion_model_inputs(
    batch: Mapping[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Select the exact ``x/valid_mask/time_s`` model input contract."""
    if not isinstance(batch, Mapping) or "inputs" not in batch:
        raise ValueError("batch must contain an inputs mapping")
    inputs = batch["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != set(MODALITIES):
        raise ValueError("batch inputs must contain exactly video and can")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for modality in MODALITIES:
        stream = inputs[modality]
        if not isinstance(stream, Mapping):
            raise ValueError(f"{modality} input must be a mapping")
        missing = set(MODEL_INPUT_FIELDS) - set(stream)
        if missing:
            raise ValueError(f"{modality} input is missing {sorted(missing)}")
        result[modality] = {
            name: stream[name] for name in MODEL_INPUT_FIELDS
        }
    return result
