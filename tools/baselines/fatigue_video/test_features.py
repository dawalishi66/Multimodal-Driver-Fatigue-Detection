"""Export and validate test video handoff features without model evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


TEST_SUBJECTS = {"C", "H", "P"}
ARRAY_FIELDS = {"x", "time_s", "valid_mask", "support_s", "observed_fraction"}
REQUEST_FIELDS = (
    "sample_id",
    "subject_id",
    "session_id",
    "window_start_ms",
    "window_end_ms",
    "source_feature",
    "source_time_reference",
)
INDEX_FIELDS = (
    "sample_id",
    "subject_id",
    "session_id",
    "window_start_ms",
    "window_end_ms",
    "feature_path",
    "feature_sha256",
    "feature_shape",
    "feature_dtype",
)
FORBIDDEN_COLUMN_PARTS = ("label", "kss", "prob", "prediction", "metric")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        return tuple(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields))
        writer.writeheader()
        writer.writerows(rows)


def verify_unlock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "scope": "test_feature_generation_and_validation_only",
        "allow_model_evaluation": False,
        "allow_threshold_or_model_changes": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"unlock.{key} must be {expected!r}")
    if not str(value.get("authorized_by", "")).strip():
        raise ValueError("unlock.authorized_by is required")
    if not str(value.get("issued_at", "")).strip():
        raise ValueError("unlock.issued_at is required")
    return value


def read_request(path: Path) -> list[dict[str, str]]:
    fields, rows = _read_csv(path)
    if fields != REQUEST_FIELDS:
        forbidden = [
            field for field in fields
            if any(part in field.lower() for part in FORBIDDEN_COLUMN_PARTS)
        ]
        if forbidden:
            raise ValueError(f"request contains forbidden scoring fields: {forbidden}")
        raise ValueError(f"request columns must be exactly {list(REQUEST_FIELDS)}")
    if not rows:
        raise ValueError("test feature request is empty")
    seen: set[str] = set()
    subjects: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        subject = row["subject_id"]
        start = int(row["window_start_ms"])
        end = int(row["window_end_ms"])
        if sample_id in seen or re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id) is None:
            raise ValueError(f"duplicate or unsafe sample_id: {sample_id!r}")
        if subject not in TEST_SUBJECTS:
            raise ValueError(f"non-test subject in request: {subject}")
        if not row["session_id"].startswith(f"{subject}_"):
            raise ValueError(f"subject/session mismatch: {sample_id}")
        if end - start != 30_000 or start < 0:
            raise ValueError(f"invalid 30-second interval: {sample_id}")
        _safe_relative(row["source_feature"])
        if row["source_time_reference"] not in {"session_relative", "sample_relative"}:
            raise ValueError(f"invalid source time reference: {sample_id}")
        seen.add(sample_id)
        subjects.add(subject)
    if subjects != TEST_SUBJECTS:
        raise ValueError(f"request subjects must be exactly C/H/P, got {sorted(subjects)}")
    return rows


def _load_and_normalize(source: Path, row: Mapping[str, str]) -> dict[str, np.ndarray]:
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_FIELDS:
            raise ValueError(f"{row['sample_id']}: NPZ fields differ from the five-array contract")
        arrays = {name: np.asarray(archive[name]).copy() for name in ARRAY_FIELDS}
    expected = {
        "x": (np.dtype("float32"), (6, 96)),
        "time_s": (np.dtype("float64"), (6,)),
        "valid_mask": (np.dtype("bool"), (6,)),
        "support_s": (np.dtype("float64"), (6, 2)),
        "observed_fraction": (np.dtype("float32"), (6,)),
    }
    for name, (dtype, shape) in expected.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError(f"{row['sample_id']}:{name} expected {dtype}{shape}")
        if arrays[name].dtype.kind == "f" and not np.isfinite(arrays[name]).all():
            raise ValueError(f"{row['sample_id']}:{name} contains NaN or Inf")
    if row["source_time_reference"] == "session_relative":
        offset = int(row["window_start_ms"]) / 1000.0
        arrays["time_s"] -= offset
        arrays["support_s"] -= offset
    return arrays


def _validate_arrays(arrays: Mapping[str, np.ndarray], *, sample_id: str) -> None:
    time_s = arrays["time_s"]
    support_s = arrays["support_s"]
    valid_mask = arrays["valid_mask"]
    observed = arrays["observed_fraction"]
    if not valid_mask.any():
        raise ValueError(f"{sample_id}: all video tokens are invalid")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError(f"{sample_id}: time_s is not strictly increasing")
    if not (
        np.all(support_s[:, 0] >= 0.0)
        and np.all(support_s[:, 1] <= 30.0)
        and np.all(support_s[:, 1] > support_s[:, 0])
        and np.all(time_s > support_s[:, 0])
        and np.all(time_s < support_s[:, 1])
    ):
        raise ValueError(f"{sample_id}: time/support is outside sample-relative 0..30s")
    if not np.all((observed >= 0.0) & (observed <= 1.0)):
        raise ValueError(f"{sample_id}: observed_fraction is outside [0,1]")


def export_features(
    request_path: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    rows = read_request(request_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fatigue_video_test_", dir=output_root.parent) as name:
        temporary = Path(name)
        features = temporary / "features_30s"
        features.mkdir()
        index_rows: list[dict[str, Any]] = []
        for row in rows:
            source = (source_root / _safe_relative(row["source_feature"])).resolve()
            try:
                source.relative_to(source_root.resolve())
            except ValueError as exc:
                raise ValueError("source feature escapes source_root") from exc
            arrays = _load_and_normalize(source, row)
            _validate_arrays(arrays, sample_id=row["sample_id"])
            relative = Path("features_30s") / f"{row['sample_id']}.npz"
            target = temporary / relative
            np.savez_compressed(target, **arrays)
            index_rows.append({
                **{field: row[field] for field in REQUEST_FIELDS[:5]},
                "feature_path": relative.as_posix(),
                "feature_sha256": _sha256(target),
                "feature_shape": "[6,96]",
                "feature_dtype": "float32",
            })
        _write_csv(temporary / "test_video_feature_index.csv", INDEX_FIELDS, index_rows)
        report = validate_package(temporary)
        (temporary / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(temporary), str(output_root))
    return report


def validate_package(root: Path) -> dict[str, Any]:
    fields, rows = _read_csv(root / "test_video_feature_index.csv")
    if fields != INDEX_FIELDS or not rows:
        raise ValueError("invalid or empty test feature index")
    subjects: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen or row["subject_id"] not in TEST_SUBJECTS:
            raise ValueError("duplicate ID or non-test subject in feature index")
        path = root / _safe_relative(row["feature_path"])
        if _sha256(path) != row["feature_sha256"]:
            raise ValueError(f"feature hash mismatch: {sample_id}")
        arrays = _load_and_normalize(path, {**row, "source_time_reference": "sample_relative"})
        _validate_arrays(arrays, sample_id=sample_id)
        if row["feature_shape"] != "[6,96]" or row["feature_dtype"] != "float32":
            raise ValueError(f"feature declaration mismatch: {sample_id}")
        subjects.add(row["subject_id"])
        seen.add(sample_id)
    if subjects != TEST_SUBJECTS:
        raise ValueError("feature package must cover exactly C/H/P")
    return {
        "status": "PASS",
        "scope": "test_feature_generation_and_validation_only",
        "subjects": sorted(subjects),
        "feature_count": len(rows),
        "feature_shape": [6, 96],
        "time_reference": "sample_relative_0_to_30_seconds",
        "model_loaded": False,
        "predictions_generated": False,
        "metrics_evaluated": False,
        "thresholds_or_models_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "validate"))
    parser.add_argument("--unlock", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify_unlock(args.unlock.resolve())
        if args.command == "export":
            if args.request is None or args.source_root is None:
                raise ValueError("export requires --request and --source-root")
            report = export_features(
                args.request.resolve(), args.source_root.resolve(), args.output_root.resolve()
            )
        else:
            report = validate_package(args.output_root.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
