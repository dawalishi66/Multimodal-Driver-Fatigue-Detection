"""Validate a single-modality CSV and its NPZ features without training.

PASS means structural checks only, not provenance, synchronization, complete
cohorts, video subwindow lineage, or test-use auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from driver_state.constants import (
    DCPT_CLASSES, DCPT_WINDOW_MS, FATIGUE_PARENT_MS, FATIGUE_STRIDE_MS,
    FATIGUE_WINDOW_MS, SCHEMA_VERSION, SPLITS, TASK_MODALITIES,
    ULDD_SPLIT_BY_SUBJECT, kss_label,
)
from driver_state.schemas import FEATURE_DTYPES, REQUIRED_FIELDS

LIMITATIONS = [
    "Does not prove source identity or cross-modal synchronization.",
    "Does not audit missing candidate rows, video subwindow lineage or complete-8 cohorts.",
    "Does not prove train-only preprocessing or correct test-access history.",
    "Does not replace modality-specific quality and provenance acceptance.",
]


def _integer(value: str, field: str) -> int:
    if not re.fullmatch(r"-?\d+", value):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _boolean(value: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError("valid must be true/false or 1/0")


def _relative_path(value: str, root: Path) -> Path:
    """Reject absolute/drive/traversal paths, including Windows paths on Linux."""
    windows = PureWindowsPath(value)
    if not value or windows.drive or windows.root or Path(value).is_absolute():
        raise ValueError("path must be a nonempty relative path")
    path = Path(value.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path escapes the configured root")
    return resolved


def covered_seconds(support_s: np.ndarray, valid_mask: np.ndarray) -> float:
    """Union of usable support intervals; overlapping time counts only once."""
    intervals = sorted((float(a), float(b)) for a, b in support_s[valid_mask])
    if not intervals:
        return 0.0
    total = 0.0
    left, right = intervals[0]
    for start, end in intervals[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def _check_feature(row: dict[str, str], root: Path, duration_s: float) -> float:
    path = _relative_path(row["feature_path"], root)
    if path.suffix.lower() != ".npz":
        raise ValueError("feature_path must reference an NPZ file")
    mask_path, separator, mask_name = row["mask"].rpartition("::")
    if not separator or mask_name != "valid_mask":
        raise ValueError("mask must use relative.npz::valid_mask")
    if _relative_path(mask_path, root) != path:
        raise ValueError("mask must reference the same NPZ as feature_path")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(FEATURE_DTYPES):
            raise ValueError("NPZ must contain exactly the five documented arrays")
        arrays = {name: archive[name] for name in FEATURE_DTYPES}
    for name, dtype in FEATURE_DTYPES.items():
        if arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"{name} dtype must be {dtype}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains NaN or Inf")
    x = arrays["x"]
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("x must have nonempty shape [T,D]")
    length = x.shape[0]
    expected_shapes = {
        "time_s": (length,), "valid_mask": (length,),
        "support_s": (length, 2), "observed_fraction": (length,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
    declared_shape = json.loads(row["feature_shape"])
    if not isinstance(declared_shape, list) or any(type(v) is not int for v in declared_shape):
        raise ValueError("feature_shape must be a JSON array of integers")
    if declared_shape != list(x.shape):
        raise ValueError("feature_shape does not match x")
    if row["feature_dtype"] != str(x.dtype):
        raise ValueError("feature_dtype does not match x")
    times, supports = arrays["time_s"], arrays["support_s"]
    if np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if (np.any(supports[:, 0] < 0) or np.any(supports[:, 1] > duration_s + 1e-9)
            or np.any(supports[:, 1] <= supports[:, 0])):
        raise ValueError("support_s must contain positive intervals within the sample")
    if np.any(times < supports[:, 0]) or np.any(times >= supports[:, 1]):
        raise ValueError("time_s must lie inside its own support interval")
    fractions = arrays["observed_fraction"]
    if np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("observed_fraction must be in [0,1]")
    return covered_seconds(supports, arrays["valid_mask"]) / duration_s


def validate_metadata(
    metadata: Path | str,
    *,
    task: str,
    feature_root: Path | str,
    min_valid_ratio: float = 0.95,
) -> dict[str, Any]:
    """Return a JSON-serializable report; 0.95 is a preflight quality default."""
    if task not in REQUIRED_FIELDS:
        raise ValueError(f"unsupported task: {task}")
    if not math.isfinite(min_valid_ratio) or not 0 <= min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio must be in [0,1]")
    report: dict[str, Any] = {
        "status": "FAIL", "scope": "structure_and_features_only",
        "schema_version": SCHEMA_VERSION, "task": task,
        "row_count": 0, "checked_feature_count": 0,
        "min_valid_ratio": min_valid_ratio, "errors": [],
        "warnings": [], "limitations": list(LIMITATIONS),
    }

    def error(code: str, message: str, row_number: int | None = None) -> None:
        report["errors"].append({"code": code, "row": row_number, "message": message})

    try:
        with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)):
                error("DUPLICATE_COLUMNS", "CSV header contains duplicate columns")
            missing = sorted(set(REQUIRED_FIELDS[task]) - set(fields))
            if missing:
                error("MISSING_COLUMNS", f"Missing columns: {missing}")
                return report
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        error("CSV_READ_FAILED", "Cannot read metadata as a UTF-8 CSV")
        return report
    report["row_count"] = len(rows)
    if not rows:
        error("EMPTY_METADATA", "CSV contains no data rows")
    seen_ids: set[str] = set()
    seen_windows: set[tuple[str, str, int]] = set()
    parent_scores: dict[tuple[str, str, int], float] = {}
    subject_splits: dict[str, str] = {}
    modalities: set[str] = set()
    root = Path(feature_root)
    for row_number, raw in enumerate(rows, start=2):
        if None in raw or any(raw.get(name) is None for name in REQUIRED_FIELDS[task]):
            error("MALFORMED_ROW", "CSV row length does not match the header", row_number)
            continue
        row = {key: value.strip() if isinstance(value, str) else value for key, value in raw.items()}
        try:
            for field in ("sample_id", "subject_id", "session_id", "source_file", "extractor_name", "extractor_version"):
                if not row[field] or row[field].lower() in ("latest", "to_be_filled"):
                    raise ValueError(f"{field} must be nonempty and versioned where applicable")
            if row["sample_id"] in seen_ids:
                error("DUPLICATE_SAMPLE_ID", "sample_id is repeated in this modality CSV", row_number)
            seen_ids.add(row["sample_id"])
            modality, subject, split = row["modality"], row["subject_id"], row["split"]
            if modality not in TASK_MODALITIES[task]:
                raise ValueError("modality does not belong to the selected task")
            modalities.add(modality)
            if split not in SPLITS:
                raise ValueError("split must be train, val or test")
            if subject in subject_splits and subject_splits[subject] != split:
                error("SUBJECT_SPLIT_LEAKAGE", "one subject occurs in multiple splits", row_number)
            subject_splits[subject] = split
            if task == "fatigue" and ULDD_SPLIT_BY_SUBJECT.get(subject) != split:
                error("WRONG_ULDD_SPLIT", "subject is absent from or disagrees with the fixed UL-DD split", row_number)
            if task == "distraction" and not re.fullmatch(r"P(?:0[1-9]|[1-3][0-9]|40)", subject):
                raise ValueError("DCPT subject_id must be P01 through P40")
            sources = json.loads(row["source_file"]) if row["source_file"].startswith("[") else [row["source_file"]]
            if not isinstance(sources, list) or not sources or any(not isinstance(s, str) for s in sources):
                raise ValueError("source_file must be a relative resource or a JSON list of resources")
            for source in sources:
                _relative_path(source, root)
            start = _integer(row["window_start_ms"], "window_start_ms")
            end = _integer(row["window_end_ms"], "window_end_ms")
            duration = _integer(row["duration_ms"], "duration_ms")
            index = _integer(row["window_index"], "window_index")
            label_id = _integer(row["label_id"], "label_id")
            if start < 0 or duration <= 0 or end - start != duration:
                raise ValueError("window times must have consistent, positive-duration integer milliseconds")
            if task == "fatigue":
                label_start = _integer(row["label_start_ms"], "label_start_ms")
                label_end = _integer(row["label_end_ms"], "label_end_ms")
                if duration != FATIGUE_WINDOW_MS or start % FATIGUE_STRIDE_MS:
                    raise ValueError("fatigue windows must be 30 seconds on the acquisition-zero 30-second grid")
                if (label_start < 0 or label_start % FATIGUE_PARENT_MS
                        or label_end - label_start != FATIGUE_PARENT_MS
                        or not label_start <= start < end <= label_end):
                    raise ValueError("fatigue window crosses or disagrees with its 240-second label interval")
                if index != (start - label_start) // FATIGUE_WINDOW_MS:
                    raise ValueError("window_index must be 0-7 within its label interval")
                score = float(row["kss_score"])
                expected_id, expected_class = kss_label(score)
                if (label_id, row["label_class"]) != (expected_id, expected_class):
                    raise ValueError("KSS, label_class and label_id disagree")
                parent_key = (subject, row["session_id"], label_start)
                if parent_key in parent_scores and parent_scores[parent_key] != score:
                    error("PARENT_LABEL_CONFLICT", "one parent interval contains different KSS values", row_number)
                parent_scores[parent_key] = score
                window_key = (subject, row["session_id"], start)
                if window_key in seen_windows:
                    error("DUPLICATE_WINDOW", "same subject/session/time occurs twice", row_number)
                seen_windows.add(window_key)
            else:
                if (start, end, duration, index) != (0, DCPT_WINDOW_MS, DCPT_WINDOW_MS, 0):
                    raise ValueError("DCPT uses one nominal [0,10000) ms clip; actual coverage belongs in features/QC")
                if not 0 <= label_id < len(DCPT_CLASSES) or row["label_class"] != DCPT_CLASSES[label_id]:
                    raise ValueError("DCPT label_id/class must match the fixed nine-class order")
            valid = _boolean(row["valid"])
            ratio = float(row["valid_ratio"])
            if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError("valid_ratio must be finite and in [0,1]")
            if valid and row["error"]:
                raise ValueError("valid rows must have empty error")
            if not valid and not row["error"]:
                raise ValueError("invalid rows must retain an explicit error")
            if valid and ratio < min_valid_ratio:
                raise ValueError("valid row falls below the configured feature-coverage threshold")
            if not row["feature_path"]:
                if valid or ratio != 0 or row["mask"] or row["feature_shape"] or row["feature_dtype"]:
                    raise ValueError("missing features require invalid status, ratio=0 and empty feature descriptors")
            else:
                actual_ratio = _check_feature(row, root, duration / 1000)
                report["checked_feature_count"] += 1
                if not math.isclose(ratio, actual_ratio, rel_tol=0, abs_tol=1e-6):
                    raise ValueError("valid_ratio differs from the union of valid support_s intervals")
        except (ValueError, TypeError, KeyError, OSError, EOFError, zipfile.BadZipFile) as exc:
            message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            if isinstance(exc, (json.JSONDecodeError, UnicodeError)):
                message = "Invalid JSON or text encoding in a metadata/feature field"
            error("INVALID_ROW", message, row_number)
    if len(modalities) > 1:
        error("MIXED_MODALITIES", "validate each modality CSV independently")
    if task == "fatigue":
        session_starts: dict[tuple[str, str], list[int]] = {}
        for subject, session, start in seen_windows:
            session_starts.setdefault((subject, session), []).append(start)
        for starts in session_starts.values():
            ordered = sorted(starts)
            if any(right - left != FATIGUE_STRIDE_MS for left, right in zip(ordered, ordered[1:])):
                error("NONSTANDARD_STRIDE", "fatigue metadata has a gap within a session; retain failed rows on the 30-second grid")
    if task == "distraction":
        report["warnings"].append("DCPT fixed subject manifest is not checked; only within-file subject isolation is checked.")
    report["status"] = "FAIL" if report["errors"] else "PASS"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(REQUIRED_FIELDS))
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report and (args.report.suffix.lower() != ".json"
                        or args.report.resolve() == args.metadata.resolve()):
        parser.error("--report must be a JSON output path distinct from the input metadata")
    try:
        report = validate_metadata(args.metadata, task=args.task, feature_root=args.feature_root,
                                   min_valid_ratio=args.min_valid_ratio)
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
