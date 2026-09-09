"""6-class-aware structural validator for the video-aligned audio metadata.

The public ``driver_state.validation.metadata`` validator hard-codes the
nine-class DCPT order, so it cannot validate the 6-class fusion CSV. This
module mirrors its structural checks but validates labels against the shared
6c scheme (tasks 01/03/04/05/07/08) and, when a subject-splits file is given,
checks every row against that exact split. PASS means structural + feature
checks only; it does not prove cross-modal synchronization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from driver_state.constants import DCPT_WINDOW_MS, SPLITS
from driver_state.preprocessing.distraction_audio.build_audio_6c_metadata import (
    AUDIO_6C_CSV_FIELDS,
    ERROR_NO_FEATURES,
)
from driver_state.preprocessing.distraction_audio.fusion_labels import (
    SIX_CLASS_ID_TO_NAME,
    SIX_CLASS_NAMES,
    SIX_CLASS_TASK_TO_ID,
    load_label_scheme,
    load_subject_splits,
)
from driver_state.schemas import FEATURE_DTYPES
from driver_state.validation.metadata import covered_seconds

SUBJECT_RE = re.compile(r"P(?:0[1-9]|[1-3][0-9]|40)")
# Fusion alignment: session_id must mirror the distraction-video module, i.e.
# P<subject>_<date>_<HHMM>_<SS> (e.g. P01_20231111_0931_43).
SESSION_ID_RE = re.compile(r"P\d{2}_\d{8}_\d{4}_\d{2}")
MAX_RATIO_TOL = 1e-6


def _relative_path(value: str, root: Path) -> Path:
    if not value or Path(value).is_absolute():
        raise ValueError("path must be a nonempty relative path")
    path = Path(value.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path escapes the configured root")
    return resolved


def _check_feature(row: dict[str, str], root: Path, duration_s: float) -> float:
    path = _relative_path(row["feature_path"], root)
    if path.suffix.lower() != ".npz":
        raise ValueError("feature_path must reference an NPZ file")
    mask_path, sep, mask_name = row["mask"].rpartition("::")
    if not sep or mask_name != "valid_mask":
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
    expected = {
        "time_s": (length,), "valid_mask": (length,),
        "support_s": (length, 2), "observed_fraction": (length,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
    declared = json.loads(row["feature_shape"])
    if not isinstance(declared, list) or any(type(v) is not int for v in declared):
        raise ValueError("feature_shape must be a JSON array of integers")
    if declared != list(x.shape) or row["feature_dtype"] != str(x.dtype):
        raise ValueError("feature_shape/feature_dtype must match x")
    times = arrays["time_s"]
    supports = arrays["support_s"]
    if np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if np.any(supports[:, 0] < 0) or np.any(supports[:, 1] > duration_s + 1e-9) \
            or np.any(supports[:, 1] <= supports[:, 0]):
        raise ValueError("support_s must be positive intervals inside the sample")
    if np.any(times < supports[:, 0]) or np.any(times >= supports[:, 1]):
        raise ValueError("time_s must lie inside its own support interval")
    fractions = arrays["observed_fraction"]
    if np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("observed_fraction must be in [0,1]")
    return covered_seconds(supports, arrays["valid_mask"]) / duration_s


def validate_audio_6c(
    metadata: Path | str,
    *,
    label_scheme_path: Path | str,
    subject_splits_path: Path | str,
    feature_root: Path | str,
    min_valid_ratio: float = 0.95,
) -> dict[str, Any]:
    if not math.isfinite(min_valid_ratio) or not 0 <= min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio must be in [0,1]")
    label_scheme = load_label_scheme(label_scheme_path)
    splits = load_subject_splits(subject_splits_path)
    report: dict[str, Any] = {
        "status": "FAIL", "scope": "structure_and_features_only_6c",
        "schema_version": "0.2.0", "label_scheme": label_scheme.get("label_scheme"),
        "row_count": 0, "checked_feature_count": 0,
        "min_valid_ratio": min_valid_ratio, "errors": [], "warnings": [],
    }

    def error(code: str, message: str, row_number: int | None = None) -> None:
        report["errors"].append({"code": code, "row": row_number, "message": message})

    root = Path(feature_root)
    try:
        with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = list(reader.fieldnames or [])
            if len(fields) != len(set(fields)):
                error("DUPLICATE_COLUMNS", "CSV header contains duplicate columns")
            missing = sorted(set(AUDIO_6C_CSV_FIELDS) - set(fields))
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
    for row_number, raw in enumerate(rows, start=2):
        if any(raw.get(name) is None for name in AUDIO_6C_CSV_FIELDS):
            error("MALFORMED_ROW", "CSV row length does not match the header", row_number)
            continue
        row = {key: value.strip() if isinstance(value, str) else value for key, value in raw.items()}
        try:
            for field in ("sample_id", "subject_id", "session_id", "source_file",
                          "extractor_name", "extractor_version"):
                if not row[field] or row[field].lower() in ("latest", "to_be_filled"):
                    raise ValueError(f"{field} must be nonempty and versioned where applicable")
            if not SESSION_ID_RE.fullmatch(row["session_id"]):
                raise ValueError(
                    "session_id must mirror the video module: P<subject>_<date>_<HHMM>_<SS>")
            sources = (json.loads(row["source_file"])
                       if row["source_file"].startswith("[") else [row["source_file"]])
            if not isinstance(sources, list) or not sources \
                    or any(not isinstance(s, str) or not s for s in sources):
                raise ValueError("source_file must be a relative resource or a JSON list of resources")
            for source in sources:
                candidate = source.replace("\\", "/")
                if not candidate or Path(candidate).is_absolute() or ".." in candidate.split("/"):
                    raise ValueError("source_file entries must be nonempty relative resource paths")
                if Path(candidate).name.rsplit(".", 1)[0] != row["sample_id"]:
                    raise ValueError("source_file basename (minus extension) must equal sample_id")
            if row["sample_id"] in seen_ids:
                error("DUPLICATE_SAMPLE_ID", "sample_id is repeated", row_number)
            seen_ids.add(row["sample_id"])
            if row["modality"] != "audio":
                raise ValueError("modality must be audio")
            if not SUBJECT_RE.fullmatch(row["subject_id"]):
                raise ValueError("subject_id must be P01 through P40")
            split = row["split"]
            if split not in SPLITS:
                raise ValueError("split must be train, val or test")
            expected_split = splits.get(row["subject_id"], "")
            if not expected_split or split != expected_split:
                raise ValueError("split disagrees with the shared 6c subject splits")
            task_key = row["sample_id"][:2]
            if task_key not in SIX_CLASS_TASK_TO_ID:
                raise ValueError("sample_id task prefix is outside the 6c set")
            label_id = int(row["label_id"])
            if label_id != SIX_CLASS_TASK_TO_ID[task_key] \
                    or row["label_class"] != SIX_CLASS_ID_TO_NAME[label_id]:
                raise ValueError("label_id/class must match the shared 6c scheme")
            start = int(row["window_start_ms"]); end = int(row["window_end_ms"])
            duration = int(row["duration_ms"]); index = int(row["window_index"])
            if (start, end, duration, index) != (0, DCPT_WINDOW_MS, DCPT_WINDOW_MS, 0):
                raise ValueError("DCPT uses one nominal [0,10000) ms clip")
            valid_raw = row["valid"].lower()
            if valid_raw not in ("true", "1", "false", "0"):
                raise ValueError("valid must be true/false or 1/0")
            valid = valid_raw in ("true", "1")
            ratio = float(row["valid_ratio"])
            if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError("valid_ratio must be finite and in [0,1]")
            if valid and row["error"]:
                raise ValueError("valid rows must have empty error")
            if not valid and not row["error"]:
                raise ValueError("invalid rows must retain an explicit error")
            if valid and ratio < min_valid_ratio:
                raise ValueError("valid row falls below the feature-coverage threshold")
            if not row["feature_path"]:
                if valid or ratio != 0 or row["mask"] or row["feature_shape"] \
                        or row["feature_dtype"]:
                    raise ValueError("missing features require invalid, ratio=0 and empty descriptors")
            else:
                if not valid:
                    raise ValueError("feature_path on an invalid row is not allowed")
                actual = _check_feature(row, root, duration / 1000)
                report["checked_feature_count"] += 1
                if not math.isclose(ratio, actual, rel_tol=0, abs_tol=MAX_RATIO_TOL):
                    raise ValueError("valid_ratio differs from the union of valid support_s intervals")
        except (ValueError, TypeError, KeyError, OSError, EOFError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            error("INVALID_ROW", message, row_number)
    report["warnings"].append(
        "6c label scheme and 24/8/8 subject splits are provisional mirrors of the "
        "distraction-video module; official freeze by the project lead is pending.")
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--label-scheme", required=True, type=Path)
    parser.add_argument("--subject-splits", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_audio_6c(
            args.metadata,
            label_scheme_path=args.label_scheme,
            subject_splits_path=args.subject_splits,
            feature_root=args.feature_root,
            min_valid_ratio=args.min_valid_ratio,
        )
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
