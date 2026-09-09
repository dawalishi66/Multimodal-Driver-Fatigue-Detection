"""Structural validator for standard DCPT video metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STANDARD_ARRAYS = {
    "x": (2, np.dtype("float32")),
    "time_s": (1, np.dtype("float64")),
    "valid_mask": (1, np.dtype("bool")),
    "support_s": (2, np.dtype("float64")),
    "observed_fraction": (1, np.dtype("float32")),
}


def validate(csv_path: Path, feature_root: Path) -> dict:
    errors = []
    warnings = []
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample_id")
    for row in rows:
        if not row["subject_id"] or not row["session_id"]:
            errors.append(f"{row['sample_id']}: empty subject/session")
        if row["split"] not in {"train", "val", "test"}:
            errors.append(f"{row['sample_id']}: invalid split")
        if row["modality"] != "video":
            errors.append(f"{row['sample_id']}: modality must be video")
        if row["window_index"] != "0" or row["window_start_ms"] != "0" or row["window_end_ms"] != "10000":
            errors.append(f"{row['sample_id']}: DCPT window must be [0,10000)")
        if row["duration_ms"] != "10000":
            errors.append(f"{row['sample_id']}: duration_ms must be 10000")

        feature_path = feature_root / row["feature_path"]
        if row["valid"].lower() == "true":
            if not feature_path.exists():
                errors.append(f"{row['sample_id']}: missing feature {feature_path}")
                continue
            data = np.load(feature_path)
            if set(data.files) != set(STANDARD_ARRAYS):
                errors.append(f"{row['sample_id']}: NPZ arrays {sorted(data.files)}")
            for name, (ndim, dtype) in STANDARD_ARRAYS.items():
                if name not in data.files:
                    continue
                if data[name].ndim != ndim:
                    errors.append(f"{row['sample_id']}: {name} ndim {data[name].ndim}")
                if str(data[name].dtype) != str(dtype):
                    errors.append(f"{row['sample_id']}: {name} dtype {data[name].dtype}")
            if not np.isfinite(data["x"]).all():
                errors.append(f"{row['sample_id']}: non-finite x")
            if not np.isfinite(data["time_s"]).all() or not np.isfinite(data["observed_fraction"]).all():
                errors.append(f"{row['sample_id']}: non-finite metadata array")
            if np.any((data["observed_fraction"] < 0) | (data["observed_fraction"] > 1)):
                errors.append(f"{row['sample_id']}: observed_fraction out of range")
            if np.any(data["time_s"][1:] <= data["time_s"][:-1]):
                errors.append(f"{row['sample_id']}: time_s not strictly increasing")
            if np.any(data["support_s"][:, 0] >= data["support_s"][:, 1]):
                errors.append(f"{row['sample_id']}: support_s malformed")
            if np.any(data["time_s"] < data["support_s"][:, 0]) or np.any(data["time_s"] >= data["support_s"][:, 1]):
                errors.append(f"{row['sample_id']}: time_s outside support")
            expected_ratio = float(row["valid_ratio"])
            if abs(expected_ratio - float(data["valid_mask"].mean())) > 1e-5:
                errors.append(f"{row['sample_id']}: valid_ratio mismatch")
        elif not row["error"]:
            errors.append(f"{row['sample_id']}: invalid row must have error")

    subject_splits = {}
    for row in rows:
        subject_splits.setdefault(row["subject_id"], set()).add(row["split"])
    for subject_id, splits in subject_splits.items():
        if len(splits) > 1:
            errors.append(f"{subject_id}: subject appears in multiple splits")
    if not rows:
        errors.append("empty metadata")

    report = {
        "status": "FAIL" if errors else "PASS",
        "scope": "distraction_video",
        "schema_version": "0.2.0",
        "row_count": len(rows),
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "DCPT official subject split not yet available",
            "real audio-video synchronization is not checked here",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = validate(Path(args.csv), Path(args.feature_root))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
