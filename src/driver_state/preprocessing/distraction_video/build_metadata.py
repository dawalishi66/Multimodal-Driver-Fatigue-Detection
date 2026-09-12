"""Build the standard six-class DCPT video metadata CSV.

Input rows come from ``distraction_video/metadata/video_samples_v0_pre.jsonl``.
The label scheme and subject splits are explicit files so this module never
silently invents a class order or split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from driver_state.preprocessing.distraction_video.fusion_labels import (
    VIDEO_LABEL_SCHEME_NAME,
    load_label_scheme,
    load_subject_splits,
)
from driver_state.schemas import COMMON_METADATA_FIELDS

VIDEO_6C_CSV_FIELDS = (
    *COMMON_METADATA_FIELDS[:12],
    "label_scheme",
    *COMMON_METADATA_FIELDS[12:],
)

FEATURE_EXTRACTOR_NAME = "torchvision_r3d_18"
FEATURE_EXTRACTOR_VERSION = "0.24.1_kinetics400_v1"
FEATURE_LAYOUT = "video_features_v1"
ERROR_NO_FEATURES = "NO_FEATURES_YET"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_feature_index(path: str | Path | None) -> dict[str, str]:
    """Map sample_id -> NPZ path relative to the feature root."""
    if path is None:
        return {}
    index: dict[str, str] = {}
    for record in _read_jsonl(Path(path)):
        feature_path = str(record["path"]).replace("\\", "/")
        if feature_path.startswith("processed/"):
            feature_path = feature_path[len("processed/"):]
        index[record["sample_id"]] = feature_path
    return index


def coverage_ratio(valid_mask: np.ndarray, support_s: np.ndarray) -> float:
    """Compute the union coverage of usable two-dimensional supports."""
    intervals = sorted(
        (float(start), float(end))
        for mask, (start, end) in zip(valid_mask, support_s)
        if bool(mask)
    )
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
    return min(1.0, (total + right - left) / 10.0)


def build_6c_rows(
    manifest_rows: list[dict],
    label_scheme: dict[str, object],
    subject_splits: dict[str, str],
    feature_index: dict[str, str] | None = None,
    feature_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Build standard video metadata rows from a parsed source manifest."""
    features = feature_index or {}
    root = Path(feature_root) if feature_root is not None else None
    class_names = list(label_scheme["class_names"])
    task_to_class = dict(label_scheme["task_to_class"])
    rows: list[dict[str, str]] = []
    for manifest in sorted(manifest_rows, key=lambda item: item["sample_id"]):
        sample_id = manifest["sample_id"]
        task_key = sample_id[:2]
        if task_key not in task_to_class:
            continue
        label_id = int(task_to_class[task_key])
        feature_rel = features.get(sample_id, f"{FEATURE_LAYOUT}/{sample_id}.npz")
        feature_path = root / feature_rel if root is not None else None
        if feature_path is not None and feature_path.is_file():
            with np.load(feature_path, allow_pickle=False) as archive:
                x = archive["x"]
                valid_ratio = coverage_ratio(
                    archive["valid_mask"].astype(bool),
                    archive["support_s"],
                )
                shape = json.dumps([int(value) for value in x.shape])
                dtype = str(x.dtype)
            valid = valid_ratio >= 0.95
            error = "" if valid else "coverage_below_0.95"
        elif feature_index is not None and sample_id in features:
            valid_ratio = 1.0
            shape = "[10, 512]"
            dtype = "float32"
            valid = True
            error = ""
        else:
            valid_ratio = 0.0
            shape = ""
            dtype = ""
            valid = False
            error = ERROR_NO_FEATURES
        source = manifest["source_refs"]["video_file"].replace("\\", "/")
        rows.append({
            "sample_id": sample_id,
            "modality": "video",
            "subject_id": manifest["subject_id"],
            "session_id": manifest["session_id"],
            "split": subject_splits[manifest["subject_id"]],
            "source_file": json.dumps([source], ensure_ascii=False),
            "window_index": "0",
            "window_start_ms": "0",
            "window_end_ms": "10000",
            "duration_ms": "10000",
            "label_class": class_names[label_id],
            "label_id": str(label_id),
            "label_scheme": str(label_scheme.get("label_scheme", VIDEO_LABEL_SCHEME_NAME)),
            "valid": "true" if valid else "false",
            "valid_ratio": f"{valid_ratio:.6f}",
            "mask": f"{feature_rel}::valid_mask" if valid else "",
            "feature_path": feature_rel if valid else "",
            "feature_shape": shape if valid else "",
            "feature_dtype": dtype if valid else "",
            "extractor_name": FEATURE_EXTRACTOR_NAME,
            "extractor_version": FEATURE_EXTRACTOR_VERSION,
            "error": error,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="video_samples JSONL path")
    parser.add_argument("--label-scheme", required=True, help="video 6c label scheme JSON")
    parser.add_argument("--subject-splits", required=True, help="subject split JSON")
    parser.add_argument("--feature-index", default=None, help="optional feature index JSONL")
    parser.add_argument("--feature-root", default=None, help="optional NPZ feature root")
    parser.add_argument("--output", required=True, help="output metadata CSV")
    args = parser.parse_args(argv)
    try:
        rows = build_6c_rows(
            _read_jsonl(Path(args.manifest)),
            load_label_scheme(args.label_scheme),
            load_subject_splits(args.subject_splits),
            load_feature_index(args.feature_index),
            args.feature_root,
        )
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_6C_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    valid_count = sum(row["valid"] == "true" for row in rows)
    print(f"rows={len(rows)} valid={valid_count} missing_features={len(rows) - valid_count}")
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
