"""Build the standard DCPT video windows CSV metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CSV_COLUMNS = [
    "sample_id",
    "modality",
    "subject_id",
    "session_id",
    "split",
    "source_file",
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "duration_ms",
    "label_class",
    "label_id",
    "label_scheme",
    "valid",
    "valid_ratio",
    "mask",
    "feature_path",
    "feature_shape",
    "feature_dtype",
    "extractor_name",
    "extractor_version",
    "error",
]


def valid_ratio_from_mask(valid_mask: np.ndarray, support_s: np.ndarray) -> float:
    covered = 0.0
    for mask, support in zip(valid_mask, support_s):
        if not mask:
            continue
        start = max(0.0, float(support[0]))
        end = min(10.0, float(support[1]))
        covered += max(0.0, end - start)
    return min(1.0, covered / 10.0)


def build(config: dict, output_csv: Path, feature_root: Path) -> None:
    manifest_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in Path(config["manifest_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    split_record = json.loads(Path(config["split_path"]).read_text(encoding="utf-8"))
    split_map = split_record["splits"]
    task_to_class = config["task_to_class"]
    class_names = config["class_names"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for sample_id, manifest in sorted(manifest_rows.items()):
            task = manifest["label_raw"]
            label_id = task_to_class[task]
            feature_rel = f"video_features_v1/{sample_id}.npz"
            feature_path = feature_root / feature_rel
            data = np.load(feature_path)
            valid_ratio = valid_ratio_from_mask(data["valid_mask"], data["support_s"])
            is_valid = valid_ratio >= 0.95
            row = {
                "sample_id": sample_id,
                "modality": "video",
                "subject_id": manifest["subject_id"],
                "session_id": manifest["session_id"],
                "split": split_map[manifest["subject_id"]],
                "source_file": json.dumps(
                    [manifest["source_refs"]["video_file"].replace("\\", "/")],
                    ensure_ascii=False,
                ),
                "window_index": 0,
                "window_start_ms": 0,
                "window_end_ms": 10000,
                "duration_ms": 10000,
                "label_class": class_names[label_id],
                "label_id": label_id,
                "label_scheme": config["label_scheme"],
                "valid": "true" if is_valid else "false",
                "valid_ratio": f"{valid_ratio:.6f}",
                "mask": f"{feature_rel}::valid_mask",
                "feature_path": feature_rel,
                "feature_shape": json.dumps([int(dim) for dim in data["x"].shape]),
                "feature_dtype": str(data["x"].dtype),
                "extractor_name": "torchvision_r3d_18",
                "extractor_version": "0.24.1_kinetics400_v1",
                "error": "" if is_valid else "coverage_below_0.95",
            }
            writer.writerow(row)
    print(f"wrote {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="distraction_video/configs/distraction_video_baseline_v1.json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--feature-root", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    build(config, Path(args.output_csv), Path(args.feature_root))


if __name__ == "__main__":
    main()
