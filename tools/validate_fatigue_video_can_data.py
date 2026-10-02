"""Validate the frozen UL-DD train/val pair manifest through the public Dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from driver_state.data import (
    FatigueVideoCanDataset,
    collate_fatigue_video_can_batch,
    make_fusion_model_inputs,
)
from driver_state.data.can import file_sha256


EXPECTED = {
    "train": {"windows": 1272, "parents": 159, "subjects": 10, "sessions": 18},
    "val": {"windows": 328, "parents": 41, "subjects": 3, "sessions": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-version", default="uncommitted")
    return parser.parse_args()


def _validate_split(
    *, pair_manifest: Path, dataset_root: Path, split: str
) -> dict[str, Any]:
    dataset = FatigueVideoCanDataset(
        pair_manifest,
        dataset_root=dataset_root,
        split=split,
        verify_feature_hashes=True,
        cache_features=True,
    )
    labels: Counter[int] = Counter()
    subjects: set[str] = set()
    sessions: set[str] = set()
    sample_ids: set[str] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        if sample_id in sample_ids:
            raise ValueError(f"duplicate loaded sample_id: {sample_id}")
        sample_ids.add(sample_id)
        labels[int(sample["label"].item())] += 1
        subjects.add(sample["subject_id"])
        sessions.add(sample["session_id"])

    actual = {
        "windows": len(dataset),
        "parents": dataset.parent_count,
        "subjects": len(subjects),
        "sessions": len(sessions),
    }
    if actual != EXPECTED[split]:
        raise ValueError(f"{split} count mismatch: expected {EXPECTED[split]}, got {actual}")

    batch = collate_fatigue_video_can_batch([dataset[0], dataset[-1]])
    model_inputs = make_fusion_model_inputs(batch)
    if tuple(batch["inputs"]["video"]["x"].shape) != (2, 6, 96):
        raise ValueError(f"unexpected {split} video batch shape")
    if tuple(batch["inputs"]["can"]["x"].shape) != (2, 300, 9):
        raise ValueError(f"unexpected {split} CAN batch shape")
    if tuple(batch["labels"].shape) != (2,) or batch["labels"].dtype != torch.int64:
        raise ValueError(f"unexpected {split} labels shape or dtype")
    if any(set(stream) != {"x", "valid_mask", "time_s"} for stream in model_inputs.values()):
        raise ValueError("model input adapter returned an invalid contract")

    return {
        **actual,
        "subject_ids": sorted(subjects),
        "session_ids": sorted(sessions),
        "label_counts": {str(label): labels[label] for label in range(3)},
        "video_shape": [6, 96],
        "can_shape": [300, 9],
        "batch_smoke": "PASS",
        "feature_hash_verification": "PASS",
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": "fatigue_video_can_dataset_validation_v1",
        "status": "FAIL",
        "formal_result": False,
        "test_manifest_accessed": False,
        "pair_manifest": args.pair_manifest.name,
        "pair_manifest_sha256": file_sha256(args.pair_manifest),
        "dataset_root_identifier": args.dataset_root.name,
        "code_version": args.code_version,
        "expected": EXPECTED,
    }
    exit_code = 1
    try:
        report["splits"] = {
            split: _validate_split(
                pair_manifest=args.pair_manifest,
                dataset_root=args.dataset_root,
                split=split,
            )
            for split in ("train", "val")
        }
        report["totals"] = {"windows": 1600, "parents": 200}
        report["time_contract"] = {
            "video_input": "session_relative",
            "video_transform": "subtract_window_start_ms_div_1000",
            "model_batch": "sample_relative_0_to_30_seconds",
            "can_input": "sample_relative_0_to_30_seconds",
        }
        report["batch_contract"] = {
            "full_fields": [
                "x", "time_s", "valid_mask", "support_s", "observed_fraction"
            ],
            "model_fields": ["x", "valid_mask", "time_s"],
            "mask_semantics": "true_is_valid",
        }
        report["status"] = "PASS"
        exit_code = 0
    except Exception as exc:  # validation entry point must preserve the failure
        report["error"] = f"{type(exc).__name__}: {exc}"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
