"""Build the 6-class label mapping and provisional subject split."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def subject_sort_key(subject_id: str) -> int:
    match = re.match(r"^P(\d+)$", subject_id)
    if match is None:
        raise ValueError(f"invalid subject_id: {subject_id}")
    return int(match.group(1))


def prepare(config: dict) -> None:
    manifest_path = Path(config["manifest_path"])
    mapping_path = Path(config["label_mapping_path"])
    split_path = Path(config["split_path"])
    training_path = Path(config["training_samples_path"])
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    task_to_class = config["task_to_class"]
    class_names = config["class_names"]
    if len(class_names) != len(task_to_class):
        raise ValueError("class_names and task_to_class sizes differ")

    mapping = {
        "label_scheme": config["label_scheme"],
        "class_names": class_names,
        "task_to_class": task_to_class,
        "note": "Provisional 6-class compression confirmed by module lead; replace with official manifest when available.",
    }
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subjects = sorted({row["subject_id"] for row in manifest}, key=subject_sort_key)
    if len(subjects) != 40:
        raise ValueError(f"expected 40 subjects, found {len(subjects)}")

    random.Random(int(config["split_seed"])).shuffle(subjects)
    train_count = int(config["train_subject_count"])
    val_count = int(config["val_subject_count"])
    split_map: dict[str, str] = {}
    for subject_id in subjects[:train_count]:
        split_map[subject_id] = "train"
    for subject_id in subjects[train_count : train_count + val_count]:
        split_map[subject_id] = "val"
    for subject_id in subjects[train_count + val_count :]:
        split_map[subject_id] = "test"

    split_record = {
        "split_version": "dcpt_subject_24_8_8_seed2026_v1_provisional",
        "split_seed": int(config["split_seed"]),
        "split_units": "subject",
        "subject_counts": {
            "train": train_count,
            "val": val_count,
            "test": len(subjects) - train_count - val_count,
        },
        "splits": split_map,
    }
    split_path.write_text(json.dumps(split_record, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    errors = []
    for row in manifest:
        task = row["label_raw"]
        if task not in task_to_class:
            errors.append({"sample_id": row["sample_id"], "task": task})
            continue
        rows.append(
            {
                "sample_id": row["sample_id"],
                "subject_id": row["subject_id"],
                "label_raw": task,
                "label_id": task_to_class[task],
                "label_scheme": config["label_scheme"],
                "split": split_map[row["subject_id"]],
            }
        )
    if errors:
        raise RuntimeError(f"unmapped tasks: {errors[:5]}")

    with training_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "sample_count": len(rows),
        "split_subject_counts": split_record["subject_counts"],
        "class_counts_by_split": {
            split_name: dict(
                sorted(Counter(r["label_id"] for r in rows if r["split"] == split_name).items())
            )
            for split_name in ("train", "val", "test")
        },
        "split_subject_samples": {
            split_name: sum(1 for r in rows if r["split"] == split_name)
            for split_name in ("train", "val", "test")
        },
    }
    summary_path = training_path.with_name("video_training_summary_6c_v1.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="distraction_video/configs/distraction_video_baseline_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prepare(config)


if __name__ == "__main__":
    main()
