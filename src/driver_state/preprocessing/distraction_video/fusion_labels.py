"""Provisional six-class DCPT distraction contract used by the video module."""

from __future__ import annotations

import json
from pathlib import Path

VIDEO_LABEL_SCHEME_NAME = "dcpt_video_6c_v1"
VIDEO_SPLIT_VERSION = "dcpt_subject_24_8_8_seed2026_v1_provisional"

SIX_CLASS_TASKS: tuple[str, ...] = ("01", "03", "04", "05", "07", "08")
SIX_CLASS_NAMES: tuple[str, ...] = (
    "No task", "Playing game", "Messaging", "Phone call", "Reading", "Eating",
)
SIX_CLASS_TASK_TO_ID: dict[str, int] = {
    task: index for index, task in enumerate(SIX_CLASS_TASKS)
}
SIX_CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(SIX_CLASS_NAMES))


def is_six_class_task(task_code: int) -> bool:
    """Return whether a DCPT one-based task code belongs to the video subset."""
    return f"{task_code:02d}" in SIX_CLASS_TASK_TO_ID


def six_class_label(task_code: int) -> tuple[int, str]:
    """Map a task code to its six-class ``(label_id, label_class)`` pair."""
    key = f"{task_code:02d}"
    if key not in SIX_CLASS_TASK_TO_ID:
        raise ValueError(f"task {task_code} is outside the six-class video set")
    label_id = SIX_CLASS_TASK_TO_ID[key]
    return label_id, SIX_CLASS_ID_TO_NAME[label_id]


def load_label_scheme(path: str | Path) -> dict[str, object]:
    """Load and validate a six-class label scheme JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("class_names") != list(SIX_CLASS_NAMES):
        raise ValueError("class_names must match the six video classes in order")
    expected_mapping = {task: index for index, task in enumerate(SIX_CLASS_TASKS)}
    if data.get("task_to_class") != expected_mapping:
        raise ValueError("task_to_class must match the video six-class mapping")
    return data


def load_subject_splits(path: str | Path) -> dict[str, str]:
    """Load a subject -> train/val/test mapping."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("subject splits file must contain an object under 'splits'")
    out: dict[str, str] = {}
    for subject, split in splits.items():
        if not isinstance(subject, str) or split not in {"train", "val", "test"}:
            raise ValueError("subject splits must map subject_id -> train/val/test")
        out[subject] = split
    return out
