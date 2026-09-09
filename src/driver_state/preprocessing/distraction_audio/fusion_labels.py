"""6-class DCPT distraction label scheme shared with the video module (fusion).

The distraction-video module (陈星宇) uses the provisional ``dcpt_video_6c_v1``
scheme: tasks 01/03/04/05/07/08 mapped onto six classes in the original DCPT
order after dropping Watching video / Listening to radio / Chatting with
passenger. For multimodal fusion the audio side MUST reuse the exact same class
order, task mapping and subject splits as the video module; this module is the
single local source for that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

SIX_CLASS_TASKS: tuple[str, ...] = ("01", "03", "04", "05", "07", "08")
SIX_CLASS_NAMES: tuple[str, ...] = (
    "No task", "Playing game", "Messaging", "Phone call", "Reading", "Eating",
)
SIX_CLASS_TASK_TO_ID: dict[str, int] = {
    task: idx for idx, task in enumerate(SIX_CLASS_TASKS)
}
SIX_CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(SIX_CLASS_NAMES))

AUDIO_LABEL_SCHEME_NAME = "dcpt_audio_6c_v1"
VIDEO_LABEL_SCHEME_NAME = "dcpt_video_6c_v1"
# Subject split reused verbatim from the video module (24/8/8, seed 2026,
# provisional until the project lead freezes an official manifest).
AUDIO_SPLIT_VERSION = "dcpt_subject_24_8_8_seed2026_v1_provisional_audio_mirror"


def is_six_class_task(task_code: int) -> bool:
    return f"{task_code:02d}" in SIX_CLASS_TASK_TO_ID


def six_class_label(task_code: int) -> tuple[int, str]:
    """Return ``(label_id, label_class)`` for a DCPT task code (1..9)."""
    key = f"{task_code:02d}"
    if key not in SIX_CLASS_TASK_TO_ID:
        raise ValueError(f"task {task_code} is outside the 6-class video-aligned set")
    label_id = SIX_CLASS_TASK_TO_ID[key]
    return label_id, SIX_CLASS_ID_TO_NAME[label_id]


def load_label_scheme(path: str | Path) -> dict[str, object]:
    """Load and sanity-check a 6-class label scheme JSON (audio or video)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data.get("class_names")
    mapping = data.get("task_to_class")
    if not isinstance(names, list) or names != list(SIX_CLASS_NAMES):
        raise ValueError("class_names must match the six video-aligned classes in order")
    if not isinstance(mapping, dict) or mapping != {
        task: idx for idx, task in enumerate(SIX_CLASS_TASKS)
    }:
        raise ValueError("task_to_class must match the video 6c task mapping")
    return data


def load_subject_splits(path: str | Path) -> dict[str, str]:
    """Load subject->split mapping and validate keys/values."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("subject splits file must contain an object under 'splits'")
    from driver_state.constants import SPLITS

    out: dict[str, str] = {}
    for subject, split in splits.items():
        if not isinstance(subject, str) or not isinstance(split, str) or split not in SPLITS:
            raise ValueError("subject splits must map subject_id -> train/val/test")
        out[subject] = split
    return out