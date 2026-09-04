"""Versioned task decisions from experimental protocol v0.2."""

import math

PROTOCOL_VERSION = "0.2"
SCHEMA_VERSION = "0.2.0"
SPLITS = ("train", "val", "test")
TRAINING_SEEDS = (11, 22, 33)
FATIGUE_WINDOW_MS = 30_000
FATIGUE_STRIDE_MS = 30_000
FATIGUE_VIDEO_SUBWINDOW_MS = 5_000
FATIGUE_PARENT_MS = 240_000
DCPT_WINDOW_MS = 10_000
KSS_CLASSES = ("low", "medium", "high")
ULDD_SUBJECT_SPLITS = {
    "train": ("D", "F", "G", "J", "K", "L", "N", "Q", "R", "S"),
    "val": ("A", "E", "O"),
    "test": ("C", "H", "P"),
}
ULDD_SPLIT_BY_SUBJECT = {
    subject: split
    for split, subjects in ULDD_SUBJECT_SPLITS.items()
    for subject in subjects
}
DCPT_CLASSES = (
    "No task", "Watching video", "Playing game", "Messaging", "Phone call",
    "Listening to radio", "Reading", "Eating", "Chatting with passenger",
)
TASK_MODALITIES = {"fatigue": ("video", "can"), "distraction": ("video", "audio")}


def kss_label(score: float) -> tuple[int, str]:
    """Map finite original KSS in [1, 9] without rounding."""
    if isinstance(score, bool):
        raise ValueError("KSS must be a number, not a boolean")
    value = float(score)
    if not math.isfinite(value) or not 1 <= value <= 9:
        raise ValueError("KSS must be finite and in [1, 9]")
    label_id = 0 if value < 4 else (1 if value < 7 else 2)
    return label_id, KSS_CLASSES[label_id]
