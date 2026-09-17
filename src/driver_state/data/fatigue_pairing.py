"""Deterministic contracts for UL-DD video/CAN 30-second pairing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from driver_state.constants import FATIGUE_PARENT_MS, FATIGUE_WINDOW_MS, kss_label


PAIRING_VERSION = "1.0.0"
PAIR_KEY_FIELDS = (
    "subject_id",
    "session_id",
    "window_start_ms",
    "window_end_ms",
)
LABEL_FIELDS = (
    "split",
    "label_start_ms",
    "label_end_ms",
    "kss_score",
    "label_class",
    "label_id",
)


def parse_bool(value: Any) -> bool:
    """Parse CSV-compatible booleans without treating ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def pairing_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    """Return the only permitted cross-modal join key."""
    return (
        str(row["subject_id"]),
        str(row["session_id"]),
        int(row["window_start_ms"]),
        int(row["window_end_ms"]),
    )


def canonical_pair_id(row: Mapping[str, Any]) -> str:
    subject, session, start_ms, end_ms = pairing_key(row)
    condition = session.rsplit("_", 1)[-1]
    return f"ULDD_{subject}_{condition}_{start_ms:09d}_{end_ms:09d}"


def compare_labels(
    video: Mapping[str, Any], can: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return label/identity fields that disagree across the two modalities."""
    mismatches: list[str] = []
    for field in LABEL_FIELDS:
        if field == "kss_score":
            equal = bool(np.isclose(float(video[field]), float(can[field])))
        elif field in {"label_start_ms", "label_end_ms", "label_id"}:
            equal = int(video[field]) == int(can[field])
        else:
            equal = str(video[field]) == str(can[field])
        if not equal:
            mismatches.append(field)
    return tuple(mismatches)


def normalize_video_time_arrays(
    time_s: np.ndarray,
    support_s: np.ndarray,
    *,
    window_start_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert video session-relative time arrays to sample-relative arrays."""
    offset_s = float(window_start_ms) / 1000.0
    normalized_time = np.asarray(time_s, dtype=np.float64) - offset_s
    normalized_support = np.asarray(support_s, dtype=np.float64) - offset_s
    return normalized_time, normalized_support


def validate_normalized_video_time(
    time_s: np.ndarray,
    support_s: np.ndarray,
    *,
    duration_s: float = 30.0,
) -> tuple[str, ...]:
    """Validate the public sample-relative time contract after conversion."""
    errors: list[str] = []
    time_s = np.asarray(time_s)
    support_s = np.asarray(support_s)
    if time_s.ndim != 1 or support_s.shape != (time_s.shape[0], 2):
        return ("shape",)
    if not np.isfinite(time_s).all() or not np.isfinite(support_s).all():
        errors.append("nonfinite")
    if len(time_s) and not np.all(np.diff(time_s) > 0):
        errors.append("time_not_strictly_increasing")
    if len(time_s) and not (
        np.all(support_s[:, 0] >= 0.0)
        and np.all(support_s[:, 1] <= duration_s)
        and np.all(support_s[:, 1] > support_s[:, 0])
        and np.all(time_s > support_s[:, 0])
        and np.all(time_s < support_s[:, 1])
    ):
        errors.append("outside_sample_support")
    return tuple(errors)


@dataclass(frozen=True)
class CompleteParent:
    parent_id: str
    split: str
    subject_id: str
    session_id: str
    label_start_ms: int
    label_end_ms: int
    kss_score: float
    label_class: str
    label_id: int
    sample_ids: tuple[str, ...]


def select_complete_parents(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[set[str], tuple[CompleteParent, ...]]:
    """Select strict 8x30-second parents from already valid paired rows."""
    grouped: dict[tuple[str, str, str, int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        if not parse_bool(row["paired_valid"]):
            continue
        key = (
            str(row["split"]),
            str(row["subject_id"]),
            str(row["session_id"]),
            int(row["label_start_ms"]),
            int(row["label_end_ms"]),
        )
        grouped[key].append(row)

    eligible_ids: set[str] = set()
    parents: list[CompleteParent] = []
    for key, group in sorted(grouped.items()):
        split, subject, session, label_start_ms, label_end_ms = key
        ordered = sorted(group, key=lambda row: int(row["window_start_ms"]))
        expected_starts = [
            label_start_ms + index * FATIGUE_WINDOW_MS for index in range(8)
        ]
        actual_starts = [int(row["window_start_ms"]) for row in ordered]
        if (
            len(ordered) != 8
            or actual_starts != expected_starts
            or label_end_ms - label_start_ms != FATIGUE_PARENT_MS
        ):
            continue
        if any(
            int(row["window_end_ms"]) - int(row["window_start_ms"])
            != FATIGUE_WINDOW_MS
            for row in ordered
        ):
            continue
        labels = {
            (
                float(row["kss_score"]),
                str(row["label_class"]),
                int(row["label_id"]),
            )
            for row in ordered
        }
        if len(labels) != 1:
            continue
        score, label_class, label_id = labels.pop()
        if kss_label(score) != (label_id, label_class):
            continue
        ids = tuple(str(row["paired_sample_id"]) for row in ordered)
        eligible_ids.update(ids)
        condition = session.rsplit("_", 1)[-1]
        parent_id = (
            f"ULDD_{subject}_{condition}_{label_start_ms:09d}_{label_end_ms:09d}"
        )
        parents.append(
            CompleteParent(
                parent_id=parent_id,
                split=split,
                subject_id=subject,
                session_id=session,
                label_start_ms=label_start_ms,
                label_end_ms=label_end_ms,
                kss_score=score,
                label_class=label_class,
                label_id=label_id,
                sample_ids=ids,
            )
        )
    return eligible_ids, tuple(parents)
