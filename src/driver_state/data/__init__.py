"""Shared data contracts; see docs/INTERFACES.md."""

from .fatigue_pairing import (
    CompleteParent,
    canonical_pair_id,
    compare_labels,
    normalize_video_time_arrays,
    pairing_key,
    select_complete_parents,
    validate_normalized_video_time,
)

__all__ = [
    "CompleteParent",
    "canonical_pair_id",
    "compare_labels",
    "normalize_video_time_arrays",
    "pairing_key",
    "select_complete_parents",
    "validate_normalized_video_time",
]
