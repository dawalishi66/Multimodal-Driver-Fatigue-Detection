"""Shared data contracts and loaders; see docs/INTERFACES.md."""

from .can import (
    CanWindowDataset,
    CanWindowRecord,
    collate_can_batch,
    load_complete_parent_ids,
    make_can_collate,
)
from .fatigue_pairing import (
    CompleteParent,
    canonical_pair_id,
    compare_labels,
    normalize_video_time_arrays,
    pairing_key,
    select_complete_parents,
    validate_normalized_video_time,
)
from .fatigue_video_can import (
    FatigueVideoCanDataset,
    FatigueVideoCanRecord,
    collate_fatigue_video_can_batch,
    make_fatigue_video_can_collate,
    make_fusion_model_inputs,
)
from .normalization import MaskedStandardizer

__all__ = [
    "CanWindowDataset",
    "CanWindowRecord",
    "CompleteParent",
    "FatigueVideoCanDataset",
    "FatigueVideoCanRecord",
    "MaskedStandardizer",
    "canonical_pair_id",
    "collate_can_batch",
    "collate_fatigue_video_can_batch",
    "compare_labels",
    "load_complete_parent_ids",
    "make_can_collate",
    "make_fatigue_video_can_collate",
    "make_fusion_model_inputs",
    "normalize_video_time_arrays",
    "pairing_key",
    "select_complete_parents",
    "validate_normalized_video_time",
]
