"""Owner: 李坤洋. Audit CAN clocks and export common 30-second features."""
"""UL-DD driver-telemetry preprocessing for the fatigue task."""

from .pipeline import (
    CAN_FEATURE_COLUMNS,
    build_can_dataset,
    detect_clock_discontinuities,
    resample_can_window,
)

__all__ = [
    "CAN_FEATURE_COLUMNS",
    "build_can_dataset",
    "detect_clock_discontinuities",
    "resample_can_window",
]
