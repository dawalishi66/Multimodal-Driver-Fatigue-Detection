"""Tests for DCPT video filename parsing and the six-class mapping."""

from __future__ import annotations

import pytest

from driver_state.preprocessing.distraction_video.fusion_labels import (
    SIX_CLASS_NAMES,
    SIX_CLASS_TASKS,
    is_six_class_task,
    six_class_label,
)
from driver_state.preprocessing.distraction_video.naming import parse_clip_filename


def test_parse_video_clip_filename() -> None:
    ref = parse_clip_filename("Upper_body_video_01/01_P01_20231111_09_31_43_12.mp4")
    assert ref is not None
    assert ref.stem == "01_P01_20231111_09_31_43_12"
    assert ref.subject_id == "P01"
    assert ref.session_id == "P01_20231111_0931_43"
    assert ref.label_id == 0
    assert ref.label_class == "No task"


def test_non_video_or_non_six_class_names_return_none() -> None:
    assert parse_clip_filename("01_P01_20231111_09_31_43_12.wav") is None
    assert parse_clip_filename("02_P01_20231111_09_31_43_12.mp4") is None
    assert parse_clip_filename("bad-name.mp4") is None


def test_six_class_mapping() -> None:
    for index, task in enumerate(SIX_CLASS_TASKS):
        assert six_class_label(int(task)) == (index, SIX_CLASS_NAMES[index])
        assert is_six_class_task(int(task))
    assert not is_six_class_task(2)
    with pytest.raises(ValueError):
        six_class_label(2)
