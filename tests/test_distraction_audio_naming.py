"""Tests for DCPT distraction-audio filename parsing."""

from __future__ import annotations

import pytest

from driver_state.constants import DCPT_CLASSES
from driver_state.preprocessing.distraction_audio.naming import (
    DcptClipRef,
    parse_clip_filename,
)

SAMPLE = "First_person_view_audio/01_P01_20231111_09_31_43_12.wav"


def test_parse_full_archive_path() -> None:
    ref = parse_clip_filename(SAMPLE)
    assert ref is not None
    assert isinstance(ref, DcptClipRef)
    assert ref.task_code == 1
    assert ref.subject_id == "P01"
    assert ref.date == "20231111"
    assert ref.start_hhmmss == "093143"
    assert ref.takeover_decisec == 12
    assert ref.stem == "01_P01_20231111_09_31_43_12"
    assert ref.session_id == "P01_20231111_0931_43"
    assert ref.extension == "wav"


def test_parse_accepts_backslash_paths() -> None:
    ref = parse_clip_filename("First_person_view_audio\\02_P40_20240101_10_20_30_25.wav")
    assert ref is not None
    assert ref.subject_id == "P40"
    assert ref.task_code == 2


@pytest.mark.parametrize("task", list(range(1, 10)))
def test_task_to_label_mapping(task: int) -> None:
    name = f"{task:02d}_P07_20231115_18_45_03_21.wav"
    ref = parse_clip_filename(name)
    assert ref is not None
    assert ref.label_id == task - 1
    assert ref.label_class == DCPT_CLASSES[task - 1]


@pytest.mark.parametrize(
    "name",
    [
        "00_P01_20231111_09_31_43_12.wav",   # task 0
        "10_P01_20231111_09_31_43_12.wav",   # task 10
        "01_P41_20231111_09_31_43_12.wav",   # person P41
        "01_P00_20231111_09_31_43_12.wav",   # person P00
        "01_P1_20231111_09_31_43_12.wav",    # person not zero padded
        "01_P01_20231111_09_31_43.wav",      # missing takeover seconds field
        "notes.txt",                         # not a DCPT clip
        "01_P01_20231111_09_31_43_12",       # no extension
    ],
)
def test_rejects_non_matching_names(name: str) -> None:
    assert parse_clip_filename(name) is None
