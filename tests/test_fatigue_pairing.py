import numpy as np
import pytest

from driver_state.data.fatigue_pairing import (
    canonical_pair_id,
    compare_labels,
    normalize_video_time_arrays,
    pairing_key,
    parse_bool,
    select_complete_parents,
    validate_normalized_video_time,
)


def make_pair(index=0, *, paired_valid="true", label_id=1, label_class="medium"):
    start = index * 30_000
    return {
        "paired_sample_id": f"ULDD_D_A_{start:09d}_{start + 30_000:09d}",
        "subject_id": "D",
        "session_id": "D_A",
        "split": "train",
        "window_start_ms": start,
        "window_end_ms": start + 30_000,
        "label_start_ms": 0,
        "label_end_ms": 240_000,
        "kss_score": 4.0,
        "label_class": label_class,
        "label_id": label_id,
        "paired_valid": paired_valid,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (0, False), ("true", True), ("false", False)],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_parse_bool_rejects_ambiguous_text():
    with pytest.raises(ValueError):
        parse_bool("yes")


def test_pairing_key_and_canonical_id():
    row = make_pair(2)
    assert pairing_key(row) == ("D", "D_A", 60_000, 90_000)
    assert canonical_pair_id(row) == "ULDD_D_A_000060000_000090000"


def test_compare_labels_reports_only_disagreements():
    video = make_pair()
    can = dict(video)
    assert compare_labels(video, can) == ()
    can["label_id"] = 2
    can["label_class"] = "high"
    assert compare_labels(video, can) == ("label_class", "label_id")


def test_video_session_time_is_converted_to_sample_time():
    time_s = np.arange(32.5, 60, 5, dtype=np.float64)
    support_s = np.column_stack(
        (np.arange(30, 60, 5, dtype=np.float64), np.arange(35, 65, 5, dtype=np.float64))
    )
    converted_time, converted_support = normalize_video_time_arrays(
        time_s, support_s, window_start_ms=30_000
    )
    np.testing.assert_array_equal(converted_time, [2.5, 7.5, 12.5, 17.5, 22.5, 27.5])
    np.testing.assert_array_equal(converted_support[0], [0.0, 5.0])
    np.testing.assert_array_equal(converted_support[-1], [25.0, 30.0])
    assert validate_normalized_video_time(converted_time, converted_support) == ()


def test_video_time_contract_rejects_unconverted_session_time():
    time_s = np.arange(32.5, 60, 5, dtype=np.float64)
    support_s = np.column_stack(
        (np.arange(30, 60, 5, dtype=np.float64), np.arange(35, 65, 5, dtype=np.float64))
    )
    assert "outside_sample_support" in validate_normalized_video_time(time_s, support_s)


def test_complete_parent_requires_exactly_eight_contiguous_windows():
    eligible, parents = select_complete_parents([make_pair(i) for i in range(8)])
    assert len(eligible) == 8
    assert len(parents) == 1
    assert parents[0].parent_id == "ULDD_D_A_000000000_000240000"


@pytest.mark.parametrize(
    "rows",
    [
        [make_pair(i) for i in range(7)],
        [make_pair(i, paired_valid="false" if i == 3 else "true") for i in range(8)],
        [make_pair(i, label_id=2, label_class="high") if i == 3 else make_pair(i) for i in range(8)],
    ],
)
def test_incomplete_or_inconsistent_parent_is_rejected(rows):
    eligible, parents = select_complete_parents(rows)
    assert eligible == set()
    assert parents == ()
