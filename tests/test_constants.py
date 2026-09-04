import json
from pathlib import Path

import pytest

from driver_state.constants import (
    DCPT_CLASSES, FATIGUE_PARENT_MS, FATIGUE_STRIDE_MS, FATIGUE_VIDEO_SUBWINDOW_MS,
    FATIGUE_WINDOW_MS, KSS_CLASSES, SCHEMA_VERSION, TRAINING_SEEDS,
    ULDD_SPLIT_BY_SUBJECT, ULDD_SUBJECT_SPLITS, kss_label,
)
from driver_state.schemas import FATIGUE_METADATA_FIELDS


@pytest.mark.parametrize("score,label", [(1, 0), (3.9, 0), (4, 1), (4.5, 1), (6, 1), (6.5, 1), (7, 2), (7.5, 2), (9, 2)])
def test_kss_boundaries(score, label):
    assert kss_label(score) == (label, KSS_CLASSES[label])


@pytest.mark.parametrize("score", [0, 10, float("nan"), float("inf"), float("-inf"), True])
def test_invalid_kss(score):
    with pytest.raises(ValueError):
        kss_label(score)


def test_fixed_split():
    assert len(ULDD_SPLIT_BY_SUBJECT) == 16
    assert tuple(map(len, ULDD_SUBJECT_SPLITS.values())) == (10, 3, 3)
    assert ULDD_SPLIT_BY_SUBJECT["A"] == "val"
    assert ULDD_SPLIT_BY_SUBJECT["P"] == "test"
    assert "B" not in ULDD_SPLIT_BY_SUBJECT


def test_config_and_code_agree():
    root = Path(__file__).resolve().parents[1]
    fatigue = json.loads((root / "configs/fatigue_uldd.json").read_text(encoding="utf-8"))
    distraction = json.loads((root / "configs/distraction_dcpt.json").read_text(encoding="utf-8"))
    assert fatigue["subject_splits"] == {k: list(v) for k, v in ULDD_SUBJECT_SPLITS.items()}
    assert fatigue["classes"] == list(KSS_CLASSES)
    assert distraction["classes"] == list(DCPT_CLASSES)
    assert fatigue["window_ms"] == FATIGUE_WINDOW_MS
    assert fatigue["stride_ms"] == FATIGUE_STRIDE_MS
    assert fatigue["parent_ms"] == FATIGUE_PARENT_MS
    assert fatigue["video_subwindow_ms"] == FATIGUE_VIDEO_SUBWINDOW_MS
    for config in (fatigue, distraction):
        assert config["schema_version"] == SCHEMA_VERSION
        assert config["training_seeds"] == list(TRAINING_SEEDS)
        assert config["status"] == "preflight_required"
    assert len(FATIGUE_METADATA_FIELDS) == len(set(FATIGUE_METADATA_FIELDS)) == 24
