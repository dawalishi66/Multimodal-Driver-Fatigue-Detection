import numpy as np
import pytest

from driver_state.evaluation import (
    aggregate_fatigue_parents,
    classification_metrics,
    majority_class_from_train,
)


def test_fixed_class_metrics_are_independently_checkable():
    probabilities = np.asarray([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
    ])
    result = classification_metrics([0, 1, 2], probabilities, num_classes=3)
    assert result["macro_f1_fixed_classes"] == 1.0
    assert result["balanced_accuracy_supported_classes"] == 1.0
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_parent_aggregation_uses_mean_of_eight_probabilities():
    records = []
    for window_index in range(8):
        records.append({
            "parent_id": "SYNTHETIC_PARENT",
            "subject_id": "D",
            "session_id": "D_A",
            "split": "train",
            "window_index": window_index,
            "label_id": 1,
            "probabilities": [0.1, 0.7 if window_index < 7 else 0.4,
                              0.2 if window_index < 7 else 0.5],
        })
    parent = aggregate_fatigue_parents(records)[0]
    assert parent["window_count"] == 8
    assert parent["prediction"] == 1
    assert parent["probabilities"] == pytest.approx(
        np.mean([row["probabilities"] for row in records], axis=0)
    )


def test_parent_aggregation_rejects_duplicate_or_missing_window():
    records = [{
        "parent_id": "P0", "subject_id": "D", "session_id": "D_A",
        "split": "train", "window_index": index if index < 7 else 6,
        "label_id": 0, "probabilities": [0.8, 0.1, 0.1],
    } for index in range(8)]
    with pytest.raises(ValueError, match="0..7 exactly once"):
        aggregate_fatigue_parents(records)


def test_majority_class_uses_train_labels_and_stable_tie_break():
    assert majority_class_from_train([1, 1, 2, 0], num_classes=3) == 1
    assert majority_class_from_train([2, 2, 1, 1], num_classes=3) == 1
