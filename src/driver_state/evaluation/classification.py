"""Version-stable classification metrics and fatigue parent aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def validate_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_classes: int,
    atol: float = 1e-5,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != num_classes:
        raise ValueError(f"probabilities must have shape [N,{num_classes}]")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=atol):
        raise ValueError("each probability row must sum to one")
    return values


def classification_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    probs = validate_probabilities(probabilities, num_classes=num_classes)
    if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0] or not y_true.size:
        raise ValueError("labels must be a non-empty [N] vector matching probabilities")
    if (y_true < 0).any() or (y_true >= num_classes).any():
        raise ValueError("labels contain an out-of-range class")
    class_ids = np.arange(num_classes)
    predictions = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=class_ids,
        zero_division=0,
    )
    supported = support > 0
    return {
        "sample_count": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1_fixed_classes": float(np.mean(f1)),
        "balanced_accuracy_supported_classes": float(
            np.mean(recall[supported]) if supported.any() else 0.0
        ),
        "per_class": [
            {
                "class_id": int(class_id),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in class_ids
        ],
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=class_ids
        ).astype(int).tolist(),
    }


def aggregate_fatigue_parents(
    records: Iterable[dict[str, Any]],
    *,
    num_classes: int = 3,
    expected_windows: int = 8,
) -> list[dict[str, Any]]:
    """Average fixed-window probabilities for each complete 240-second parent."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["parent_id"])].append(record)
    if not grouped:
        raise ValueError("cannot aggregate an empty prediction collection")

    parents: list[dict[str, Any]] = []
    for parent_id in sorted(grouped):
        rows = grouped[parent_id]
        indices = [int(row["window_index"]) for row in rows]
        if sorted(indices) != list(range(expected_windows)):
            raise ValueError(
                f"parent {parent_id} must contain window indices "
                f"0..{expected_windows - 1} exactly once"
            )
        labels = {int(row["label_id"]) for row in rows}
        subjects = {str(row["subject_id"]) for row in rows}
        sessions = {str(row["session_id"]) for row in rows}
        splits = {str(row["split"]) for row in rows}
        if len(labels) != 1 or len(subjects) != 1 or len(sessions) != 1 or len(splits) != 1:
            raise ValueError(f"parent {parent_id} has inconsistent metadata")
        probs = validate_probabilities(
            [row["probabilities"] for row in rows], num_classes=num_classes
        )
        mean_probabilities = probs.mean(axis=0)
        parents.append({
            "parent_id": parent_id,
            "subject_id": next(iter(subjects)),
            "session_id": next(iter(sessions)),
            "split": next(iter(splits)),
            "label_id": next(iter(labels)),
            "probabilities": mean_probabilities.tolist(),
            "prediction": int(mean_probabilities.argmax()),
            "window_count": expected_windows,
        })
    return parents


def majority_class_from_train(labels: Sequence[int], *, num_classes: int) -> int:
    values = [int(label) for label in labels]
    if not values or any(label < 0 or label >= num_classes for label in values):
        raise ValueError("training labels are empty or out of range")
    counts = Counter(values)
    maximum = max(counts.values())
    return min(class_id for class_id in range(num_classes) if counts[class_id] == maximum)


def per_subject_metrics(
    subject_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_classes: int,
) -> dict[str, dict[str, Any]]:
    if len(subject_ids) != len(labels):
        raise ValueError("subject_ids and labels must have equal length")
    probs = validate_probabilities(probabilities, num_classes=num_classes)
    if probs.shape[0] != len(subject_ids):
        raise ValueError("subject_ids and probabilities must have equal length")
    result = {}
    for subject in sorted(set(subject_ids)):
        indices = [index for index, value in enumerate(subject_ids) if value == subject]
        result[subject] = classification_metrics(
            [labels[index] for index in indices],
            probs[indices],
            num_classes=num_classes,
        )
    return result
