"""Tests for the video-v4-compatible audio ``evaluation.json`` builder."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from driver_state.preprocessing.distraction_audio.build_audio_evaluation import (
    build_audio_evaluation,
)
from driver_state.preprocessing.distraction_audio.fusion_labels import (
    AUDIO_LABEL_SCHEME_NAME,
    SIX_CLASS_NAMES,
)

N_CLASSES = len(SIX_CLASS_NAMES)
SEEDS = [11, 22, 33]


# --------------------------------------------------------------------------- #
# Synthetic data helpers (independent reference implementation)
# --------------------------------------------------------------------------- #
def _samples() -> list[dict]:
    """A fixed test set: 2 subjects, 8 samples each, all 6 classes present."""
    samples: list[dict] = []
    for subj in ("P01", "P02"):
        labels = [0, 1, 2, 3, 4, 5, 0, 1]  # classes 0..5 present, with repeats
        for i, lab in enumerate(labels):
            samples.append({
                "sample_id": f"01_{subj}_20231111_09_31_43_{i:02d}",
                "subject_id": subj,
                "label": lab,
                "prediction": lab,
            })
    return samples


def _preds_for_seed(seed: int, samples: list[dict]) -> list[dict]:
    """Introduce a deterministic, seed-dependent confusion pattern."""
    out: list[dict] = []
    for idx, s in enumerate(samples):
        lab = s["label"]
        pred = lab
        # Flip a controlled subset: mimic class confusion between adjacent classes.
        flip = (idx + seed) % 5 == 0
        if flip and lab < N_CLASSES - 1:
            pred = (lab + 1) % N_CLASSES
        rec = dict(s)
        rec["prediction"] = pred
        rec["probabilities"] = [0.0] * N_CLASSES
        rec["probabilities"][pred] = 1.0
        rec["seed"] = seed
        rec["split"] = "test"
        out.append(rec)
    return out


def _per_class_stats(preds: list[int], true: list[int]) -> dict:
    per_precision: list[float] = []
    per_recall: list[float] = []
    per_f1: list[float] = []
    support: list[int] = []
    cm = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    for p, t in zip(preds, true):
        cm[t][p] += 1
    for c in range(N_CLASSES):
        tp = sum(1 for p, t in zip(preds, true) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, true) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, true) if p != c and t == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)
        support.append(tp + fn)
    accuracy = sum(1 for p, t in zip(preds, true) if p == t) / len(preds) if preds else 0.0
    macro_f1 = float(np.mean(per_f1)) if per_f1 else 0.0
    balanced_accuracy = float(np.mean(per_recall)) if per_recall else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "per_class_precision": per_precision,
        "per_class_recall": per_recall,
        "per_class_f1": per_f1,
        "class_support": support,
        "confusion_matrix": cm,
    }


def _write_run(run_root: Path, summary_path: Path) -> dict[int, Path]:
    samples = _samples()
    workdirs: dict[int, Path] = {}
    per_seed_metrics: list[dict] = []
    for seed in SEEDS:
        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        preds = _preds_for_seed(seed, samples)
        with (seed_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as fh:
            for rec in preds:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats = _per_class_stats(
            [r["prediction"] for r in preds], [r["label"] for r in preds])
        metrics = {"seed": seed, "test_metrics": stats}
        (seed_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
        per_seed_metrics.append(metrics)
        workdirs[seed] = seed_dir
    summary = {
        "baseline_version": "v_synth",
        "seeds": SEEDS,
        "test_macro_f1_mean": float(np.mean([m["test_metrics"]["macro_f1"]
                                             for m in per_seed_metrics])),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return workdirs


@pytest.fixture()
def synth(tmp_path: Path):
    run_root = tmp_path / "runs"
    summary_path = tmp_path / "summary.json"
    workdirs = _write_run(run_root, summary_path)
    return run_root, summary_path, workdirs


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_evaluation_schema_fields(synth):
    run_root, summary_path, workdirs = synth
    res = build_audio_evaluation(summary_path, workdirs)
    assert res["label_scheme"] == AUDIO_LABEL_SCHEME_NAME
    assert set(res.keys()) == {
        "label_scheme", "test_sample_count", "seed_summaries", "seed_mean",
        "per_class", "confusion_matrix_total", "top_confusions", "per_subject",
        "majority_baseline",
    }
    assert res["majority_baseline"] is None
    assert res["test_sample_count"] == 16


def test_seed_mean_matches_hand_mean(synth):
    run_root, summary_path, workdirs = synth
    res = build_audio_evaluation(summary_path, workdirs)
    expected = [float(m["test_metrics"]["macro_f1"])
                for m in _pred_for_seed_metrics(workdirs)]
    assert math.isclose(res["seed_mean"]["macro_f1"], float(np.mean(expected)))
    assert math.isclose(
        res["seed_mean"]["macro_f1_std"], float(np.std(expected)))
    assert len(res["seed_summaries"]) == 3
    assert [s["seed"] for s in res["seed_summaries"]] == SEEDS


def _pred_for_seed_metrics(workdirs: dict[int, Path]) -> list[dict]:
    return [
        json.loads((workdirs[seed] / "metrics.json").read_text(encoding="utf-8"))
        for seed in sorted(workdirs)
    ]


def test_confusion_total_is_sum(synth):
    run_root, summary_path, workdirs = synth
    res = build_audio_evaluation(summary_path, workdirs)
    expected = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for m in _pred_for_seed_metrics(workdirs):
        expected += np.asarray(m["test_metrics"]["confusion_matrix"], dtype=int)
    np.testing.assert_array_equal(res["confusion_matrix_total"], expected)
    assert int(np.sum(res["confusion_matrix_total"])) == 3 * 16


def test_top_confusions_sorted_excludes_diagonal(synth):
    run_root, summary_path, workdirs = synth
    res = build_audio_evaluation(summary_path, workdirs)
    counts = [c["count"] for c in res["top_confusions"]]
    assert counts == sorted(counts, reverse=True)
    for c in res["top_confusions"]:
        assert c["true_class"] != c["pred_class"]
        assert c["count"] > 0
    assert len(res["top_confusions"]) <= 10


def test_per_subject_counts_and_classes(synth):
    run_root, summary_path, workdirs = synth
    res = build_audio_evaluation(summary_path, workdirs)
    assert [p["subject_id"] for p in res["per_subject"]] == ["P01", "P02"]
    for p in res["per_subject"]:
        assert p["sample_count"] == 8
        assert p["classes_present"] == list(SIX_CLASS_NAMES)
        assert sum(p["class_support"]) == 8
        assert math.isclose(p["fixed_macro_f1_mean"], p["present_macro_f1_mean"])


def test_per_subject_macro_f1_defaults_fixed_when_full_coverage(synth):
    run_root, summary_path, workdirs = synth
    res = build_audio_evaluation(summary_path, workdirs)
    # full 6-class coverage for every subject -> fixed == present
    for p in res["per_subject"]:
        assert all(v > 0 for v in p["class_support"])
        assert math.isclose(p["fixed_macro_f1_mean"], p["present_macro_f1_mean"])


def test_missing_predictions_raises(tmp_path):
    run_root = tmp_path / "runs"
    seed_dir = run_root / "seed_11"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "metrics.json").write_text(json.dumps({
        "seed": 11, "test_metrics": {
            "accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0,
            "per_class_precision": [0.0] * N_CLASSES,
            "per_class_recall": [0.0] * N_CLASSES,
            "per_class_f1": [0.0] * N_CLASSES,
            "class_support": [1] * N_CLASSES,
            "confusion_matrix": [[0] * N_CLASSES for _ in range(N_CLASSES)],
        },
    }), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        build_audio_evaluation(summary_path, {11: seed_dir})
