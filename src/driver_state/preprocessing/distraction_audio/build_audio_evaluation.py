"""Derive a video-v4-compatible ``evaluation.json`` from an audio baseline.

The distraction-video module (陈星宇) publishes an ``evaluation.json`` with a
fixed schema (label_scheme / test_sample_count / seed_summaries / seed_mean /
per_class / confusion_matrix_total / top_confusions / per_subject /
majority_baseline) so downstream fusion readers can consume either modality the
same way. This module re-derives that exact structure for the audio baseline from
the per-seed ``metrics.json`` (which carries test_metrics.confusion_matrix,
per_class metrics and per_subject_test_macro_f1) plus the top-level ``summary.json``.

Only the values are recomputed from the audio runs; the contract (keys, naming,
class order, subject set) is hard-wired so the output is byte-compatible in shape
with ``distraction_video/.../evaluation.json``. No fusion logic lives here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from driver_state.preprocessing.distraction_audio.fusion_labels import (
    AUDIO_LABEL_SCHEME_NAME,
    SIX_CLASS_NAMES,
)


def _column_mean(values: list[list[float]]) -> list[float]:
    """Mean of each column (per-class) across a list of seed runs."""
    return [float(np.mean(x)) for x in values]


def _column_std(values: list[list[float]]) -> list[float]:
    """Population std (ddof=0) of each column across a list of seed runs."""
    return [float(np.std(x)) for x in values]


def _per_class_f1(preds: list[int], true: list[int], n_classes: int) -> list[float]:
    """Per-class F1 for a single prediction/ground-truth block."""
    out: list[float] = []
    for c in range(n_classes):
        tp = sum(1 for p, t in zip(preds, true) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, true) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, true) if p != c and t == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(f1)
    return out


def build_audio_evaluation(
    summary_path: Path | str,
    workdirs: dict[int, Path | str],
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Assemble an evaluation.json mirroring the video v4 structure.

    ``workdirs`` maps each seed to its run directory containing ``metrics.json``
    (and, for per-subject detail, ``predictions_test.jsonl`` when present).
    """
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    seeds = sorted(workdirs)
    seed_metrics: list[dict[str, Any]] = []
    seed_workdirs: dict[int, Path] = {}
    for seed in seeds:
        workdir = Path(workdirs[seed])
        seed_workdirs[seed] = workdir
        metrics = json.loads((workdir / "metrics.json").read_text(encoding="utf-8"))
        seed_metrics.append(metrics)

    class_names = list(SIX_CLASS_NAMES)
    n_classes = len(class_names)

    seed_summaries: list[dict[str, Any]] = []
    for metrics in seed_metrics:
        t = metrics["test_metrics"]
        seed_summaries.append({
            "seed": int(metrics["seed"]),
            "accuracy": float(t["accuracy"]),
            "macro_f1": float(t["macro_f1"]),
            "balanced_accuracy": float(t["balanced_accuracy"]),
            "per_class_f1": [float(v) for v in t["per_class_f1"]],
            "per_class_precision": [float(v) for v in t["per_class_precision"]],
            "per_class_recall": [float(v) for v in t["per_class_recall"]],
            "class_support": [int(v) for v in t["class_support"]],
        })

    macro_f1s = [float(m["test_metrics"]["macro_f1"]) for m in seed_metrics]
    bal_accs = [float(m["test_metrics"]["balanced_accuracy"]) for m in seed_metrics]
    accs = [float(m["test_metrics"]["accuracy"]) for m in seed_metrics]
    seed_mean = {
        "macro_f1": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "balanced_accuracy": float(np.mean(bal_accs)),
        "accuracy": float(np.mean(accs)),
    }

    per_class = {
        "class_names": class_names,
        "precision_mean": _column_mean([
            [float(v) for v in m["test_metrics"]["per_class_precision"]] for m in seed_metrics
        ]),
        "recall_mean": _column_mean([
            [float(v) for v in m["test_metrics"]["per_class_recall"]] for m in seed_metrics
        ]),
        "f1_mean": _column_mean([
            [float(v) for v in m["test_metrics"]["per_class_f1"]] for m in seed_metrics
        ]),
        "f1_std": _column_std([
            [float(v) for v in m["test_metrics"]["per_class_f1"]] for m in seed_metrics
        ]),
        "support": [int(v) for v in seed_metrics[0]["test_metrics"]["class_support"]],
    }

    cm_total = np.zeros((n_classes, n_classes), dtype=int)
    for metrics in seed_metrics:
        cm_total += np.asarray(metrics["test_metrics"]["confusion_matrix"], dtype=int)
    confusion_matrix_total = cm_total.tolist()

    off_diagonal = []
    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            if true_idx == pred_idx:
                continue
            count = int(cm_total[true_idx][pred_idx])
            if count > 0:
                off_diagonal.append((class_names[true_idx], class_names[pred_idx], count))
    off_diagonal.sort(key=lambda item: (-item[2], item[0], item[1]))
    top_confusions = [
        {"true_class": true_c, "pred_class": pred_c, "count": count}
        for true_c, pred_c, count in off_diagonal[:10]
    ]

    # Per-subject macro-F1 over the shared test subject set, recomputed from the
    # per-seed predictions so we also get sample_count / class_support and can
    # report both fixed (all classes) and present (only classes seen) macro-F1.
    # Every audio baseline run emits predictions_test.jsonl; require it so no
    # fabricated values are ever emitted.
    for seed in seeds:
        if not (seed_workdirs[seed] / "predictions_test.jsonl").is_file():
            raise ValueError(f"missing predictions_test.jsonl for seed {seed}")

    subjects: set[str] = set()
    per_seed_subject_f1: dict[int, dict[str, float]] = {}
    subject_positions: dict[str, list[int]] = defaultdict(lambda: [0] * n_classes)

    for seed in seeds:
        workdir = seed_workdirs[seed]
        by_subject: dict[str, list[int]] = defaultdict(list)
        by_subject_true: dict[str, list[int]] = defaultdict(list)
        for line in (workdir / "predictions_test.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            by_subject[rec["subject_id"]].append(int(rec["prediction"]))
            by_subject_true[rec["subject_id"]].append(int(rec["label"]))
            subjects.add(rec["subject_id"])
        f1: dict[str, float] = {}
        for subj, preds in by_subject.items():
            true = by_subject_true[subj]
            f1[subj] = float(np.mean(_per_class_f1(preds, true, n_classes)))
        per_seed_subject_f1[int(seed)] = f1

    # Per-subject support / sample counts from the first seed's predictions.
    first_pred = seed_workdirs[seeds[0]] / "predictions_test.jsonl"
    for line in first_pred.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        subject_positions[rec["subject_id"]][int(rec["label"])] += 1
    subject_samples = {subj: sum(vals) for subj, vals in subject_positions.items()}

    per_subject = []
    for subj in sorted(subjects):
        raw = [per_seed_subject_f1[seed].get(subj, 0.0)
               for seed in seeds if subj in per_seed_subject_f1[seed]]
        mean_f1 = float(np.mean(raw)) if raw else 0.0
        std_f1 = float(np.std(raw)) if raw else 0.0
        support = subject_positions.get(subj, [0] * n_classes)
        classes_present = [class_names[i] for i in range(n_classes) if support[i] > 0]
        # Present-class macro-F1: average only over classes that actually occur in
        # this subject's test labels. When a subject has full 6-class coverage
        # (the case for the audio test set), fixed and present coincide; we still
        # emit both keys for schema parity (mirrors the video v4 contract).
        if support.count(0) == 0:
            present_mean, present_std = mean_f1, std_f1
        else:
            per_seed_present = []
            for seed in seeds:
                preds = []
                true = []
                for line in (seed_workdirs[seed] / "predictions_test.jsonl").read_text(
                        encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec["subject_id"] != subj:
                        continue
                    preds.append(int(rec["prediction"]))
                    true.append(int(rec["label"]))
                present_indices = [i for i, c in enumerate(support) if support[i] > 0]
                f1s = _per_class_f1(preds, true, n_classes)
                per_seed_present.append(float(np.mean([f1s[i] for i in present_indices])))
            present_mean = float(np.mean(per_seed_present)) if per_seed_present else 0.0
            present_std = float(np.std(per_seed_present)) if per_seed_present else 0.0
        per_subject.append({
            "subject_id": subj,
            "sample_count": subject_samples.get(subj, 0),
            "classes_present": classes_present,
            "class_support": [int(v) for v in support],
            "fixed_macro_f1_mean": mean_f1,
            "fixed_macro_f1_std": std_f1,
            "present_macro_f1_mean": present_mean,
            "present_macro_f1_std": present_std,
        })

    test_sample_count = int(sum(subject_samples.values()))

    result: dict[str, Any] = {
        "label_scheme": AUDIO_LABEL_SCHEME_NAME,
        "test_sample_count": test_sample_count,
        "seed_summaries": seed_summaries,
        "seed_mean": seed_mean,
        "per_class": per_class,
        "confusion_matrix_total": confusion_matrix_total,
        "top_confusions": top_confusions,
        "per_subject": per_subject,
        "majority_baseline": None,
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--seeds", required=True,
                        help="comma-separated seed values matching run dir names")
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="directory that contains one subdir per seed (e.g. seed_11)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    workdirs = {seed: args.run_dir / f"seed_{seed}" for seed in seeds}
    try:
        result = build_audio_evaluation(args.summary, workdirs, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["seed_mean"], ensure_ascii=False, indent=2))
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
