"""Produce detailed evaluation records for the video single-modality baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def class_metrics(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        confusion[true_label, pred_label] += 1
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    support = np.bincount(labels, minlength=num_classes)
    for class_id in range(num_classes):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        precision[class_id] = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall[class_id] = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1[class_id] = (
            2 * precision[class_id] * recall[class_id] / (precision[class_id] + recall[class_id])
            if precision[class_id] + recall[class_id] > 0
            else 0.0
        )
    supported = np.where(support > 0)[0]
    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": float(f1.mean()),
        "macro_f1_present": float(f1[supported].mean()) if supported.size else 0.0,
        "balanced_accuracy": float(recall[supported].mean()) if supported.size else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "class_support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(config: dict, tag: str | None) -> None:
    num_classes = len(config["class_names"])
    class_names = config["class_names"]
    artifact_dir = Path(config["artifact_dir"])
    if tag is not None:
        artifact_dir = artifact_dir / tag
    predictions_by_subject: dict[str, list[list[dict]]] = defaultdict(list)
    total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    seed_summaries = []

    for seed in config["seeds"]:
        predictions = load_records(artifact_dir / f"seed_{seed}" / "predictions_test.jsonl")
        labels = np.array([r["label"] for r in predictions], dtype=np.int64)
        preds = np.array([r["prediction"] for r in predictions], dtype=np.int64)
        metrics = class_metrics(labels, preds, num_classes)
        total_confusion += np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        seed_summaries.append(
            {
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "per_class_f1": metrics["per_class_f1"],
                "per_class_precision": metrics["per_class_precision"],
                "per_class_recall": metrics["per_class_recall"],
                "class_support": metrics["class_support"],
            }
        )
        by_subject: dict[str, list[dict]] = defaultdict(list)
        for record in predictions:
            by_subject[record["subject_id"]].append(record)
        for subject_id, records in by_subject.items():
            predictions_by_subject[subject_id].append(records)

    subjects = sorted(
        predictions_by_subject,
        key=lambda subject_id: int(subject_id[1:]),
    )
    per_subject = []
    for subject_id in subjects:
        subject_seed_metrics = []
        for records in predictions_by_subject[subject_id]:
            labels = np.array([r["label"] for r in records], dtype=np.int64)
            preds = np.array([r["prediction"] for r in records], dtype=np.int64)
            subject_seed_metrics.append(class_metrics(labels, preds, num_classes))
        fixed_f1 = np.array([m["macro_f1"] for m in subject_seed_metrics])
        present_f1 = np.array([m["macro_f1_present"] for m in subject_seed_metrics])
        support = np.array(subject_seed_metrics[0]["class_support"], dtype=np.int64)
        per_subject.append(
            {
                "subject_id": subject_id,
                "sample_count": int(support.sum()),
                "classes_present": [class_names[i] for i in range(num_classes) if support[i] > 0],
                "class_support": support.tolist(),
                "fixed_macro_f1_mean": float(np.mean(fixed_f1)),
                "fixed_macro_f1_std": float(np.std(fixed_f1, ddof=1)) if len(fixed_f1) > 1 else 0.0,
                "present_macro_f1_mean": float(np.mean(present_f1)),
                "present_macro_f1_std": float(np.std(present_f1, ddof=1)) if len(present_f1) > 1 else 0.0,
            }
        )

    per_class_f1 = np.array([s["per_class_f1"] for s in seed_summaries])
    per_class_precision = np.array([s["per_class_precision"] for s in seed_summaries])
    per_class_recall = np.array([s["per_class_recall"] for s in seed_summaries])
    top_confusions = []
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            if true_class == pred_class:
                continue
            count = int(total_confusion[true_class, pred_class])
            if count > 0:
                top_confusions.append(
                    {
                        "true_class": class_names[true_class],
                        "pred_class": class_names[pred_class],
                        "count": count,
                    }
                )
    top_confusions.sort(key=lambda item: item["count"], reverse=True)

    majority_path = artifact_dir / "majority_baseline.json"
    majority = json.loads(majority_path.read_text(encoding="utf-8")) if majority_path.exists() else None

    evaluation = {
        "label_scheme": config["label_scheme"],
        "test_sample_count": sum(seed_summaries[0]["class_support"]),
        "seed_summaries": seed_summaries,
        "seed_mean": {
            "macro_f1": float(np.mean([s["macro_f1"] for s in seed_summaries])),
            "macro_f1_std": float(np.std([s["macro_f1"] for s in seed_summaries], ddof=1))
            if len(seed_summaries) > 1
            else 0.0,
            "balanced_accuracy": float(np.mean([s["balanced_accuracy"] for s in seed_summaries])),
            "accuracy": float(np.mean([s["accuracy"] for s in seed_summaries])),
        },
        "per_class": {
            "class_names": class_names,
            "precision_mean": per_class_precision.mean(axis=0).tolist(),
            "recall_mean": per_class_recall.mean(axis=0).tolist(),
            "f1_mean": per_class_f1.mean(axis=0).tolist(),
            "f1_std": per_class_f1.std(axis=0, ddof=1).tolist() if len(per_class_f1) > 1 else [0.0] * num_classes,
            "support": seed_summaries[0]["class_support"],
        },
        "confusion_matrix_total": total_confusion.tolist(),
        "top_confusions": top_confusions[:10],
        "per_subject": per_subject,
        "majority_baseline": majority,
    }
    (artifact_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_name = (
        f"video_baseline_confusion_{tag}.png"
        if tag is not None
        else "video_baseline_confusion.png"
    )
    figure_path = Path(config["data_root"]) / "reports" / figure_name
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.imshow(total_confusion, cmap="Blues")
    axis.set_xticks(range(num_classes), class_names, rotation=40, ha="right")
    axis.set_yticks(range(num_classes), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Distraction video baseline total confusion")
    for row in range(num_classes):
        for col in range(num_classes):
            axis.text(col, row, int(total_confusion[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(json.dumps(evaluation["seed_mean"], ensure_ascii=False, indent=2))
    print(json.dumps(evaluation["per_class"], ensure_ascii=False, indent=2))
    print("top_confusions", json.dumps(evaluation["top_confusions"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="distraction_video/configs/distraction_video_baseline_v1.json",
    )
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    evaluate(config, args.tag)


if __name__ == "__main__":
    main()
