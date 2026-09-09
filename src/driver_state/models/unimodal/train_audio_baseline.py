"""Train the video-aligned 6-class audio single-modality baseline (GRU).

Mirrors distraction_video baseline v4 semantics: frozen PANNs token features
[5, 2048] -> Linear proj -> BiGRU -> mean+max+last pooling -> 6-class head.
Same seeds / early stopping / selection metric as the video baseline so the two
modalities are comparable and ready for feature-level fusion later.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

NUM_CLASSES = 6
HIDDEN_DIM = 192
DROPOUT = 0.25
POOLING = "mean_max_last"
BIDIRECTIONAL = True
LR = 3e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
SEEDS = (11, 22, 33)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset):
    def __init__(self, samples: list[dict], feature_root: Path):
        self.items = []
        for s in samples:
            path = feature_root / Path(s["feature_path"])
            with np.load(path, allow_pickle=False) as z:
                x = z["x"].astype(np.float32)
                mask = z["valid_mask"].astype(bool)
            self.items.append({
                "sample_id": s["sample_id"],
                "subject_id": s["subject_id"],
                "label": int(s["label_id"]),
                "x": x,
                "valid_mask": mask,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {k: item[k] for k in ("sample_id", "subject_id", "label", "x", "valid_mask")}


def collate(batch: list[dict]) -> dict:
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "x": torch.stack([torch.as_tensor(b["x"]) for b in batch], dim=0),
        "valid_mask": torch.stack([torch.as_tensor(b["valid_mask"]) for b in batch], dim=0),
    }


class AudioGRUBaseline(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2 * 3, num_classes)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        x = self.proj(x)
        x, _ = self.gru(x)
        mask = valid_mask.unsqueeze(-1).float()
        counts = mask.sum(dim=1).clamp(min=1.0)
        masked = x * mask
        mean_pool = masked.sum(dim=1) / counts
        max_pool = (masked + (1.0 - mask) * -1e9).max(dim=1).values
        lengths = valid_mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        last_pool = x[torch.arange(x.shape[0], device=x.device), last_idx]
        pooled = torch.cat([mean_pool, max_pool, last_pool], dim=-1)
        logits = self.head(self.dropout(pooled))
        return {"logits": logits}


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray,
                    class_names: list[str]) -> dict:
    preds = probabilities.argmax(axis=1)
    per_class_precision, per_class_recall, per_class_f1 = [], [], []
    for c in range(len(class_names)):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    support = [int((labels == c).sum()) for c in range(len(class_names))]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "class_support": support,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
    }


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    records: list[dict] = []
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["valid_mask"].to(device))["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(batch["label"].numpy())
            for i, sample_id in enumerate(batch["sample_id"]):
                records.append({
                    "sample_id": sample_id,
                    "subject_id": batch["subject_id"][i],
                    "label": int(batch["label"][i]),
                    "probabilities": probs[i].tolist(),
                    "prediction": int(probs[i].argmax()),
                })
    return np.concatenate(probs_all, 0), np.concatenate(labels_all, 0), records


def load_rows(metadata_csv: Path) -> list[dict]:
    with metadata_csv.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_seed(seed: int, rows: list[dict], feature_root: Path, class_names: list[str],
             artifact_dir: Path, device: torch.device, config: dict) -> dict:
    set_seed(seed)
    split_map = {r["sample_id"]: r["split"] for r in rows}
    by_split = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split[r["split"]].append(r)

    def loader(name: str, shuffle: bool):
        ds = FeatureDataset(by_split[name], feature_root)
        gen = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate,
                          generator=gen)

    train_loader = loader("train", True)
    val_loader = loader("val", False)
    test_loader = loader("test", False)

    input_dim = int(rows[0].get("feature_shape", "[5, 2048]")[1:-1].split(",")[1])
    model = AudioGRUBaseline(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = artifact_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    best_val = -1.0
    best_epoch = -1
    patience = 0
    best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["x"].to(device), batch["valid_mask"].to(device))["logits"]
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["label"])
            total += len(batch["label"])
        train_loss = total_loss / max(total, 1)
        probs, labels, _ = predict_loader(model, val_loader, device)
        val_metrics = compute_metrics(labels, probs, class_names)
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")
    probs_t, labels_t, test_records = predict_loader(model, test_loader, device)
    test_metrics = compute_metrics(labels_t, probs_t, class_names)
    for rec, label in zip(test_records, labels_t):
        rec["seed"] = seed
        rec["split"] = "test"
    pred_path = run_dir / "predictions_test.jsonl"
    pred_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test_records) + "\n",
                         encoding="utf-8")
    metrics = {"seed": seed, "best_epoch": best_epoch, "best_val_macro_f1": best_val,
               "train_loss_last": train_loss, "val_metrics": val_metrics,
               "test_metrics": test_metrics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    # per-subject macro f1 on test
    subj: dict[str, list[int]] = defaultdict(list)
    subj_label: dict[str, list[int]] = defaultdict(list)
    for rec in test_records:
        subj[rec["subject_id"]].append(rec["prediction"])
        subj_label[rec["subject_id"]].append(rec["label"])
    per_subject = {s: float(f1_score(subj_label[s], subj[s], average="macro", zero_division=0))
                   for s in subj}
    metrics["per_subject_test_macro_f1"] = per_subject
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"seed {seed}: best_val_f1={best_val:.4f} test_macro_f1={test_metrics['macro_f1']:.4f} "
          f"epochs={best_epoch}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--label-scheme", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="11,22,33")
    args = parser.parse_args(argv)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    scheme = json.loads(args.label_scheme.read_text(encoding="utf-8"))
    class_names = list(scheme["class_names"])
    seeds = [int(s) for s in args.seeds.split(",")]
    config = {
        "label_scheme": scheme.get("label_scheme"),
        "class_names": class_names,
        "model": "AudioGRUBaseline", "input_dim": 2048, "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT, "pooling": POOLING, "bidirectional": BIDIRECTIONAL,
        "optimizer": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "effective_batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE, "selection_metric": "val_clip_macro_f1",
        "seeds": seeds, "device": str(device),
        "feature_version": "panns_cnn14_16k_v1",
    }
    rows = load_rows(args.metadata)
    per_seed: list[dict] = []
    for seed in seeds:
        per_seed.append(run_seed(seed, rows, args.feature_root, class_names,
                                 args.artifact_dir, device, config))
    test_mf1 = [m["test_metrics"]["macro_f1"] for m in per_seed]
    test_ba = [m["test_metrics"]["balanced_accuracy"] for m in per_seed]
    test_acc = [m["test_metrics"]["accuracy"] for m in per_seed]
    per_class = np.mean([[m["test_metrics"]["per_class_f1"]] for m in per_seed], axis=0)[0].tolist()
    subjects = sorted({s for m in per_seed for s in m["per_subject_test_macro_f1"]})
    per_subject_mean = {s: float(np.mean([m["per_subject_test_macro_f1"][s] for m in per_seed]))
                        for s in subjects}
    summary = {
        "baseline_version": "v1",
        "model": "AudioGRUBaseline",
        "seeds": seeds,
        "test_macro_f1_mean": float(np.mean(test_mf1)),
        "test_macro_f1_std": float(np.std(test_mf1)),
        "test_balanced_accuracy_mean": float(np.mean(test_ba)),
        "test_accuracy_mean": float(np.mean(test_acc)),
        "per_class_test_f1_mean": per_class,
        "per_subject_test_macro_f1_mean": per_subject_mean,
        "seed_results": per_seed,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    lines = [
        "# Distraction Audio Baseline v1 Result (胡煦轩)",
        "",
        f"- Task: DCPT distraction / audio / 6-class video-aligned set (714 clips).",
        f"- Features: frozen PANNs Cnn14_16k `[5, 2048]` (feature_version panns_cnn14_16k_v1).",
        f"- Model: AudioGRUBaseline (BiGRU hidden 192, dropout 0.25, mean+max+last), "
        f"AdamW {LR}/{WEIGHT_DECAY}, batch {BATCH_SIZE}, max_epochs {MAX_EPOCHS}, "
        f"patience {PATIENCE}, selection val_clip_macro_f1.",
        "",
        "## Test Results",
        "",
        "| Seed | Best val Macro-F1 | Test Macro-F1 | Test balanced accuracy | Test accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in per_seed:
        t = m["test_metrics"]
        lines.append(f"| {m['seed']} | {m['best_val_macro_f1']:.4f} | {t['macro_f1']:.4f} "
                     f"| {t['balanced_accuracy']:.4f} | {t['accuracy']:.4f} |")
    lines += [
        "",
        f"Mean test Macro-F1: {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}.",
        "",
        "## Caveats",
        "",
        "- Provisional 6-class labels and 24/8/8 subject split (mirror of the video module).",
        "- Official results require the shared manifest and split frozen by the project lead.",
        "- Fusion with the video baseline is a separate step (feature-level, owned by the fusion "
        "lead); this module only guarantees a 1:1 sample alignment.",
        "",
    ]
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary: {args.summary_out}")
    print(f"report:  {args.report_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())