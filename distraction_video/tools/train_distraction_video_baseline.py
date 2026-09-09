"""Train a GRU video single-modality baseline on cached R3D-18 features."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_index(index_path: Path) -> dict[str, str]:
    index = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index[record["sample_id"]] = record["path"]
    return index


class VideoFeatureDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        feature_dir: Path,
        index: dict[str, str],
        normalize_features: str = "none",
    ):
        self.items = []
        for sample in samples:
            rel_path = index[sample["sample_id"]]
            feature_path = feature_dir / Path(rel_path).name
            npz = np.load(feature_path)
            x = npz["x"].astype(np.float32)
            if normalize_features == "l2":
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                x = x / np.maximum(norms, 1e-6)
            self.items.append(
                {
                    "sample_id": sample["sample_id"],
                    "subject_id": sample["subject_id"],
                    "label": int(sample["label_id"]),
                    "x": x,
                    "valid_mask": npz["valid_mask"].astype(bool),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "sample_id": item["sample_id"],
            "subject_id": item["subject_id"],
            "x": item["x"],
            "valid_mask": item["valid_mask"],
            "label": item["label"],
        }


def collate_batch(batch: list[dict]) -> dict:
    sample_ids = [item["sample_id"] for item in batch]
    subject_ids = [item["subject_id"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    x = torch.stack([torch.as_tensor(item["x"]) for item in batch], dim=0)
    valid_mask = torch.stack([torch.as_tensor(item["valid_mask"]) for item in batch], dim=0)
    return {
        "sample_id": sample_ids,
        "subject_id": subject_ids,
        "x": x,
        "valid_mask": valid_mask,
        "label": labels,
    }


class BaselineGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool = False,
        pooling: str = "mean_last",
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        if pooling == "attention":
            self.attention = nn.Linear(hidden_dim * directions, 1)
            pool_dim = hidden_dim * directions
        elif pooling == "mean_last":
            pool_dim = hidden_dim * directions * 2
        elif pooling == "mean_max_last":
            pool_dim = hidden_dim * directions * 3
        elif pooling == "mean":
            pool_dim = hidden_dim * directions
        else:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        x = self.proj(x)
        x, _ = self.gru(x)
        mask = valid_mask.unsqueeze(-1).float()
        counts = mask.sum(dim=1).clamp(min=1.0)
        masked = x * mask
        mean_pool = masked.sum(dim=1) / counts

        if self.pooling == "mean":
            pooled = mean_pool
        elif self.pooling == "mean_last":
            lengths = valid_mask.sum(dim=1).long()
            last_idx = (lengths - 1).clamp(min=0)
            last_pool = x[torch.arange(x.shape[0], device=x.device), last_idx]
            pooled = torch.cat([mean_pool, last_pool], dim=-1)
        elif self.pooling == "mean_max_last":
            max_pool = (masked + (1.0 - mask) * (-1e9)).max(dim=1).values
            lengths = valid_mask.sum(dim=1).long()
            last_idx = (lengths - 1).clamp(min=0)
            last_pool = x[torch.arange(x.shape[0], device=x.device), last_idx]
            pooled = torch.cat([mean_pool, max_pool, last_pool], dim=-1)
        elif self.pooling == "attention":
            scores = self.attention(x).squeeze(-1)
            scores = scores.masked_fill(~valid_mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        else:
            raise ValueError(f"unsupported pooling: {self.pooling}")
        return {"logits": self.head(pooled)}


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        pooling: str = "mean_last",
        num_layers: int = 2,
        num_heads: int = 4,
        positional: bool = True,
    ):
        super().__init__()
        self.positional = positional
        self.pooling = pooling
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, 10, hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        x = self.proj(x)
        if self.positional:
            x = x + self.position_embedding[:, : x.shape[1], :]
        x = self.encoder(x, src_key_padding_mask=~valid_mask)
        mask = valid_mask.unsqueeze(-1).float()
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (x * mask).sum(dim=1) / counts
        lengths = valid_mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        last_pool = x[torch.arange(x.shape[0], device=x.device), last_idx]
        pooled = torch.cat([mean_pool, last_pool], dim=-1)
        return {"logits": self.head(pooled)}


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, num_classes: int) -> dict:
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        confusion[true_label, pred_label] += 1

    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
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

    supported = np.where(np.bincount(labels, minlength=num_classes) > 0)[0]
    balanced_accuracy = float(recall[supported].mean()) if supported.size else 0.0
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": balanced_accuracy,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "class_support": np.bincount(labels, minlength=num_classes).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[dict, dict]:
    model.eval()
    all_probs = []
    all_labels = []
    records = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["valid_mask"].to(device))["logits"]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probabilities)
            all_labels.append(batch["label"].numpy())
            for idx, sample_id in enumerate(batch["sample_id"]):
                records.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": batch["subject_id"][idx],
                        "label": int(batch["label"][idx]),
                        "probabilities": probabilities[idx].tolist(),
                        "prediction": int(probabilities[idx].argmax()),
                    }
                )
    probabilities = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(labels, probabilities, num_classes)
    return metrics, records


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch["x"].to(device), batch["valid_mask"].to(device))["logits"]
        loss = loss_fn(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(batch["label"])
        total += len(batch["label"])
    return total_loss / total


def run_seed(config: dict, samples: list[dict], split_map: dict, device: torch.device, seed: int) -> dict:
    set_seed(seed)
    num_classes = len(config["class_names"])
    feature_dir = Path(config["feature_dir"])
    index = load_index(Path(config["feature_index_path"]))

    train_samples = [s for s in samples if split_map[s["subject_id"]] == "train"]
    val_samples = [s for s in samples if split_map[s["subject_id"]] == "val"]
    test_samples = [s for s in samples if split_map[s["subject_id"]] == "test"]

    generator = torch.Generator().manual_seed(seed)
    normalize_features = config.get("normalize_features", "none")
    train_dataset = VideoFeatureDataset(train_samples, feature_dir, index, normalize_features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["effective_batch_size"]),
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        VideoFeatureDataset(val_samples, feature_dir, index, normalize_features),
        batch_size=int(config["effective_batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        VideoFeatureDataset(test_samples, feature_dir, index, normalize_features),
        batch_size=int(config["effective_batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )

    input_dim = train_dataset[0]["x"].shape[1]
    model_type = config.get("model_type", "gru")
    if model_type == "gru":
        model = BaselineGRU(
            input_dim=input_dim,
            hidden_dim=int(config["hidden_dim"]),
            num_classes=num_classes,
            dropout=float(config["dropout"]),
            bidirectional=bool(config.get("bidirectional", False)),
            pooling=config.get("pooling", "mean_last"),
        ).to(device)
    elif model_type == "transformer":
        model = BaselineTransformer(
            input_dim=input_dim,
            hidden_dim=int(config["hidden_dim"]),
            num_classes=num_classes,
            dropout=float(config["dropout"]),
            pooling=config.get("pooling", "mean_last"),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            positional=bool(config.get("positional", True)),
        ).to(device)
    else:
        raise ValueError(f"unsupported model_type: {model_type}")
    focal_gamma = float(config.get("focal_gamma", 0.0))
    if focal_gamma > 0:
        def focal_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(logits, labels, reduction="none")
            probabilities = torch.exp(-ce)
            return ((1.0 - probabilities) ** focal_gamma * ce).mean()

        loss_fn = focal_loss_fn
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    run_dir = Path(config["artifact_dir"]) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = -1
    patience = 0

    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics, _ = evaluate(model, val_loader, device, num_classes)
        val_f1 = val_metrics["macro_f1"]
        print(f"seed={seed} epoch={epoch} train_loss={train_loss:.4f} val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "config": config,
                },
                run_dir / "best.pt",
            )
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                print(f"seed={seed} early stop at epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_records = evaluate(model, val_loader, device, num_classes)

    test_metrics = None
    test_records = []
    if not config.get("val_only", False):
        test_metrics, test_records = evaluate(model, test_loader, device, num_classes)

    majority_class = int(Counter(s["label_id"] for s in train_samples).most_common(1)[0][0])
    majority_test = {
        "class": majority_class,
        "accuracy": float(np.mean([s["label_id"] == majority_class for s in test_samples])),
    }

    predictions_path = run_dir / "predictions_test.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in sorted(test_records, key=lambda item: item["sample_id"]):
            record["seed"] = seed
            record["split"] = "test"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if config.get("val_only", False):
        predictions_path = run_dir / "predictions_val.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for record in sorted(val_records, key=lambda item: item["sample_id"]):
                record["seed"] = seed
                record["split"] = "val"
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "train_loss": float(train_loss),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "majority_baseline_test": majority_test,
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("seed", "best_epoch", "best_val_macro_f1", "test_metrics") if k in result}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="distraction_video/configs/distraction_video_baseline_v1.json",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--bidirectional", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--positional", type=str, default=None)
    parser.add_argument("--normalize-features", type=str, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.tag is not None:
        config["artifact_dir"] = str(Path(config["artifact_dir"]) / args.tag)
    if args.val_only:
        config["val_only"] = True
    if args.pooling is not None:
        config["pooling"] = args.pooling
    if args.bidirectional is not None:
        config["bidirectional"] = args.bidirectional.lower() in ("1", "true", "yes")
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim
    if args.dropout is not None:
        config["dropout"] = args.dropout
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.model_type is not None:
        config["model_type"] = args.model_type
    if args.num_layers is not None:
        config["num_layers"] = args.num_layers
    if args.num_heads is not None:
        config["num_heads"] = args.num_heads
    if args.positional is not None:
        config["positional"] = args.positional.lower() in ("1", "true", "yes")
    if args.normalize_features is not None:
        config["normalize_features"] = args.normalize_features
    if args.focal_gamma is not None:
        config["focal_gamma"] = args.focal_gamma

    samples = [
        json.loads(line)
        for line in Path(config["training_samples_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        samples = samples[: args.limit]
    split_record = json.loads(Path(config["split_path"]).read_text(encoding="utf-8"))
    split_map = split_record["splits"]

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    results = []
    for seed in config["seeds"]:
        results.append(run_seed(config, samples, split_map, device, int(seed)))

    summary_results = []
    for r in results:
        item = {
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "best_val_macro_f1": r["best_val_macro_f1"],
            "val_macro_f1": r["val_metrics"]["macro_f1"],
            "val_balanced_accuracy": r["val_metrics"]["balanced_accuracy"],
        }
        if r["test_metrics"] is not None:
            item.update(
                {
                    "test_macro_f1": r["test_metrics"]["macro_f1"],
                    "test_balanced_accuracy": r["test_metrics"]["balanced_accuracy"],
                    "test_accuracy": r["test_metrics"]["accuracy"],
                }
            )
        summary_results.append(item)
    summary = {"results": summary_results}
    if all(r["test_metrics"] is not None for r in results):
        test_f1s = [r["test_metrics"]["macro_f1"] for r in results]
        summary["test_macro_f1_mean"] = float(np.mean(test_f1s))
        summary["test_macro_f1_std"] = float(np.std(test_f1s, ddof=1)) if len(test_f1s) > 1 else 0.0
    summary_path = Path(config["artifact_dir"]) / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
