"""Train the UL-DD CAN-only GRU on frozen complete-8 train/val manifests.

This entry point deliberately has no test-manifest path and cannot evaluate
test data. It produces development train/validation results only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from driver_state.baselines.fatigue_can import CanGruBaseline
from driver_state.data import CanWindowDataset, MaskedStandardizer, make_can_collate
from driver_state.data.can import file_sha256
from driver_state.engine import (
    EarlyStoppingTracker,
    capture_environment,
    capture_git_state,
    environment_text,
    set_random_seed,
)
from driver_state.evaluation import (
    aggregate_fatigue_parents,
    classification_metrics,
    majority_class_from_train,
    per_subject_metrics,
)
from driver_state.preprocessing.fatigue_can.pipeline import CAN_FEATURE_COLUMNS


CLASS_NAMES = ("low", "medium", "high")
PROBABILITY_FIELDS = ("prob_low", "prob_medium", "prob_high")
WINDOW_PREDICTION_FIELDS = (
    "run_id", "seed", "split", "sample_id", "parent_id", "subject_id",
    "session_id", "window_index", "label_id", *PROBABILITY_FIELDS, "prediction",
)
PARENT_PREDICTION_FIELDS = (
    "run_id", "seed", "split", "parent_id", "subject_id", "session_id",
    "label_id", "window_count", *PROBABILITY_FIELDS, "prediction",
)


class CachedDataset(Dataset):
    """Cache one verified manifest dataset without changing its public samples."""

    def __init__(self, source: CanWindowDataset) -> None:
        self.split = source.split
        self.manifest_sha256 = source.manifest_sha256
        self.records = source.records
        self.samples = tuple(source[index] for index in range(len(source)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _reject_placeholders(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_placeholders(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_placeholders(nested, path=f"{path}[{index}]")
    elif isinstance(value, str) and "TO_BE_FILLED" in value:
        raise ValueError(f"unresolved placeholder at {path}")


def verify_train_val_config(config: dict[str, Any]) -> None:
    """Reject protocol drift and any test-data path in a training config."""
    _reject_placeholders(config)
    expected = {
        "protocol_version": "0.2",
        "schema_version": "0.2.0",
        "status": "development_train_val_not_formal_result",
        "task": "fatigue",
        "dataset": "UL-DD",
        "modality": "can",
        "label_scheme": "uldd_kss_4_7_v1",
        "num_classes": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if config.get("classes") != list(CLASS_NAMES):
        raise ValueError("CAN fatigue classes must be low/medium/high")
    data = config.get("data", {})
    forbidden = [key for key in data if "test" in key.lower()]
    if forbidden:
        raise ValueError(f"train/val config must not contain test data keys: {forbidden}")
    required_data = {
        "train_windows", "train_parents", "val_windows", "val_parents",
        "split_manifest", "expected_counts",
    }
    missing = sorted(required_data - set(data))
    if missing:
        raise ValueError(f"training data config is missing keys: {missing}")
    if config["input"]["shape"] != [300, 9]:
        raise ValueError("CAN input shape must be [300,9]")
    if config["input"]["feature_columns"] != list(CAN_FEATURE_COLUMNS):
        raise ValueError("CAN feature order differs from preprocessing v1")
    if config["input"]["normalization"]["fit_split"] != "train":
        raise ValueError("normalization must be fitted on train")
    model = config["model"]
    if (
        model["projection_dim"] != 64
        or model["hidden_dim"] != 64
        or model["gru_layers"] != 1
        or not math.isclose(float(model["dropout"]), 0.2)
    ):
        raise ValueError("gru_v1 architecture must remain projection64/GRU64x1/dropout0.2")
    training = config["training"]
    fixed_training = {
        "loss": "CrossEntropyLoss",
        "class_imbalance_strategy": "none_in_v1_default",
        "optimizer": "AdamW",
        "physical_batch_size": 32,
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": "none",
        "max_epochs": 100,
        "early_stopping_patience": 15,
        "selection_metric": "val_parent_macro_f1",
        "selection_tie_break": "earlier_epoch",
        "seeds": [11, 22, 33],
        "automatic_mixed_precision": False,
        "scheduler": "none",
    }
    for key, value in fixed_training.items():
        if training.get(key) != value:
            raise ValueError(f"training.{key} must be {value!r}")
    if not math.isclose(float(training["learning_rate"]), 3e-4):
        raise ValueError("training.learning_rate must be 3e-4")
    if not math.isclose(float(training["weight_decay"]), 1e-4):
        raise ValueError("training.weight_decay must be 1e-4")
    evaluation = config["evaluation"]
    if evaluation.get("test_access") is not False:
        raise ValueError("training config must explicitly lock test access")
    if evaluation.get("aggregation") != "mean_of_8_window_probability_vectors":
        raise ValueError("fatigue parent aggregation must average eight probability vectors")
    if evaluation.get("fixed_class_ids") != [0, 1, 2]:
        raise ValueError("fatigue metrics must retain fixed class IDs [0,1,2]")
    execution = config["execution"]
    if execution.get("num_workers") != 0:
        raise ValueError("gru_v1 uses num_workers=0 for the recorded Windows run")
    if execution.get("require_clean_git") is not True:
        raise ValueError("training must require a clean committed Git state")


def _resolve_under(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError("config manifest paths must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("config path escapes processed_root") from exc
    return resolved


def _verify_manifest_hashes(
    processed_root: Path, config: dict[str, Any]
) -> dict[str, str]:
    data = config["data"]
    summary_path = _resolve_under(processed_root, data["split_manifest"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError("CAN split manifest is not PASS")
    expected = {entry["path"]: entry["sha256"] for entry in summary["files"]}
    checked: dict[str, str] = {}
    for key in ("train_windows", "train_parents", "val_windows", "val_parents"):
        relative = data[key]
        if relative not in expected:
            raise ValueError(f"{relative} is not frozen by the split manifest")
        actual = file_sha256(_resolve_under(processed_root, relative))
        if actual != expected[relative]:
            raise ValueError(f"frozen manifest hash mismatch: {relative}")
        checked[relative] = actual
    return checked


def _build_model(config: dict[str, Any]) -> CanGruBaseline:
    model = config["model"]
    return CanGruBaseline(
        input_dim=int(config["input"]["shape"][1]),
        projection_dim=int(model["projection_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        num_classes=int(config["num_classes"]),
        dropout=float(model["dropout"]),
    )


def _move_inputs(
    inputs: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        modality: {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in values.items()
        }
        for modality, values in inputs.items()
    }


def _make_loader(
    dataset: Dataset,
    *,
    standardizer: MaskedStandardizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=make_can_collate(standardizer),
        generator=generator,
        drop_last=False,
    )


def _train_epoch(
    model: CanGruBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        inputs = _move_inputs(batch["inputs"], device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("training logits contain NaN or Inf")
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is NaN or Inf")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("training gradients are missing or non-finite")
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    if total_samples != len(loader.dataset):
        raise RuntimeError("training loader did not cover every configured window")
    return total_loss / total_samples


def _predict(
    model: CanGruBaseline,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with torch.no_grad():
        for batch in loader:
            inputs = _move_inputs(batch["inputs"], device)
            labels = batch["labels"].to(device)
            logits = model(inputs)["logits"]
            if not torch.isfinite(logits).all():
                raise RuntimeError("validation logits contain NaN or Inf")
            loss = loss_function(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("validation probabilities contain NaN or Inf")
            probability_array = probabilities.detach().cpu().double().numpy()
            label_array = labels.detach().cpu().numpy()
            total_loss += float(loss.detach().cpu()) * int(labels.shape[0])
            for index, sample_id in enumerate(batch["sample_id"]):
                if sample_id in sample_ids:
                    raise RuntimeError(f"duplicate validation prediction: {sample_id}")
                sample_ids.add(sample_id)
                rows.append({
                    "sample_id": sample_id,
                    "parent_id": batch["parent_id"][index],
                    "subject_id": batch["subject_id"][index],
                    "session_id": batch["session_id"][index],
                    "window_index": int(batch["window_index"][index]),
                    "split": batch["split"][index],
                    "label_id": int(label_array[index]),
                    "probabilities": probability_array[index].tolist(),
                    "prediction": int(probability_array[index].argmax()),
                })
    if len(rows) != len(loader.dataset):
        raise RuntimeError("validation prediction count differs from the frozen dataset")
    return total_loss / len(rows), rows


def _metric_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label_id"]) for row in rows]
    probabilities = [row["probabilities"] for row in rows]
    subjects = [str(row["subject_id"]) for row in rows]
    window_metrics = classification_metrics(labels, probabilities, num_classes=3)
    parent_rows = aggregate_fatigue_parents(rows, num_classes=3, expected_windows=8)
    parent_labels = [int(row["label_id"]) for row in parent_rows]
    parent_probabilities = [row["probabilities"] for row in parent_rows]
    parent_subjects = [str(row["subject_id"]) for row in parent_rows]
    return {
        "window_30s": window_metrics,
        "parent_240s": classification_metrics(
            parent_labels, parent_probabilities, num_classes=3
        ),
        "per_subject_window_30s": per_subject_metrics(
            subjects, labels, probabilities, num_classes=3
        ),
        "per_subject_parent_240s": per_subject_metrics(
            parent_subjects, parent_labels, parent_probabilities, num_classes=3
        ),
        "parent_rows": parent_rows,
    }


def _majority_bundle(
    train_labels: Sequence[int], val_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    majority_class = majority_class_from_train(train_labels, num_classes=3)
    probabilities = np.zeros((len(val_rows), 3), dtype=np.float64)
    probabilities[:, majority_class] = 1.0
    majority_rows = []
    for row, probability in zip(val_rows, probabilities, strict=True):
        majority_row = dict(row)
        majority_row["probabilities"] = probability.tolist()
        majority_row["prediction"] = majority_class
        majority_rows.append(majority_row)
    metrics = _metric_bundle(majority_rows)
    metrics.pop("parent_rows")
    return {"majority_class": majority_class, **metrics}


def _export_window_rows(
    rows: list[dict[str, Any]], *, run_id: str, seed: int
) -> list[dict[str, Any]]:
    exported = []
    for row in rows:
        probability = row["probabilities"]
        exported.append({
            "run_id": run_id,
            "seed": seed,
            "split": row["split"],
            "sample_id": row["sample_id"],
            "parent_id": row["parent_id"],
            "subject_id": row["subject_id"],
            "session_id": row["session_id"],
            "window_index": row["window_index"],
            "label_id": row["label_id"],
            "prob_low": format(float(probability[0]), ".17g"),
            "prob_medium": format(float(probability[1]), ".17g"),
            "prob_high": format(float(probability[2]), ".17g"),
            "prediction": row["prediction"],
        })
    return exported


def _export_parent_rows(
    rows: list[dict[str, Any]], *, run_id: str, seed: int
) -> list[dict[str, Any]]:
    exported = []
    for row in rows:
        probability = row["probabilities"]
        exported.append({
            "run_id": run_id,
            "seed": seed,
            "split": row["split"],
            "parent_id": row["parent_id"],
            "subject_id": row["subject_id"],
            "session_id": row["session_id"],
            "label_id": row["label_id"],
            "window_count": row["window_count"],
            "prob_low": format(float(probability[0]), ".17g"),
            "prob_medium": format(float(probability[1]), ".17g"),
            "prob_high": format(float(probability[2]), ".17g"),
            "prediction": row["prediction"],
        })
    return exported


def _write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    fields = ("actual\\predicted", *CLASS_NAMES)
    rows = []
    for class_name, values in zip(CLASS_NAMES, matrix, strict=True):
        rows.append({
            "actual\\predicted": class_name,
            **{name: int(value) for name, value in zip(CLASS_NAMES, values, strict=True)},
        })
    _write_csv(path, fields, rows)


def _class_counts(records: Sequence[Any]) -> dict[str, int]:
    counts = Counter(int(record.label_id) for record in records)
    return {CLASS_NAMES[index]: counts.get(index, 0) for index in range(3)}


def _dataset_coverage(dataset: CachedDataset) -> dict[str, Any]:
    return {
        "windows": len(dataset),
        "parents": len({record.parent_id for record in dataset.records}),
        "subjects": sorted({record.subject_id for record in dataset.records}),
        "subject_count": len({record.subject_id for record in dataset.records}),
        "sessions": sorted({record.session_id for record in dataset.records}),
        "session_count": len({record.session_id for record in dataset.records}),
        "class_counts": _class_counts(dataset.records),
    }


def _assert_expected_counts(
    train_dataset: CachedDataset,
    val_dataset: CachedDataset,
    expected: dict[str, int],
) -> None:
    train = _dataset_coverage(train_dataset)
    val = _dataset_coverage(val_dataset)
    actual = {
        "train_windows": train["windows"],
        "train_parents": train["parents"],
        "train_subjects": train["subject_count"],
        "train_sessions": train["session_count"],
        "val_windows": val["windows"],
        "val_parents": val["parents"],
        "val_subjects": val["subject_count"],
        "val_sessions": val["session_count"],
    }
    if actual != expected:
        raise RuntimeError(f"frozen cohort count mismatch: expected={expected}, actual={actual}")


def _collect_file_records(root: Path, *, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in (exclude or set())}
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded or path.suffix == ".tmp":
            continue
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return records


def _checkpoint_payload(
    model: CanGruBaseline,
    *,
    epoch: int,
    metric: float,
    seed: int,
    config: dict[str, Any],
    config_sha256: str,
    manifest_hashes: dict[str, str],
    normalizer_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": "can_gru_state_dict_v1",
        "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config["model"]),
        "input_shape": list(config["input"]["shape"]),
        "classes": list(config["classes"]),
        "seed": seed,
        "best_epoch": epoch,
        "selection_metric": "val_parent_macro_f1",
        "selection_metric_value": metric,
        "config_sha256": config_sha256,
        "manifest_hashes": dict(manifest_hashes),
        "normalizer_sha256": normalizer_sha256,
        "git_commit": git_commit,
        "formal_result": False,
        "test_manifest_accessed": False,
    }


def _load_checkpoint_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[CanGruBaseline, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _seed_readme(
    *, seed: int, best_epoch: int, epochs_ran: int, metrics: dict[str, Any]
) -> str:
    parent = metrics["validation"]["parent_240s"]
    window = metrics["validation"]["window_30s"]
    return (
        "# UL-DD CAN GRU train/val development run\n\n"
        f"- Seed: {seed}\n"
        f"- Best epoch: {best_epoch}; epochs run: {epochs_ran}\n"
        f"- Validation parent Macro-F1: {parent['macro_f1_fixed_classes']:.6f}\n"
        f"- Validation window Macro-F1: {window['macro_f1_fixed_classes']:.6f}\n"
        "- Test manifest accessed: false\n"
        "- Formal result: false\n\n"
        "This run uses the CAN-only complete-8 cohort. It is not the frozen "
        "video+CAN paired fair-comparison cohort and must not be presented as "
        "the formal multimodal baseline result.\n"
    )


def run_seed(
    *,
    experiment_id: str,
    experiment_root: Path,
    seed: int,
    config: dict[str, Any],
    config_sha256: str,
    train_dataset: CachedDataset,
    val_dataset: CachedDataset,
    standardizer: MaskedStandardizer,
    manifest_hashes: dict[str, str],
    git_state: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    set_random_seed(seed)
    run_id = f"{experiment_id}_seed{seed}"
    seed_root = experiment_root / f"seed_{seed}"
    if seed_root.exists():
        raise FileExistsError(f"refusing to overwrite seed output: {seed_root}")
    seed_root.mkdir(parents=True)
    normalizer_path = seed_root / "preprocess_state" / "normalizer.json"
    standardizer.save_json(normalizer_path)
    normalizer_sha256 = file_sha256(normalizer_path)

    training = config["training"]
    batch_size = int(training["physical_batch_size"])
    train_loader = _make_loader(
        train_dataset,
        standardizer=standardizer,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = _make_loader(
        val_dataset,
        standardizer=standardizer,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_function = nn.CrossEntropyLoss()
    tracker = EarlyStoppingTracker(patience=int(training["early_stopping_patience"]))
    checkpoint_path = seed_root / "checkpoints" / "best.pt"
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_function, device)
        val_loss, val_rows = _predict(model, val_loader, loss_function, device)
        metrics = _metric_bundle(val_rows)
        parent_metric = float(metrics["parent_240s"]["macro_f1_fixed_classes"])
        improved = tracker.update(epoch=epoch, metric=parent_metric)
        if improved:
            _atomic_torch_save(
                checkpoint_path,
                _checkpoint_payload(
                    model,
                    epoch=epoch,
                    metric=parent_metric,
                    seed=seed,
                    config=config,
                    config_sha256=config_sha256,
                    manifest_hashes=manifest_hashes,
                    normalizer_sha256=normalizer_sha256,
                    git_commit=str(git_state["commit"]),
                ),
            )
        log_row = {
            "epoch": epoch,
            "train_loss": format(train_loss, ".17g"),
            "val_loss": format(val_loss, ".17g"),
            "val_window_macro_f1": format(
                float(metrics["window_30s"]["macro_f1_fixed_classes"]), ".17g"
            ),
            "val_window_balanced_accuracy": format(
                float(metrics["window_30s"]["balanced_accuracy_supported_classes"]),
                ".17g",
            ),
            "val_parent_macro_f1": format(parent_metric, ".17g"),
            "val_parent_balanced_accuracy": format(
                float(metrics["parent_240s"]["balanced_accuracy_supported_classes"]),
                ".17g",
            ),
            "is_best": str(improved).lower(),
            "epochs_without_improvement": tracker.epochs_without_improvement,
        }
        log_rows.append(log_row)
        print(
            f"seed={seed} epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_parent_macro_f1={parent_metric:.6f} "
            f"best_epoch={tracker.best_epoch}",
            flush=True,
        )
        if tracker.should_stop:
            break

    if tracker.best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("training finished without a best checkpoint")
    log_fields = tuple(log_rows[0])
    _write_csv(seed_root / "train_val_log.csv", log_fields, log_rows)

    best_model, checkpoint = _load_checkpoint_model(checkpoint_path, config, device)
    best_val_loss, best_rows = _predict(best_model, val_loader, loss_function, device)
    best_bundle = _metric_bundle(best_rows)
    best_parent_metric = float(best_bundle["parent_240s"]["macro_f1_fixed_classes"])
    if not math.isclose(best_parent_metric, tracker.best_metric, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("reloaded checkpoint metric differs from the selected best metric")

    second_model, _ = _load_checkpoint_model(checkpoint_path, config, device)
    _, second_rows = _predict(second_model, val_loader, loss_function, device)
    first_probabilities = np.asarray([row["probabilities"] for row in best_rows])
    second_probabilities = np.asarray([row["probabilities"] for row in second_rows])
    reload_max_difference = float(np.max(np.abs(first_probabilities - second_probabilities)))
    if not np.allclose(first_probabilities, second_probabilities, rtol=0, atol=1e-12):
        raise RuntimeError("checkpoint reload changed validation probabilities")

    parent_rows = best_bundle.pop("parent_rows")
    train_labels = [int(record.label_id) for record in train_dataset.records]
    majority = _majority_bundle(train_labels, best_rows)
    class2_recall = float(best_bundle["parent_240s"]["per_class"][2]["recall"])
    metrics_report = {
        "status": "PASS",
        "formal_result": False,
        "test_manifest_accessed": False,
        "cohort": config["data"]["cohort"],
        "cohort_limitation": config["data"]["cohort_status"],
        "run_id": run_id,
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_ran": len(log_rows),
        "early_stopped": len(log_rows) < int(training["max_epochs"]),
        "best_val_loss": best_val_loss,
        "selection_metric": "val_parent_macro_f1",
        "validation": best_bundle,
        "class_2_kss_ge_7_parent_recall": class2_recall,
        "majority_baseline": majority,
        "checkpoint_reload_max_probability_difference": reload_max_difference,
    }
    _write_json(seed_root / "metrics.json", metrics_report)
    _write_csv(
        seed_root / "predictions" / "val_windows.csv",
        WINDOW_PREDICTION_FIELDS,
        _export_window_rows(best_rows, run_id=run_id, seed=seed),
    )
    _write_csv(
        seed_root / "parent_predictions" / "val_parents.csv",
        PARENT_PREDICTION_FIELDS,
        _export_parent_rows(parent_rows, run_id=run_id, seed=seed),
    )
    _write_confusion_matrix(
        seed_root / "confusion_matrix" / "val_window.csv",
        best_bundle["window_30s"]["confusion_matrix"],
    )
    _write_confusion_matrix(
        seed_root / "confusion_matrix" / "val_parent.csv",
        best_bundle["parent_240s"]["confusion_matrix"],
    )
    coverage = {
        "status": "PASS",
        "cohort": config["data"]["cohort"],
        "complete_parent_policy": config["data"]["complete_parent_policy"],
        "train": _dataset_coverage(train_dataset),
        "val": _dataset_coverage(val_dataset),
        "validation_prediction_windows": len(best_rows),
        "validation_prediction_parents": len(parent_rows),
        "missing_validation_predictions": 0,
        "duplicate_validation_predictions": 0,
        "test_manifest_accessed": False,
    }
    _write_json(seed_root / "coverage_report.json", coverage)
    _atomic_text(
        seed_root / "README.md",
        _seed_readme(
            seed=seed,
            best_epoch=int(checkpoint["best_epoch"]),
            epochs_ran=len(log_rows),
            metrics=metrics_report,
        ),
    )
    run_manifest_path = seed_root / "run_manifest.json"
    run_manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEVELOPMENT_TRAIN_VAL_COMPLETE",
        "formal_result": False,
        "task": "fatigue",
        "dataset": "UL-DD",
        "modality": "can",
        "cohort": config["data"]["cohort"],
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "selection_metric": "val_parent_macro_f1",
        "selection_metric_value": best_parent_metric,
        "test_manifest_accessed": False,
        "git": git_state,
        "config_sha256": config_sha256,
        "manifest_hashes": manifest_hashes,
        "normalizer_sha256": normalizer_sha256,
        "files": _collect_file_records(seed_root, exclude={run_manifest_path}),
    }
    _write_json(run_manifest_path, run_manifest)
    return {
        "status": "PASS",
        "run_id": run_id,
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_ran": len(log_rows),
        "early_stopped": metrics_report["early_stopped"],
        "val_window_macro_f1": float(
            best_bundle["window_30s"]["macro_f1_fixed_classes"]
        ),
        "val_window_balanced_accuracy": float(
            best_bundle["window_30s"]["balanced_accuracy_supported_classes"]
        ),
        "val_window_accuracy": float(best_bundle["window_30s"]["accuracy"]),
        "val_parent_macro_f1": best_parent_metric,
        "val_parent_balanced_accuracy": float(
            best_bundle["parent_240s"]["balanced_accuracy_supported_classes"]
        ),
        "val_parent_accuracy": float(best_bundle["parent_240s"]["accuracy"]),
        "val_parent_class2_recall": class2_recall,
        "majority_val_parent_macro_f1": float(
            majority["parent_240s"]["macro_f1_fixed_classes"]
        ),
        "checkpoint_reload_max_probability_difference": reload_max_difference,
        "validation_windows": len(best_rows),
        "validation_parents": len(parent_rows),
        "test_manifest_accessed": False,
    }


def _summary_statistics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "val_window_macro_f1",
        "val_window_balanced_accuracy",
        "val_window_accuracy",
        "val_parent_macro_f1",
        "val_parent_balanced_accuracy",
        "val_parent_accuracy",
        "val_parent_class2_recall",
        "majority_val_parent_macro_f1",
    )
    result: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in seed_results], dtype=np.float64)
        result[name] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std_ddof_1": float(values.std(ddof=1)) if len(values) > 1 else None,
            "seed_count": len(values),
        }
    return result


def _write_seed_summary_csv(path: Path, seed_results: list[dict[str, Any]]) -> None:
    fields = (
        "row_type", "seed", "best_epoch", "epochs_ran",
        "val_window_macro_f1", "val_window_balanced_accuracy", "val_window_accuracy",
        "val_parent_macro_f1", "val_parent_balanced_accuracy", "val_parent_accuracy",
        "val_parent_class2_recall", "majority_val_parent_macro_f1",
    )
    rows = [{"row_type": "seed", **row} for row in seed_results]
    numeric = fields[4:]
    rows.append({
        "row_type": "mean",
        **{
            name: float(np.mean([float(row[name]) for row in seed_results]))
            for name in numeric
        },
    })
    rows.append({
        "row_type": "sample_std_ddof_1",
        **{
            name: (
                float(np.std([float(row[name]) for row in seed_results], ddof=1))
                if len(seed_results) > 1 else ""
            )
            for name in numeric
        },
    })
    _write_csv(path, fields, rows)


def train_experiment(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    processed_root = Path(args.processed_root).resolve()
    config_path = Path(args.config).resolve()
    if not processed_root.is_dir():
        raise FileNotFoundError(f"processed_root does not exist: {processed_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_train_val_config(config)
    configured_seeds = [int(seed) for seed in config["training"]["seeds"]]
    requested_seeds = configured_seeds if args.seeds is None else [int(seed) for seed in args.seeds]
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("requested seeds must be unique")
    if any(seed not in configured_seeds for seed in requested_seeds):
        raise ValueError("requested seeds must be a subset of configured seeds 11/22/33")
    if not requested_seeds or requested_seeds[0] != 11:
        raise ValueError("seed 11 must run first as the development gate")

    git_state = capture_git_state(repository)
    if not git_state.get("commit") or git_state.get("dirty") is not False:
        raise RuntimeError("training requires a clean committed Git checkout")
    manifest_hashes = _verify_manifest_hashes(processed_root, config)
    data = config["data"]
    verify_features = bool(config["execution"]["verify_each_feature_hash_once"])
    train_source = CanWindowDataset(
        _resolve_under(processed_root, data["train_windows"]),
        feature_root=processed_root,
        split="train",
        parent_manifest=_resolve_under(processed_root, data["train_parents"]),
        verify_feature_hash=verify_features,
    )
    val_source = CanWindowDataset(
        _resolve_under(processed_root, data["val_windows"]),
        feature_root=processed_root,
        split="val",
        parent_manifest=_resolve_under(processed_root, data["val_parents"]),
        verify_feature_hash=verify_features,
    )
    if not config["execution"]["cache_features_in_memory"]:
        raise ValueError("recorded gru_v1 training requires in-memory verified feature caching")
    train_dataset = CachedDataset(train_source)
    val_dataset = CachedDataset(val_source)
    _assert_expected_counts(train_dataset, val_dataset, data["expected_counts"])
    standardizer = MaskedStandardizer.fit_can_dataset(
        train_dataset, feature_names=CAN_FEATURE_COLUMNS
    )
    if standardizer.source_manifest_sha256 != manifest_hashes[data["train_windows"]]:
        raise RuntimeError("normalizer is not bound to the frozen train manifest")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_random_seed(requested_seeds[0])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"can_gru_v1_trainval_{timestamp}"
    experiment_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else processed_root / "baselines" / "fatigue_can" / "gru_v1" / "runs" / experiment_id
    )
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {experiment_root}")
    experiment_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = file_sha256(config_path)
    resolved_config = copy.deepcopy(config)
    resolved_config["execution"]["resolved_device"] = str(device)
    resolved_config["execution"]["requested_seeds"] = requested_seeds
    _write_json(experiment_root / "resolved_config.json", resolved_config)
    _atomic_text(
        experiment_root / "environment.txt",
        environment_text(capture_environment()),
    )
    manifest_path = experiment_root / "experiment_manifest.json"
    experiment_manifest = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "formal_result": False,
        "task": "fatigue",
        "dataset": "UL-DD",
        "modality": "can",
        "cohort": data["cohort"],
        "cohort_limitation": data["cohort_status"],
        "requested_seeds": requested_seeds,
        "completed_seeds": [],
        "test_manifest_accessed": False,
        "git": git_state,
        "config_sha256": config_sha256,
        "manifest_hashes": manifest_hashes,
    }
    _write_json(manifest_path, experiment_manifest)
    seed_results: list[dict[str, Any]] = []
    try:
        for seed in requested_seeds:
            result = run_seed(
                experiment_id=experiment_id,
                experiment_root=experiment_root,
                seed=seed,
                config=config,
                config_sha256=config_sha256,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                standardizer=standardizer,
                manifest_hashes=manifest_hashes,
                git_state=git_state,
                device=device,
            )
            if result["validation_windows"] != 328 or result["validation_parents"] != 41:
                raise RuntimeError("seed gate prediction counts are not 328 windows / 41 parents")
            if result["checkpoint_reload_max_probability_difference"] > 1e-12:
                raise RuntimeError("seed gate checkpoint reload is not reproducible")
            if result["test_manifest_accessed"] is not False:
                raise RuntimeError("seed gate detected forbidden test access")
            seed_results.append(result)
            experiment_manifest["completed_seeds"] = [row["seed"] for row in seed_results]
            _write_json(manifest_path, experiment_manifest)
    except Exception as exc:
        experiment_manifest.update({
            "status": "FAILED",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "test_manifest_accessed": False,
        })
        _write_json(manifest_path, experiment_manifest)
        raise

    statistics = _summary_statistics(seed_results)
    summary = {
        "status": "PASS",
        "formal_result": False,
        "test_manifest_accessed": False,
        "experiment_id": experiment_id,
        "cohort": data["cohort"],
        "cohort_limitation": data["cohort_status"],
        "seed_results": seed_results,
        "statistics": statistics,
        "aggregation_note": "seed results are separate models; no voting or best-seed selection",
    }
    _write_json(experiment_root / "seed_summary.json", summary)
    _write_seed_summary_csv(experiment_root / "seed_summary.csv", seed_results)
    _atomic_text(
        experiment_root / "README.md",
        "# CAN GRU v1 train/val development experiment\n\n"
        "This experiment uses the CAN-only complete-8 cohort and evaluates only "
        "validation data. It did not access test data and is not a formal "
        "video+CAN fair-comparison result.\n",
    )
    experiment_manifest.update({
        "status": "DEVELOPMENT_TRAIN_VAL_COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_seeds": [row["seed"] for row in seed_results],
        "summary_file": "seed_summary.json",
        "files": _collect_file_records(experiment_root, exclude={manifest_path}),
    })
    _write_json(manifest_path, experiment_manifest)
    summary["output_root"] = str(experiment_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / "baselines"
            / "fatigue_can"
            / "gru_v1_train_val.json"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        report = train_experiment(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "formal_result": False,
            "test_manifest_accessed": False,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
