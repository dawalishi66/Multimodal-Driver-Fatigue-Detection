"""Train fatigue video/CAN fusion models on the frozen train/val cohort only."""

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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from driver_state.data import (
    FatigueVideoCanDataset,
    MaskedStandardizer,
    make_fatigue_video_can_collate,
    make_fusion_model_inputs,
)
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
from driver_state.models.mult import DualModalMulT
from driver_state.models.simple_fusion import SimpleFusion
from driver_state.preprocessing.fatigue_can.pipeline import CAN_FEATURE_COLUMNS


CLASS_NAMES = ("low", "medium", "high")
MODALITIES = ("video", "can")
PROBABILITY_FIELDS = ("prob_low", "prob_medium", "prob_high")
WINDOW_FIELDS = (
    "run_id", "seed", "split", "sample_id", "video_sample_id", "can_sample_id",
    "parent_id", "subject_id", "session_id", "window_index", "label_id",
    *PROBABILITY_FIELDS, "prediction",
)
PARENT_FIELDS = (
    "run_id", "seed", "split", "parent_id", "subject_id", "session_id",
    "label_id", "window_count", *PROBABILITY_FIELDS, "prediction",
)


class CachedDataset(Dataset):
    """Materialize verified train/val samples once for repeated seeded training."""

    def __init__(self, source: FatigueVideoCanDataset) -> None:
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


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
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


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def verify_train_val_config(config: dict[str, Any]) -> None:
    """Reject protocol drift, unverified inputs, and every test-data path."""
    _reject_placeholders(config)
    expected = {
        "config_version": "1.0.0-train-val",
        "protocol_version": "0.2",
        "schema_version": "0.2.0",
        "status": "development_train_val_not_formal_result",
        "task": "fatigue",
        "dataset": "UL-DD",
        "modalities": list(MODALITIES),
        "label_scheme": "uldd_kss_4_7_v1",
        "num_classes": 3,
        "classes": list(CLASS_NAMES),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")

    data = config.get("data", {})
    forbidden = [key for key in data if "test" in key.lower()]
    if forbidden:
        raise ValueError(f"train/val config must not contain test data keys: {forbidden}")
    required_data = {
        "cohort", "pair_manifest", "pair_manifest_sha256",
        "dataset_validation_report", "dataset_validation_report_sha256",
        "g2_preflight_report", "g2_preflight_report_sha256",
        "video_feature_version", "video_source_checkpoint_sha256",
        "can_feature_version", "sync_status", "complete_parent_policy",
        "expected_counts",
    }
    missing = sorted(required_data - set(data))
    if missing:
        raise ValueError(f"training data config is missing keys: {missing}")
    for key in (
        "pair_manifest_sha256",
        "dataset_validation_report_sha256",
        "g2_preflight_report_sha256",
        "video_source_checkpoint_sha256",
    ):
        if not _is_sha256(data[key]):
            raise ValueError(f"data.{key} is not a SHA-256 digest")
    if data["expected_counts"] != {
        "train_windows": 1272,
        "train_parents": 159,
        "train_subjects": 10,
        "train_sessions": 18,
        "val_windows": 328,
        "val_parents": 41,
        "val_subjects": 3,
        "val_sessions": 5,
    }:
        raise ValueError("unexpected frozen paired-cohort counts")
    if data["sync_status"] != "provider_documented_aligned_not_hardware_clock_verified":
        raise ValueError("synchronization evidence was overstated or changed")

    inputs = config["inputs"]
    if inputs["video"]["shape"] != [6, 96] or inputs["can"]["shape"] != [300, 9]:
        raise ValueError("training inputs must be video [6,96] and CAN [300,9]")
    if inputs["can"]["feature_columns"] != list(CAN_FEATURE_COLUMNS):
        raise ValueError("CAN feature order differs from preprocessing v1")
    for modality in MODALITIES:
        if inputs[modality]["normalization"] != "train_only_masked_zscore_v1":
            raise ValueError(f"{modality} normalization must be train-only")
        if inputs[modality]["time_reference"] != "sample_relative_0_to_30_seconds":
            raise ValueError(f"{modality} time reference changed")
    if inputs["mask_semantics"] != "true_is_valid":
        raise ValueError("mask semantics must remain True=valid")

    model = config["model"]
    if model.get("model_id") == "simple_fusion_v1":
        required_model = {
            "model_id": "simple_fusion_v1",
            "class": "SimpleFusion",
            "projection_dim": 64,
            "classifier_hidden_dim": 64,
            "dropout": 0.2,
            "time_s_used_for_prediction": False,
        }
    elif model.get("model_id") == "dual_modal_mult_v1":
        required_model = {
            "model_id": "dual_modal_mult_v1",
            "class": "DualModalMulT",
            "d_model": 30,
            "num_heads": 5,
            "cross_layers": 5,
            "memory_layers": 5,
            "dropout": 0.0,
            "enable_a_from_b": True,
            "enable_b_from_a": True,
            "causal_attention": False,
            "position_encoding": "sequence_sinusoidal",
            "time_s_used_for_prediction": False,
        }
    else:
        raise ValueError("model_id must be simple_fusion_v1 or dual_modal_mult_v1")
    if model != required_model:
        raise ValueError(f"{model.get('model_id')} development architecture changed")

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
        raise ValueError("parent aggregation must average eight window probabilities")
    if evaluation.get("fixed_class_ids") != [0, 1, 2]:
        raise ValueError("fatigue metrics must retain fixed classes 0,1,2")
    execution = config["execution"]
    if execution.get("num_workers") != 0:
        raise ValueError("the recorded Windows run requires num_workers=0")
    if execution.get("require_clean_git") is not True:
        raise ValueError("training must require a clean committed Git state")


def _resolve_under(root: Path, relative: str, *, field: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{field} must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes pair_root") from exc
    return resolved


def _verify_frozen_inputs(pair_root: Path, config: dict[str, Any]) -> dict[str, str]:
    data = config["data"]
    pairs = {
        "pair_manifest": (data["pair_manifest"], data["pair_manifest_sha256"]),
        "dataset_validation_report": (
            data["dataset_validation_report"],
            data["dataset_validation_report_sha256"],
        ),
        "g2_preflight_report": (
            data["g2_preflight_report"],
            data["g2_preflight_report_sha256"],
        ),
    }
    checked: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for key, (relative, expected_hash) in pairs.items():
        path = _resolve_under(pair_root, relative, field=key)
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"frozen {key} hash mismatch")
        paths[key] = path
        checked[key] = actual_hash
    dataset_report = json.loads(paths["dataset_validation_report"].read_text(encoding="utf-8"))
    if dataset_report.get("status") != "PASS" or dataset_report.get("test_manifest_accessed") is not False:
        raise ValueError("paired Dataset validation is not an approved train/val PASS")
    if dataset_report.get("pair_manifest_sha256") != checked["pair_manifest"]:
        raise ValueError("Dataset validation refers to a different pair manifest")
    g2_report = json.loads(paths["g2_preflight_report"].read_text(encoding="utf-8"))
    if (
        g2_report.get("status") != "PASS"
        or g2_report.get("formal_result") is not False
        or g2_report.get("metrics_evaluated") is not False
        or g2_report.get("test_manifest_accessed") is not False
    ):
        raise ValueError("required real-feature G2 preflight is not PASS")
    if g2_report.get("frozen_inputs", {}).get("pair_manifest") != checked["pair_manifest"]:
        raise ValueError("G2 preflight refers to a different pair manifest")
    return checked


def _build_model(config: dict[str, Any]) -> nn.Module:
    model = config["model"]
    input_dims = {"video": 96, "can": 9}
    if model["model_id"] == "simple_fusion_v1":
        return SimpleFusion(
            modalities=MODALITIES,
            input_dims=input_dims,
            num_classes=3,
            projection_dim=int(model["projection_dim"]),
            classifier_hidden_dim=int(model["classifier_hidden_dim"]),
            dropout=float(model["dropout"]),
        )
    return DualModalMulT(
        modalities=MODALITIES,
        input_dims=input_dims,
        num_classes=3,
        d_model=int(model["d_model"]),
        num_heads=int(model["num_heads"]),
        cross_layers=int(model["cross_layers"]),
        memory_layers=int(model["memory_layers"]),
        dropout=float(model["dropout"]),
        enable_a_from_b=bool(model["enable_a_from_b"]),
        enable_b_from_a=bool(model["enable_b_from_a"]),
        causal_attention=bool(model["causal_attention"]),
    )


def _move_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, dict[str, Tensor]]:
    inputs = make_fusion_model_inputs(batch)
    return {
        modality: {name: value.to(device) for name, value in stream.items()}
        for modality, stream in inputs.items()
    }


def _make_loader(
    dataset: Dataset,
    *,
    standardizers: Mapping[str, MaskedStandardizer],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=make_fatigue_video_can_collate(standardizers),
        generator=generator,
        drop_last=False,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        inputs = _move_inputs(batch, device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("training logits contain NaN or Inf")
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is NaN or Inf")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
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
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(_move_inputs(batch, device))["logits"]
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
                if sample_id in seen:
                    raise RuntimeError(f"duplicate validation prediction: {sample_id}")
                seen.add(sample_id)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "video_sample_id": batch["video_sample_id"][index],
                        "can_sample_id": batch["can_sample_id"][index],
                        "parent_id": batch["parent_id"][index],
                        "subject_id": batch["subject_id"][index],
                        "session_id": batch["session_id"][index],
                        "window_index": int(batch["window_index"][index]),
                        "split": batch["split"][index],
                        "label_id": int(label_array[index]),
                        "probabilities": probability_array[index].tolist(),
                        "prediction": int(probability_array[index].argmax()),
                    }
                )
    if len(rows) != len(loader.dataset):
        raise RuntimeError("validation prediction count differs from the frozen Dataset")
    return total_loss / len(rows), rows


def _metric_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label_id"]) for row in rows]
    probabilities = [row["probabilities"] for row in rows]
    subjects = [str(row["subject_id"]) for row in rows]
    parents = aggregate_fatigue_parents(rows, num_classes=3, expected_windows=8)
    parent_labels = [int(row["label_id"]) for row in parents]
    parent_probabilities = [row["probabilities"] for row in parents]
    parent_subjects = [str(row["subject_id"]) for row in parents]
    return {
        "window_30s": classification_metrics(labels, probabilities, num_classes=3),
        "parent_240s": classification_metrics(parent_labels, parent_probabilities, num_classes=3),
        "per_subject_window_30s": per_subject_metrics(
            subjects, labels, probabilities, num_classes=3
        ),
        "per_subject_parent_240s": per_subject_metrics(
            parent_subjects, parent_labels, parent_probabilities, num_classes=3
        ),
        "parent_rows": parents,
    }


def _majority_bundle(train_labels: Sequence[int], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    majority = majority_class_from_train(train_labels, num_classes=3)
    rows = []
    for row in val_rows:
        changed = dict(row)
        changed["probabilities"] = [float(index == majority) for index in range(3)]
        changed["prediction"] = majority
        rows.append(changed)
    metrics = _metric_bundle(rows)
    metrics.pop("parent_rows")
    return {"majority_class": majority, **metrics}


def _export_window_rows(rows: list[dict[str, Any]], *, run_id: str, seed: int) -> list[dict[str, Any]]:
    exported = []
    for row in rows:
        probability = row["probabilities"]
        exported.append(
            {
                "run_id": run_id,
                "seed": seed,
                **{key: row[key] for key in (
                    "split", "sample_id", "video_sample_id", "can_sample_id", "parent_id",
                    "subject_id", "session_id", "window_index", "label_id",
                )},
                "prob_low": format(float(probability[0]), ".17g"),
                "prob_medium": format(float(probability[1]), ".17g"),
                "prob_high": format(float(probability[2]), ".17g"),
                "prediction": row["prediction"],
            }
        )
    return exported


def _export_parent_rows(rows: list[dict[str, Any]], *, run_id: str, seed: int) -> list[dict[str, Any]]:
    exported = []
    for row in rows:
        probability = row["probabilities"]
        exported.append(
            {
                "run_id": run_id,
                "seed": seed,
                **{key: row[key] for key in (
                    "split", "parent_id", "subject_id", "session_id", "label_id", "window_count",
                )},
                "prob_low": format(float(probability[0]), ".17g"),
                "prob_medium": format(float(probability[1]), ".17g"),
                "prob_high": format(float(probability[2]), ".17g"),
                "prediction": row["prediction"],
            }
        )
    return exported


def _write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    fields = ("actual\\predicted", *CLASS_NAMES)
    rows = [
        {
            "actual\\predicted": class_name,
            **{name: int(value) for name, value in zip(CLASS_NAMES, values, strict=True)},
        }
        for class_name, values in zip(CLASS_NAMES, matrix, strict=True)
    ]
    _write_csv(path, fields, rows)


def _dataset_coverage(dataset: Dataset) -> dict[str, Any]:
    records = dataset.records
    counts = Counter(int(record.label_id) for record in records)
    return {
        "windows": len(dataset),
        "parents": len({record.parent_id for record in records}),
        "subjects": sorted({record.subject_id for record in records}),
        "subject_count": len({record.subject_id for record in records}),
        "sessions": sorted({record.session_id for record in records}),
        "session_count": len({record.session_id for record in records}),
        "class_counts": {CLASS_NAMES[index]: counts.get(index, 0) for index in range(3)},
    }


def _assert_expected_counts(train_dataset: Dataset, val_dataset: Dataset, expected: Mapping[str, int]) -> None:
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
    if actual != dict(expected):
        raise RuntimeError(f"frozen cohort count mismatch: expected={dict(expected)}, actual={actual}")


def _collect_file_records(root: Path, *, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in (exclude or set())}
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.resolve() not in excluded and path.suffix != ".tmp"
    ]


def _checkpoint_payload(
    model: nn.Module,
    *,
    epoch: int,
    metric: float,
    seed: int,
    config: dict[str, Any],
    config_sha256: str,
    input_hashes: Mapping[str, str],
    normalizer_hashes: Mapping[str, str],
    git_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": "fatigue_video_can_fusion_state_dict_v1",
        "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config["model"]),
        "input_shapes": {key: value["shape"] for key, value in config["inputs"].items() if key in MODALITIES},
        "classes": list(config["classes"]),
        "seed": seed,
        "best_epoch": epoch,
        "selection_metric": "val_parent_macro_f1",
        "selection_metric_value": metric,
        "config_sha256": config_sha256,
        "input_hashes": dict(input_hashes),
        "normalizer_hashes": dict(normalizer_hashes),
        "git_commit": git_commit,
        "formal_result": False,
        "test_manifest_accessed": False,
    }


def _load_checkpoint_model(path: Path, config: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = _build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def run_seed(
    *,
    experiment_id: str,
    experiment_root: Path,
    seed: int,
    config: dict[str, Any],
    config_sha256: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    standardizers: Mapping[str, MaskedStandardizer],
    input_hashes: Mapping[str, str],
    git_state: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    set_random_seed(seed)
    run_id = f"{experiment_id}_seed{seed}"
    seed_root = experiment_root / f"seed_{seed}"
    if seed_root.exists():
        raise FileExistsError(f"refusing to overwrite seed output: {seed_root}")
    seed_root.mkdir(parents=True)
    normalizer_hashes: dict[str, str] = {}
    for modality, standardizer in standardizers.items():
        path = seed_root / "preprocess_state" / f"{modality}_normalizer.json"
        standardizer.save_json(path)
        normalizer_hashes[modality] = file_sha256(path)

    training = config["training"]
    batch_size = int(training["physical_batch_size"])
    train_loader = _make_loader(
        train_dataset,
        standardizers=standardizers,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = _make_loader(
        val_dataset,
        standardizers=standardizers,
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
                    input_hashes=input_hashes,
                    normalizer_hashes=normalizer_hashes,
                    git_commit=str(git_state["commit"]),
                ),
            )
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": format(train_loss, ".17g"),
                "val_loss": format(val_loss, ".17g"),
                "val_window_macro_f1": format(
                    float(metrics["window_30s"]["macro_f1_fixed_classes"]), ".17g"
                ),
                "val_window_balanced_accuracy": format(
                    float(metrics["window_30s"]["balanced_accuracy_supported_classes"]), ".17g"
                ),
                "val_parent_macro_f1": format(parent_metric, ".17g"),
                "val_parent_balanced_accuracy": format(
                    float(metrics["parent_240s"]["balanced_accuracy_supported_classes"]), ".17g"
                ),
                "is_best": str(improved).lower(),
                "epochs_without_improvement": tracker.epochs_without_improvement,
            }
        )
        print(
            f"model={config['model']['model_id']} seed={seed} epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_parent_macro_f1={parent_metric:.6f} best_epoch={tracker.best_epoch}",
            flush=True,
        )
        if tracker.should_stop:
            break

    if tracker.best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("training finished without a best checkpoint")
    _write_csv(seed_root / "train_val_log.csv", tuple(log_rows[0]), log_rows)
    best_model, checkpoint = _load_checkpoint_model(checkpoint_path, config, device)
    best_val_loss, best_rows = _predict(best_model, val_loader, loss_function, device)
    best_bundle = _metric_bundle(best_rows)
    best_parent_metric = float(best_bundle["parent_240s"]["macro_f1_fixed_classes"])
    if not math.isclose(best_parent_metric, tracker.best_metric, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("reloaded checkpoint metric differs from selected best metric")
    second_model, _ = _load_checkpoint_model(checkpoint_path, config, device)
    _, second_rows = _predict(second_model, val_loader, loss_function, device)
    first_probabilities = np.asarray([row["probabilities"] for row in best_rows])
    second_probabilities = np.asarray([row["probabilities"] for row in second_rows])
    reload_difference = float(np.max(np.abs(first_probabilities - second_probabilities)))
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
        "task": "fatigue",
        "modalities": list(MODALITIES),
        "model_id": config["model"]["model_id"],
        "cohort": config["data"]["cohort"],
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
        "checkpoint_reload_max_probability_difference": reload_difference,
    }
    _write_json(seed_root / "metrics.json", metrics_report)
    _write_csv(
        seed_root / "predictions" / "val_windows.csv",
        WINDOW_FIELDS,
        _export_window_rows(best_rows, run_id=run_id, seed=seed),
    )
    _write_csv(
        seed_root / "parent_predictions" / "val_parents.csv",
        PARENT_FIELDS,
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
        "# Fatigue video + CAN train/val development run\n\n"
        f"- Model: {config['model']['model_id']}\n"
        f"- Seed: {seed}\n"
        f"- Best epoch: {checkpoint['best_epoch']}; epochs run: {len(log_rows)}\n"
        f"- Validation parent Macro-F1: {best_parent_metric:.6f}\n"
        "- Test manifest accessed: false\n"
        "- Formal result: false\n\n"
        "This is a fixed-cohort train/validation development result, not a final test result.\n",
    )
    run_manifest_path = seed_root / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "DEVELOPMENT_TRAIN_VAL_COMPLETE",
            "formal_result": False,
            "task": "fatigue",
            "modalities": list(MODALITIES),
            "model_id": config["model"]["model_id"],
            "cohort": config["data"]["cohort"],
            "seed": seed,
            "best_epoch": int(checkpoint["best_epoch"]),
            "selection_metric": "val_parent_macro_f1",
            "selection_metric_value": best_parent_metric,
            "test_manifest_accessed": False,
            "git": dict(git_state),
            "config_sha256": config_sha256,
            "input_hashes": dict(input_hashes),
            "normalizer_hashes": normalizer_hashes,
            "files": _collect_file_records(seed_root, exclude={run_manifest_path}),
        },
    )
    return {
        "status": "PASS",
        "run_id": run_id,
        "seed": seed,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_ran": len(log_rows),
        "early_stopped": metrics_report["early_stopped"],
        "val_window_macro_f1": float(best_bundle["window_30s"]["macro_f1_fixed_classes"]),
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
        "checkpoint_reload_max_probability_difference": reload_difference,
        "validation_windows": len(best_rows),
        "validation_parents": len(parent_rows),
        "test_manifest_accessed": False,
    }


def _summary_statistics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "val_window_macro_f1", "val_window_balanced_accuracy", "val_window_accuracy",
        "val_parent_macro_f1", "val_parent_balanced_accuracy", "val_parent_accuracy",
        "val_parent_class2_recall", "majority_val_parent_macro_f1",
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
    rows.append(
        {
            "row_type": "mean",
            **{
                name: float(np.mean([float(row[name]) for row in seed_results]))
                for name in numeric
            },
        }
    )
    rows.append(
        {
            "row_type": "sample_std_ddof_1",
            **{
                name: (
                    float(np.std([float(row[name]) for row in seed_results], ddof=1))
                    if len(seed_results) > 1
                    else ""
                )
                for name in numeric
            },
        }
    )
    _write_csv(path, fields, rows)


def train_experiment(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    pair_root = Path(args.pair_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    config_path = Path(args.config).resolve()
    if not pair_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("pair_root and dataset_root must both exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_train_val_config(config)
    fixed_seeds = list(config["training"]["seeds"])
    requested_seeds = fixed_seeds if args.seeds is None else list(args.seeds)
    if requested_seeds not in ([11], [11, 22], [11, 22, 33]):
        raise ValueError("requested seeds must be the ordered prefix 11, 22, 33")
    git_state = capture_git_state(repository)
    if config["execution"]["require_clean_git"] and git_state.get("dirty") is not False:
        raise RuntimeError("fusion training requires a clean committed Git worktree")
    input_hashes = _verify_frozen_inputs(pair_root, config)
    pair_manifest = _resolve_under(pair_root, config["data"]["pair_manifest"], field="pair_manifest")
    verify_features = bool(config["execution"]["verify_each_feature_hash"])
    train_dataset = CachedDataset(
        FatigueVideoCanDataset(
            pair_manifest,
            dataset_root=dataset_root,
            split="train",
            verify_feature_hashes=verify_features,
            cache_features=True,
        )
    )
    val_dataset = CachedDataset(
        FatigueVideoCanDataset(
            pair_manifest,
            dataset_root=dataset_root,
            split="val",
            verify_feature_hashes=verify_features,
            cache_features=True,
        )
    )
    _assert_expected_counts(train_dataset, val_dataset, config["data"]["expected_counts"])
    standardizers = {
        "video": MaskedStandardizer.fit_dataset_modality(
            train_dataset,
            modality="video",
            feature_names=tuple(f"video_feature_{index:03d}" for index in range(96)),
        ),
        "can": MaskedStandardizer.fit_dataset_modality(
            train_dataset,
            modality="can",
            feature_names=CAN_FEATURE_COLUMNS,
        ),
    }
    if any(
        standardizer.source_manifest_sha256 != input_hashes["pair_manifest"]
        for standardizer in standardizers.values()
    ):
        raise RuntimeError("a normalizer is not bound to the frozen pair manifest")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_random_seed(requested_seeds[0])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_id = config["model"]["model_id"]
    experiment_id = f"fatigue_video_can_{model_id}_trainval_{timestamp}"
    experiment_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else pair_root / "fusion" / "fatigue_video_can" / model_id / "runs" / experiment_id
    )
    try:
        experiment_root.relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("training outputs and weights must remain outside the Git repository")
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {experiment_root}")
    experiment_root.mkdir(parents=True, exist_ok=True)
    resolved_config = copy.deepcopy(config)
    resolved_config["execution"]["resolved_device"] = str(device)
    resolved_config["execution"]["requested_seeds"] = requested_seeds
    _write_json(experiment_root / "resolved_config.json", resolved_config)
    _atomic_text(experiment_root / "environment.txt", environment_text(capture_environment()))
    config_sha256 = file_sha256(config_path)
    manifest_path = experiment_root / "experiment_manifest.json"
    manifest = {
        "experiment_id": experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "formal_result": False,
        "task": "fatigue",
        "modalities": list(MODALITIES),
        "model_id": model_id,
        "cohort": config["data"]["cohort"],
        "requested_seeds": requested_seeds,
        "completed_seeds": [],
        "test_manifest_accessed": False,
        "git": git_state,
        "config_sha256": config_sha256,
        "input_hashes": input_hashes,
    }
    _write_json(manifest_path, manifest)
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
                standardizers=standardizers,
                input_hashes=input_hashes,
                git_state=git_state,
                device=device,
            )
            if result["validation_windows"] != 328 or result["validation_parents"] != 41:
                raise RuntimeError("seed gate requires 328 validation windows and 41 parents")
            if result["checkpoint_reload_max_probability_difference"] > 1e-12:
                raise RuntimeError("seed gate checkpoint reload is not reproducible")
            if result["test_manifest_accessed"] is not False:
                raise RuntimeError("seed gate detected forbidden test access")
            seed_results.append(result)
            manifest["completed_seeds"] = [row["seed"] for row in seed_results]
            _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "FAILED",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "test_manifest_accessed": False,
            }
        )
        _write_json(manifest_path, manifest)
        raise

    summary = {
        "status": "PASS",
        "formal_result": False,
        "test_manifest_accessed": False,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "cohort": config["data"]["cohort"],
        "seed_results": seed_results,
        "statistics": _summary_statistics(seed_results),
        "aggregation_note": "seeds are separate models; no best-seed selection or voting",
    }
    _write_json(experiment_root / "seed_summary.json", summary)
    _write_seed_summary_csv(experiment_root / "seed_summary.csv", seed_results)
    _atomic_text(
        experiment_root / "README.md",
        "# Fatigue video + CAN train/val development experiment\n\n"
        f"Model: `{model_id}`. This experiment uses the frozen paired cohort, "
        "evaluates validation only, and did not access test data. It is not a final test result.\n",
    )
    manifest.update(
        {
            "status": "DEVELOPMENT_TRAIN_VAL_COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_seeds": [row["seed"] for row in seed_results],
            "summary_file": "seed_summary.json",
            "files": _collect_file_records(experiment_root, exclude={manifest_path}),
        }
    )
    _write_json(manifest_path, manifest)
    summary["output_root"] = str(experiment_root)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        summary = train_experiment(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "formal_result": False,
                    "test_manifest_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
