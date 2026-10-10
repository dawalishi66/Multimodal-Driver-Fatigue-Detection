"""Run the non-scoring CAN baseline preflight on frozen train/val manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from driver_state.data import CanWindowDataset, MaskedStandardizer, make_can_collate
from driver_state.data.can import file_sha256
from driver_state.engine import (
    capture_environment,
    capture_git_state,
    environment_text,
    set_random_seed,
)
from driver_state.evaluation import majority_class_from_train
from driver_state.baselines.fatigue_can import CanGruBaseline
from driver_state.preprocessing.fatigue_can.pipeline import CAN_FEATURE_COLUMNS


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _reject_placeholders(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_placeholders(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_placeholders(nested, path=f"{path}[{index}]")
    elif isinstance(value, str) and "TO_BE_FILLED" in value:
        raise ValueError(f"unresolved placeholder at {path}")


def _verify_config(config: dict[str, Any]) -> None:
    _reject_placeholders(config)
    expected = {
        "protocol_version": "0.2",
        "schema_version": "0.2.0",
        "status": "preflight_only_not_formal_result",
        "task": "fatigue",
        "dataset": "UL-DD",
        "modality": "can",
        "label_scheme": "uldd_kss_4_7_v1",
        "num_classes": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if config["classes"] != ["low", "medium", "high"]:
        raise ValueError("CAN fatigue classes must be low/medium/high")
    if config["input"]["shape"] != [300, 9]:
        raise ValueError("CAN input shape must be [300,9]")
    if config["input"]["feature_columns"] != list(CAN_FEATURE_COLUMNS):
        raise ValueError("CAN feature order differs from preprocessing v1")
    if config["input"]["normalization"]["fit_split"] != "train":
        raise ValueError("normalization must be fitted on train")
    if config["model"]["gru_layers"] != 1:
        raise ValueError("the v1 baseline must use one GRU layer")
    if config["training"]["class_imbalance_strategy"] != "none_in_v1_default":
        raise ValueError("v1 must not silently enable class weighting or resampling")
    if config["training"]["seeds"] != [11, 22, 33]:
        raise ValueError("formal training seeds must remain 11/22/33")
    if config["training"]["selection_metric"] != "val_parent_macro_f1":
        raise ValueError("fatigue checkpoint selection must use val parent Macro-F1")
    if config["preflight"]["access_test_manifest"]:
        raise ValueError("preflight is forbidden from accessing test manifests")
    if config["preflight"]["evaluate_metrics"]:
        raise ValueError("preflight must not report performance metrics")


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


def _verify_manifest_hashes(processed_root: Path, config: dict[str, Any]) -> dict[str, str]:
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


def _model_from_config(config: dict[str, Any]) -> CanGruBaseline:
    model = config["model"]
    return CanGruBaseline(
        input_dim=config["input"]["shape"][1],
        projection_dim=model["projection_dim"],
        hidden_dim=model["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=model["dropout"],
    )


def _inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        modality: {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in values.items()
        }
        for modality, values in inputs.items()
    }


def _assert_tail_padding_invariance(
    model: CanGruBaseline,
    inputs: dict[str, Any],
    *,
    atol: float = 1e-6,
) -> float:
    model.eval()
    with torch.no_grad():
        reference = model(inputs)["logits"]
        can = inputs["can"]
        batch_size, _, feature_dim = can["x"].shape
        tail = 7
        padded_can = dict(can)
        padded_can["x"] = torch.cat(
            (
                can["x"],
                torch.randn(
                    batch_size, tail, feature_dim,
                    dtype=can["x"].dtype, device=can["x"].device,
                ) * 1000,
            ),
            dim=1,
        )
        padded_can["valid_mask"] = torch.cat(
            (
                can["valid_mask"],
                torch.zeros(
                    batch_size, tail, dtype=torch.bool,
                    device=can["valid_mask"].device,
                ),
            ),
            dim=1,
        )
        padded_can["time_s"] = torch.cat(
            (
                can["time_s"],
                torch.zeros(
                    batch_size, tail, dtype=can["time_s"].dtype,
                    device=can["time_s"].device,
                ),
            ),
            dim=1,
        )
        padded = model({"can": padded_can})["logits"]
    maximum_difference = float((reference - padded).abs().max().cpu())
    if not torch.allclose(reference, padded, rtol=0, atol=atol):
        raise RuntimeError(
            f"tail padding changed logits; max difference={maximum_difference}"
        )
    return maximum_difference


def _class_counts(dataset: CanWindowDataset) -> dict[str, int]:
    counts = Counter(record.label_class for record in dataset.records)
    return {name: counts.get(name, 0) for name in ("low", "medium", "high")}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    processed_root = Path(args.processed_root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_config(config)
    if not processed_root.is_dir():
        raise FileNotFoundError(f"processed_root does not exist: {processed_root}")
    checked_hashes = (
        _verify_manifest_hashes(processed_root, config)
        if config["preflight"]["verify_split_manifest_hashes"] else {}
    )
    data = config["data"]
    train_dataset = CanWindowDataset(
        _resolve_under(processed_root, data["train_windows"]),
        feature_root=processed_root,
        split="train",
        parent_manifest=_resolve_under(processed_root, data["train_parents"]),
        verify_feature_hash=config["preflight"]["verify_each_feature_hash"],
    )
    val_dataset = CanWindowDataset(
        _resolve_under(processed_root, data["val_windows"]),
        feature_root=processed_root,
        split="val",
        parent_manifest=_resolve_under(processed_root, data["val_parents"]),
        verify_feature_hash=config["preflight"]["verify_each_feature_hash"],
    )
    normalizer = MaskedStandardizer.fit_can_dataset(
        train_dataset, feature_names=CAN_FEATURE_COLUMNS
    )
    if normalizer.source_manifest_sha256 != checked_hashes.get(
        data["train_windows"], normalizer.source_manifest_sha256
    ):
        raise ValueError("normalizer is not bound to the frozen train manifest")

    seed = int(config["preflight"]["seed"])
    set_random_seed(seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    batch_size = int(config["preflight"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_can_collate(normalizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_can_collate(normalizer),
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    train_inputs = _inputs_to_device(train_batch["inputs"], device)
    val_inputs = _inputs_to_device(val_batch["inputs"], device)
    labels = train_batch["labels"].to(device)

    model = _model_from_config(config).to(device)
    padding_max_difference = _assert_tail_padding_invariance(model, train_inputs)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    output = model(train_inputs)
    loss = nn.CrossEntropyLoss()(output["logits"], labels)
    if not torch.isfinite(loss):
        raise RuntimeError("smoke loss is NaN or Inf")
    loss.backward()
    gradient_norm = math.sqrt(sum(
        float(parameter.grad.detach().float().pow(2).sum().cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    ))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("smoke gradients are missing or non-finite")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        post_step_logits = model(train_inputs)["logits"]
        val_logits = model(val_inputs)["logits"]
    if post_step_logits.shape != (len(train_batch["labels"]), 3):
        raise RuntimeError("train smoke logits have the wrong shape")
    if val_logits.shape != (len(val_batch["labels"]), 3):
        raise RuntimeError("validation smoke logits have the wrong shape")
    if not torch.isfinite(post_step_logits).all() or not torch.isfinite(val_logits).all():
        raise RuntimeError("smoke logits contain NaN or Inf")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root else (
            processed_root
            / "baselines"
            / "fatigue_can"
            / "gru_v1"
            / "preflight"
            / f"can_preflight_gru_v1_{timestamp}"
        )
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    normalizer_path = output_root / "normalizer.json"
    normalizer.save_json(normalizer_path)
    checkpoint_path = output_root / "SMOKE_ONLY_DO_NOT_USE.pt"
    torch.save(
        {
            "purpose": "one-step save/reload smoke test; not a trained model",
            "model_state_dict": model.state_dict(),
            "config_version": config["config_version"],
            "seed": seed,
        },
        checkpoint_path,
    )
    reloaded = _model_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(train_inputs)["logits"]
    reload_max_difference = float(
        (post_step_logits - reloaded_logits).abs().max().cpu()
    )
    if not torch.allclose(post_step_logits, reloaded_logits, rtol=0, atol=1e-7):
        raise RuntimeError("checkpoint reload changed model outputs")

    environment = capture_environment()
    git_state = capture_git_state(repository)
    train_labels = [record.label_id for record in train_dataset.records]
    report = {
        "status": "PASS",
        "purpose": "CAN baseline preflight only; no performance result",
        "scope": ["G1_interface", "G2_real_train_val_smoke_partial"],
        "not_claimed": ["G0_complete", "formal_training", "validation_score", "test_score"],
        "test_manifest_accessed": False,
        "device": str(device),
        "cohort": data["cohort"],
        "cohort_limitation": data["cohort_status"],
        "train": {
            "windows": len(train_dataset),
            "parents": len({record.parent_id for record in train_dataset.records}),
            "subjects": sorted({record.subject_id for record in train_dataset.records}),
            "sessions": len({record.session_id for record in train_dataset.records}),
            "class_counts": _class_counts(train_dataset),
        },
        "val": {
            "windows": len(val_dataset),
            "parents": len({record.parent_id for record in val_dataset.records}),
            "subjects": sorted({record.subject_id for record in val_dataset.records}),
            "sessions": len({record.session_id for record in val_dataset.records}),
            "class_counts": _class_counts(val_dataset),
        },
        "normalization": {
            "version": normalizer.version,
            "fitted_split": normalizer.fitted_split,
            "valid_token_count": normalizer.valid_token_count,
            "zero_variance_indices": list(normalizer.zero_variance_indices),
            "source_manifest_sha256": normalizer.source_manifest_sha256,
        },
        "model": {
            "name": config["model"]["name"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "logit_shape": list(post_step_logits.shape),
            "embedding_shape": list(output["embedding"].shape),
        },
        "smoke": {
            "optimizer_steps": 1,
            "loss_finite": True,
            "gradient_norm": gradient_norm,
            "tail_padding_max_logit_difference": padding_max_difference,
            "checkpoint_reload_max_logit_difference": reload_max_difference,
            "majority_class_from_train_only": majority_class_from_train(
                train_labels, num_classes=3
            ),
            "metrics_evaluated": False,
        },
        "frozen_manifest_hashes_checked": checked_hashes,
        "config_sha256": file_sha256(config_path),
        "git": git_state,
        "environment_file": "environment.txt",
        "normalizer_file": "normalizer.json",
        "checkpoint_file": checkpoint_path.name,
    }
    run_manifest = {
        "run_id": f"can_preflight_gru_v1_{timestamp}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SMOKE_ONLY_NOT_FORMAL",
        "task": "fatigue",
        "dataset": "UL-DD",
        "modality": "can",
        "protocol_version": config["protocol_version"],
        "schema_version": config["schema_version"],
        "config_version": config["config_version"],
        "feature_version": data["feature_version"],
        "split_version": data["split_version"],
        "cohort": data["cohort"],
        "seed": seed,
        "test_manifest_accessed": False,
        "git": git_state,
        "files": {
            "config": {"source_sha256": file_sha256(config_path)},
            "manifests": checked_hashes,
            "normalizer": {"sha256": file_sha256(normalizer_path)},
            "checkpoint": {"sha256": file_sha256(checkpoint_path)},
        },
    }
    _write_json(output_root / "resolved_config.json", config)
    _atomic_text(output_root / "environment.txt", environment_text(environment))
    _write_json(output_root / "preflight_report.json", report)
    _write_json(output_root / "run_manifest.json", run_manifest)
    report["output_root"] = str(output_root)
    return report


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
            / "gru_v1.json"
        ),
    )
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        report = prepare(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "test_manifest_accessed": False,
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
