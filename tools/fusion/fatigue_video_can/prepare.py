"""Run a non-scoring real-data G2 smoke test for fatigue video/CAN models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from driver_state.data import (
    FatigueVideoCanDataset,
    MaskedStandardizer,
    make_fatigue_video_can_collate,
    make_fusion_model_inputs,
)
from driver_state.data.can import file_sha256
from driver_state.engine import (
    capture_environment,
    capture_git_state,
    environment_text,
    set_random_seed,
)
from driver_state.models.mult import DualModalMulT
from driver_state.models.simple_fusion import SimpleFusion
from driver_state.preprocessing.fatigue_can.pipeline import CAN_FEATURE_COLUMNS


MODEL_NAMES = ("simple_fusion", "dual_modal_mult")
MODALITIES = ("video", "can")


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


def _verify_sha256(value: Any, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return normalized


def _verify_config(config: dict[str, Any]) -> None:
    _reject_placeholders(config)
    expected = {
        "config_version": "1.0.0",
        "protocol_version": "0.2",
        "schema_version": "0.2.0",
        "status": "g2_preflight_only_not_formal_result",
        "task": "fatigue",
        "dataset": "UL-DD",
        "label_scheme": "uldd_kss_4_7_v1",
        "num_classes": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if config.get("modalities") != list(MODALITIES):
        raise ValueError("modalities must be ['video','can'] in that order")
    if config.get("classes") != ["low", "medium", "high"]:
        raise ValueError("fatigue classes must be low/medium/high")

    data = config["data"]
    if set(data) != {
        "cohort",
        "pair_manifest",
        "pair_manifest_sha256",
        "validation_report",
        "validation_report_sha256",
        "expected_counts",
        "test_access",
    }:
        raise ValueError("data must contain only the frozen train/val G2 fields")
    _verify_sha256(data["pair_manifest_sha256"], field="pair_manifest_sha256")
    _verify_sha256(data["validation_report_sha256"], field="validation_report_sha256")
    if data["test_access"] != "forbidden_in_preflight":
        raise ValueError("G2 preflight must forbid test access")
    if data["expected_counts"] != {
        "train_windows": 1272,
        "train_parents": 159,
        "val_windows": 328,
        "val_parents": 41,
    }:
        raise ValueError("unexpected frozen paired-cohort counts")

    inputs = config["inputs"]
    if inputs["video"]["shape"] != [6, 96] or inputs["can"]["shape"] != [300, 9]:
        raise ValueError("G2 inputs must be video [6,96] and CAN [300,9]")
    if inputs["video"]["feature_names"] != "video_feature_000_to_095":
        raise ValueError("unexpected video feature-name scheme")
    if inputs["can"]["feature_names"] != list(CAN_FEATURE_COLUMNS):
        raise ValueError("CAN feature order differs from preprocessing v1")
    for modality in MODALITIES:
        if inputs[modality]["normalization"] != "train_only_masked_zscore_v1":
            raise ValueError(f"{modality} normalization must be train-only masked z-score")

    if set(config["models"]) != set(MODEL_NAMES):
        raise ValueError("G2 must cover SimpleFusion and DualModalMulT")
    simple = config["models"]["simple_fusion"]
    if simple != {
        "class": "SimpleFusion",
        "projection_dim": 64,
        "classifier_hidden_dim": 64,
        "dropout": 0.2,
    }:
        raise ValueError("unexpected SimpleFusion G2 configuration")
    mult = config["models"]["dual_modal_mult"]
    if mult != {
        "class": "DualModalMulT",
        "d_model": 30,
        "num_heads": 5,
        "cross_layers": 5,
        "memory_layers": 5,
        "dropout": 0.0,
        "enable_a_from_b": True,
        "enable_b_from_a": True,
        "causal_attention": False,
    }:
        raise ValueError("MulT G2 parameters must match protocol v0.2 defaults")

    training = config["training"]
    if training != {
        "loss": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
    }:
        raise ValueError("unexpected one-step optimizer configuration")
    preflight = config["preflight"]
    if preflight["seed"] != 11 or preflight["optimizer_steps"] != 1:
        raise ValueError("G2 preflight must run one step with seed 11")
    if not 5 <= int(preflight["windows_per_split"]) <= 25:
        raise ValueError("G2 must exercise 10-50 real windows across train and val")
    if preflight["access_test_manifest"] or preflight["evaluate_metrics"]:
        raise ValueError("G2 preflight cannot access test or evaluate performance")


def _resolve_under(root: Path, relative: str, *, field: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{field} must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes its declared root") from exc
    return resolved


def _verify_frozen_inputs(pair_root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    data = config["data"]
    pair_manifest = _resolve_under(pair_root, data["pair_manifest"], field="pair_manifest")
    validation_report = _resolve_under(
        pair_root, data["validation_report"], field="validation_report"
    )
    checked = {
        "pair_manifest": file_sha256(pair_manifest),
        "validation_report": file_sha256(validation_report),
    }
    if checked["pair_manifest"] != data["pair_manifest_sha256"]:
        raise ValueError("frozen pair manifest hash mismatch")
    if checked["validation_report"] != data["validation_report_sha256"]:
        raise ValueError("frozen dataset validation report hash mismatch")
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("test_manifest_accessed") is not False:
        raise ValueError("paired Dataset validation is not an approved train/val PASS")
    if report.get("pair_manifest_sha256") != checked["pair_manifest"]:
        raise ValueError("validation report refers to a different pair manifest")
    if report.get("totals") != {"windows": 1600, "parents": 200}:
        raise ValueError("validation report has unexpected paired-cohort totals")
    return pair_manifest, checked


def _assert_expected_counts(
    train_dataset: FatigueVideoCanDataset,
    val_dataset: FatigueVideoCanDataset,
    expected: Mapping[str, int],
) -> None:
    actual = {
        "train_windows": len(train_dataset),
        "train_parents": train_dataset.parent_count,
        "val_windows": len(val_dataset),
        "val_parents": val_dataset.parent_count,
    }
    if actual != dict(expected):
        raise ValueError(f"paired-cohort counts changed: expected {dict(expected)}, got {actual}")


def _video_feature_names(dimension: int) -> tuple[str, ...]:
    return tuple(f"video_feature_{index:03d}" for index in range(dimension))


def _select_smoke_indices(dataset: FatigueVideoCanDataset, count: int) -> list[int]:
    if count > len(dataset):
        raise ValueError("smoke count exceeds Dataset size")
    by_label: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for index, record in enumerate(dataset.records):
        by_label[record.label_id].append(index)
    if any(not indices for indices in by_label.values()):
        raise ValueError("smoke selection requires all three fatigue classes")
    quotas = [count // 3 + (1 if label < count % 3 else 0) for label in range(3)]
    selected: list[int] = []
    for label, quota in enumerate(quotas):
        candidates = by_label[label]
        positions = np.linspace(0, len(candidates) - 1, num=quota, dtype=int)
        selected.extend(candidates[int(position)] for position in positions)
    if len(set(selected)) != count:
        raise RuntimeError("deterministic smoke selection produced duplicate rows")
    return sorted(selected)


def _model_from_config(model_name: str, config: dict[str, Any]) -> nn.Module:
    parameters = config["models"][model_name]
    input_dims = {"video": 96, "can": 9}
    if model_name == "simple_fusion":
        return SimpleFusion(
            modalities=MODALITIES,
            input_dims=input_dims,
            num_classes=3,
            projection_dim=int(parameters["projection_dim"]),
            classifier_hidden_dim=int(parameters["classifier_hidden_dim"]),
            dropout=float(parameters["dropout"]),
        )
    if model_name == "dual_modal_mult":
        return DualModalMulT(
            modalities=MODALITIES,
            input_dims=input_dims,
            num_classes=3,
            d_model=int(parameters["d_model"]),
            num_heads=int(parameters["num_heads"]),
            cross_layers=int(parameters["cross_layers"]),
            memory_layers=int(parameters["memory_layers"]),
            dropout=float(parameters["dropout"]),
            enable_a_from_b=bool(parameters["enable_a_from_b"]),
            enable_b_from_a=bool(parameters["enable_b_from_a"]),
            causal_attention=bool(parameters["causal_attention"]),
        )
    raise ValueError(f"unknown model_name {model_name!r}")


def _inputs_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, dict[str, Tensor]]:
    selected = make_fusion_model_inputs(batch)
    return {
        modality: {name: value.to(device) for name, value in stream.items()}
        for modality, stream in selected.items()
    }


def _tail_padding_difference(model: nn.Module, inputs: Mapping[str, Mapping[str, Tensor]]) -> float:
    model.eval()
    padded: dict[str, dict[str, Tensor]] = {}
    for modality, stream in inputs.items():
        batch_size, _, dimension = stream["x"].shape
        tail = 2
        padded[modality] = {
            "x": torch.cat(
                (
                    stream["x"],
                    torch.full(
                        (batch_size, tail, dimension),
                        12345.0,
                        dtype=stream["x"].dtype,
                        device=stream["x"].device,
                    ),
                ),
                dim=1,
            ),
            "valid_mask": torch.cat(
                (
                    stream["valid_mask"],
                    torch.zeros(
                        (batch_size, tail),
                        dtype=torch.bool,
                        device=stream["valid_mask"].device,
                    ),
                ),
                dim=1,
            ),
            "time_s": torch.cat(
                (
                    stream["time_s"],
                    torch.zeros(
                        (batch_size, tail),
                        dtype=stream["time_s"].dtype,
                        device=stream["time_s"].device,
                    ),
                ),
                dim=1,
            ),
        }
    with torch.no_grad():
        reference = model(inputs)["logits"]
        changed = model(padded)["logits"]
    difference = float((reference - changed).abs().max().cpu())
    if not torch.allclose(reference, changed, rtol=0, atol=1e-6):
        raise RuntimeError(f"tail padding changed logits; maximum difference={difference}")
    return difference


def _run_model_smoke(
    *,
    model_name: str,
    config: dict[str, Any],
    train_batch: Mapping[str, Any],
    val_batch: Mapping[str, Any],
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    seed = int(config["preflight"]["seed"])
    set_random_seed(seed)
    model = _model_from_config(model_name, config).to(device)
    train_inputs = _inputs_to_device(train_batch, device)
    val_inputs = _inputs_to_device(val_batch, device)
    labels = train_batch["labels"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    padding_difference = _tail_padding_difference(model, train_inputs)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_inputs)["logits"]
    if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all():
        raise RuntimeError(f"{model_name} returned invalid train logits")
    loss = nn.CrossEntropyLoss()(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{model_name} smoke loss is NaN or Inf")
    loss.backward()
    gradient_norm = math.sqrt(
        sum(
            float(parameter.grad.detach().float().pow(2).sum().cpu())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError(f"{model_name} gradients are missing or non-finite")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        trained_logits = model(train_inputs)["logits"]
        val_logits = model(val_inputs)["logits"]
    if val_logits.shape != (len(val_batch["labels"]), 3) or not torch.isfinite(val_logits).all():
        raise RuntimeError(f"{model_name} returned invalid validation logits")

    model_root = output_root / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / "SMOKE_ONLY_DO_NOT_USE.pt"
    torch.save(
        {
            "purpose": "one-step G2 save/reload smoke; not a trained model",
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config_version": config["config_version"],
            "seed": seed,
        },
        checkpoint_path,
    )
    reloaded = _model_from_config(model_name, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(train_inputs)["logits"]
    reload_difference = float((trained_logits - reloaded_logits).abs().max().cpu())
    if not torch.allclose(trained_logits, reloaded_logits, rtol=0, atol=1e-7):
        raise RuntimeError(f"{model_name} checkpoint reload changed logits")

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "status": "PASS",
        "model_class": config["models"][model_name]["class"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_logits_shape": list(trained_logits.shape),
        "val_logits_shape": list(val_logits.shape),
        "one_step_loss_not_a_metric": float(loss.detach().cpu()),
        "gradient_l2_norm": gradient_norm,
        "tail_padding_max_logit_difference": padding_difference,
        "checkpoint_reload_max_logit_difference": reload_difference,
        "checkpoint": f"{model_name}/SMOKE_ONLY_DO_NOT_USE.pt",
        "peak_cuda_memory_bytes": peak_bytes,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    pair_root = Path(args.pair_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    config_path = Path(args.config).resolve()
    if not pair_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("pair_root and dataset_root must both exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_config(config)
    pair_manifest, checked_hashes = _verify_frozen_inputs(pair_root, config)

    git_state = capture_git_state(repository)
    if config["preflight"]["require_clean_git"] and git_state.get("dirty") is not False:
        raise RuntimeError("G2 preflight requires a clean Git worktree")
    train_dataset = FatigueVideoCanDataset(
        pair_manifest,
        dataset_root=dataset_root,
        split="train",
        verify_feature_hashes=bool(config["preflight"]["verify_each_feature_hash"]),
        cache_features=True,
    )
    val_dataset = FatigueVideoCanDataset(
        pair_manifest,
        dataset_root=dataset_root,
        split="val",
        verify_feature_hashes=bool(config["preflight"]["verify_each_feature_hash"]),
        cache_features=True,
    )
    _assert_expected_counts(train_dataset, val_dataset, config["data"]["expected_counts"])

    standardizers = {
        "video": MaskedStandardizer.fit_dataset_modality(
            train_dataset,
            modality="video",
            feature_names=_video_feature_names(96),
        ),
        "can": MaskedStandardizer.fit_dataset_modality(
            train_dataset,
            modality="can",
            feature_names=CAN_FEATURE_COLUMNS,
        ),
    }
    if any(
        standardizer.source_manifest_sha256 != checked_hashes["pair_manifest"]
        for standardizer in standardizers.values()
    ):
        raise RuntimeError("a normalizer is not bound to the frozen train manifest")

    sample_count = int(config["preflight"]["windows_per_split"])
    train_indices = _select_smoke_indices(train_dataset, sample_count)
    val_indices = _select_smoke_indices(val_dataset, sample_count)
    collate = make_fatigue_video_can_collate(standardizers)
    train_batch = next(
        iter(
            DataLoader(
                Subset(train_dataset, train_indices),
                batch_size=sample_count,
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
            )
        )
    )
    val_batch = next(
        iter(
            DataLoader(
                Subset(val_dataset, val_indices),
                batch_size=sample_count,
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
            )
        )
    )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"fatigue_video_can_g2_v1_{timestamp}"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else pair_root / "preflight" / "fatigue_video_can" / "g2_v1" / run_id
    )
    try:
        output_root.relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("G2 outputs and checkpoints must remain outside the Git repository")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "resolved_config.json", config)
    for modality, standardizer in standardizers.items():
        standardizer.save_json(output_root / "normalizers" / f"{modality}.json")

    model_reports = {
        model_name: _run_model_smoke(
            model_name=model_name,
            config=config,
            train_batch=train_batch,
            val_batch=val_batch,
            device=device,
            output_root=output_root,
        )
        for model_name in MODEL_NAMES
    }
    environment = capture_environment()
    _atomic_text(output_root / "environment.txt", environment_text(environment))
    report = {
        "schema_version": "fatigue_video_can_g2_preflight_v1",
        "status": "PASS",
        "purpose": "real train/val G2 interface smoke only; no performance result",
        "formal_result": False,
        "metrics_evaluated": False,
        "test_manifest_accessed": False,
        "run_id": run_id,
        "git": git_state,
        "config_sha256": file_sha256(config_path),
        "frozen_inputs": checked_hashes,
        "cohort": config["data"]["cohort"],
        "counts": config["data"]["expected_counts"],
        "inputs": {
            "video_shape": [sample_count, 6, 96],
            "can_shape": [sample_count, 300, 9],
            "time_reference": "sample_relative_0_to_30_seconds",
            "mask_semantics": "true_is_valid",
        },
        "normalization": {
            modality: {
                "fitted_split": standardizer.fitted_split,
                "feature_dim": standardizer.feature_dim,
                "valid_token_count": standardizer.valid_token_count,
                "zero_variance_indices": list(standardizer.zero_variance_indices),
                "state": f"normalizers/{modality}.json",
            }
            for modality, standardizer in standardizers.items()
        },
        "smoke_batches": {
            "train": {
                "windows": len(train_batch["sample_id"]),
                "subjects": sorted(set(train_batch["subject_id"])),
                "label_counts": dict(sorted(Counter(train_batch["labels"].tolist()).items())),
            },
            "val": {
                "windows": len(val_batch["sample_id"]),
                "subjects": sorted(set(val_batch["subject_id"])),
                "label_counts": dict(sorted(Counter(val_batch["labels"].tolist()).items())),
            },
        },
        "device": str(device),
        "models": model_reports,
        "checks": {
            "real_feature_read": "PASS",
            "train_only_normalization": "PASS",
            "forward_backward": "PASS",
            "tail_padding_invariance": "PASS",
            "checkpoint_save_reload": "PASS",
        },
        "known_limitations": [
            "No accuracy, F1, threshold, early stopping, or test evaluation was run.",
            "DualModalMulT v1 validates time_s but uses sequence sinusoidal positions, not real-time encoding.",
            "The saved checkpoints are one-step smoke artifacts and must never be used as trained models.",
        ],
    }
    _write_json(output_root / "preflight_report.json", report)
    _atomic_text(
        output_root / "README.md",
        "# Fatigue video + CAN G2 preflight\n\n"
        "This package verifies real train/validation feature loading, train-only "
        "normalization, forward/backward, padding masks, and checkpoint reload for "
        "SimpleFusion and DualModalMulT. It contains no performance result and did "
        "not access test data. Files named `SMOKE_ONLY_DO_NOT_USE.pt` are not trained "
        "model weights.\n",
    )
    return {
        "status": "PASS",
        "run_id": run_id,
        "output_root": str(output_root),
        "metrics_evaluated": False,
        "test_manifest_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / "fusion"
            / "fatigue_video_can"
            / "g2_preflight_v1.json"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "formal_result": False,
                    "metrics_evaluated": False,
                    "test_manifest_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
