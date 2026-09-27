"""Run the DCPT Video+Audio model-side G2 harness.

This module accepts a batch produced by an external public Dataset/collate.  It
does not read DCPT files, construct a Dataset, or provide a synthetic fallback
for a missing real batch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from driver_state.models.mult import DualModalMulT
from driver_state.models.simple_fusion import SimpleFusion


MODEL_NAMES = ("simple_fusion", "dual_modal_mult")
MODALITIES = ("video", "audio")
INPUT_SHAPES = {"video": (10, 512), "audio": (5, 2048)}
MODEL_BATCH_SOURCES = {
    "synthetic_test",
    "owner_provided_public_dataset_collate",
}


def _reject_placeholders(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
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
        "config_version": "1.0.0",
        "protocol_version": "0.2",
        "schema_version": "0.2.0",
        "status": "model_side_g2_preflight_awaiting_real_public_batch",
        "task": "distraction",
        "dataset": "DCPT",
        "modalities": list(MODALITIES),
        "label_scheme": "dcpt_video_6c_v1",
        "classes": ["01", "03", "04", "05", "07", "08"],
        "label_mapping": {"01": 0, "03": 1, "04": 2, "05": 3, "07": 4, "08": 5},
        "num_classes": 6,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")

    if config.get("input_dims") != {"video": 512, "audio": 2048}:
        raise ValueError("input_dims must be video=512 and audio=2048")

    data = config.get("data")
    if not isinstance(data, dict) or set(data) != {
        "real_batch_source",
        "real_batch_status",
        "split_status",
        "p04_status",
        "test_access",
    }:
        raise ValueError("data must contain only model-side upstream status fields")
    if data != {
        "real_batch_source": "owner_provided_public_dataset_collate",
        "real_batch_status": "required_before_real_g2",
        "split_status": "owner_freeze_required",
        "p04_status": "signed_evidence_required",
        "test_access": "forbidden",
    }:
        raise ValueError("distraction G2 data status fields are not approved")

    inputs = config.get("inputs")
    expected_inputs = {
        "video": {
            "shape": [10, 512],
            "dtype": "float32",
            "fields": ["x", "valid_mask", "time_s"],
        },
        "audio": {
            "shape": [5, 2048],
            "dtype": "float32",
            "fields": ["x", "valid_mask", "time_s"],
        },
    }
    if inputs != expected_inputs:
        raise ValueError("inputs must declare the DCPT model batch contract")

    models = config.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError("config must cover SimpleFusion and DualModalMulT")
    simple = models["simple_fusion"]
    if simple != {
        "class": "SimpleFusion",
        "projection_dim": 5,
        "classifier_hidden_dim": 4,
        "dropout": 0.2,
        "parameter_status": "model_side_g2_smoke_only_not_selected_hyperparameters",
    }:
        raise ValueError("SimpleFusion parameters must remain smoke-only example values")
    mult = models["dual_modal_mult"]
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
        "position_encoding": "sequence_sinusoidal",
        "real_time_encoding": False,
        "parameter_status": "v0.2_first_version_default",
    }:
        raise ValueError("MulT parameters must match protocol v0.2")

    preflight = config.get("preflight")
    if preflight != {
        "seed": 11,
        "loss": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "optimizer_steps": 1,
        "evaluate_metrics": False,
        "access_test_manifest": False,
        "formal_result": False,
    }:
        raise ValueError("unexpected model-side G2 preflight settings")


def _model_from_config(model_name: str, config: dict[str, Any]) -> nn.Module:
    _verify_config(config)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model_name {model_name!r}")
    parameters = config["models"][model_name]
    input_dims = {"video": 512, "audio": 2048}
    if model_name == "simple_fusion":
        return SimpleFusion(
            modalities=MODALITIES,
            input_dims=input_dims,
            num_classes=6,
            projection_dim=int(parameters["projection_dim"]),
            classifier_hidden_dim=int(parameters["classifier_hidden_dim"]),
            dropout=float(parameters["dropout"]),
        )
    return DualModalMulT(
        modalities=MODALITIES,
        input_dims=input_dims,
        num_classes=6,
        d_model=int(parameters["d_model"]),
        num_heads=int(parameters["num_heads"]),
        cross_layers=int(parameters["cross_layers"]),
        memory_layers=int(parameters["memory_layers"]),
        dropout=float(parameters["dropout"]),
        enable_a_from_b=bool(parameters["enable_a_from_b"]),
        enable_b_from_a=bool(parameters["enable_b_from_a"]),
        causal_attention=bool(parameters["causal_attention"]),
    )


def _validate_batch(batch: Mapping[str, Any]) -> tuple[dict[str, dict[str, Tensor]], Tensor]:
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be a mapping")
    inputs = batch.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(MODALITIES):
        raise ValueError("batch.inputs must contain exactly video and audio")
    labels = batch.get("labels")
    if not isinstance(labels, Tensor) or labels.dtype != torch.int64 or labels.ndim != 1:
        raise ValueError("labels must be an int64 Tensor with shape [B]")
    batch_size = int(labels.shape[0])
    if batch_size == 0:
        raise ValueError("batch size must be greater than zero")
    if labels.min().item() < 0 or labels.max().item() > 5:
        raise ValueError("labels must contain only class ids 0..5")

    selected: dict[str, dict[str, Tensor]] = {}
    for modality in MODALITIES:
        stream = inputs[modality]
        if not isinstance(stream, Mapping):
            raise ValueError(f"{modality} must be a mapping")
        required = {"x", "valid_mask", "time_s"}
        if not required.issubset(stream):
            raise ValueError(f"{modality} must contain x, valid_mask, and time_s")
        x = stream["x"]
        valid_mask = stream["valid_mask"]
        time_s = stream["time_s"]
        if not isinstance(x, Tensor) or x.dtype != torch.float32:
            raise ValueError(f"{modality}.x must be a float32 Tensor")
        if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
            raise ValueError(f"{modality}.valid_mask must be a bool Tensor")
        if not isinstance(time_s, Tensor) or not torch.is_floating_point(time_s):
            raise ValueError(f"{modality}.time_s must be a floating Tensor")
        expected_steps, expected_dim = INPUT_SHAPES[modality]
        if x.shape != (batch_size, expected_steps, expected_dim):
            raise ValueError(
                f"{modality}.x must have shape [B,{expected_steps},{expected_dim}]"
            )
        if valid_mask.shape != (batch_size, expected_steps):
            raise ValueError(f"{modality}.valid_mask must have shape [B,{expected_steps}]")
        if time_s.shape != (batch_size, expected_steps):
            raise ValueError(f"{modality}.time_s must have shape [B,{expected_steps}]")
        if not torch.isfinite(x).all().item():
            raise ValueError(f"{modality}.x must contain only finite values")
        if not torch.isfinite(time_s).all().item():
            raise ValueError(f"{modality}.time_s must contain only finite values")
        if not valid_mask.any(dim=1).all().item():
            raise ValueError(f"every {modality} sample must have a valid token")
        selected[modality] = {
            "x": x,
            "valid_mask": valid_mask,
            "time_s": time_s,
        }
    return selected, labels


def _tail_padding_difference(
    model: nn.Module,
    inputs: Mapping[str, Mapping[str, Tensor]],
) -> float:
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
                        dtype=torch.float32,
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
    if not torch.allclose(reference, changed, rtol=0.0, atol=1e-6):
        raise RuntimeError(f"tail padding changed logits; maximum difference={difference}")
    return difference


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_device(device: torch.device | str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def run_model_g2(
    *,
    model_name: str,
    config: dict[str, Any],
    batch: Mapping[str, Any] | None,
    batch_source: str,
    device: torch.device | str,
    output_root: Path,
) -> dict[str, Any]:
    _verify_config(config)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model_name {model_name!r}")
    if batch_source not in MODEL_BATCH_SOURCES:
        raise ValueError(f"batch_source must be one of {sorted(MODEL_BATCH_SOURCES)}")
    if batch is None:
        raise ValueError("REAL_DCPT_BATCH_REQUIRED")
    if batch_source == "owner_provided_public_dataset_collate" and not batch:
        raise ValueError("REAL_DCPT_BATCH_REQUIRED")

    inputs, labels = _validate_batch(batch)
    resolved_device = _resolve_device(device)
    seed = int(config["preflight"]["seed"])
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _model_from_config(model_name, config).to(resolved_device)
    model_inputs = {
        modality: {name: value.to(resolved_device) for name, value in stream.items()}
        for modality, stream in inputs.items()
    }
    model_labels = labels.to(resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    model.eval()
    padding_difference = _tail_padding_difference(model, model_inputs)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["preflight"]["learning_rate"]),
        weight_decay=float(config["preflight"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(model_inputs)["logits"]
    expected_logits_shape = (int(labels.shape[0]), 6)
    if logits.shape != expected_logits_shape:
        raise RuntimeError(f"{model_name} logits must have shape {expected_logits_shape}")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError(f"{model_name} logits must be finite")
    loss = nn.CrossEntropyLoss()(logits, model_labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"{model_name} one-step loss is NaN or Inf")
    loss.backward()
    gradients = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"{model_name} has a participating parameter without a gradient")
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"{model_name} has a non-finite gradient")
        gradients.append(parameter.grad.detach().float().pow(2).sum())
    if not gradients:
        raise RuntimeError(f"{model_name} has no participating gradients")
    gradient_norm = math.sqrt(float(torch.stack(gradients).sum().cpu()))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise RuntimeError(f"{model_name} gradient norm must be finite and positive")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        trained_logits = model(model_inputs)["logits"]
    if not torch.isfinite(trained_logits).all().item():
        raise RuntimeError(f"{model_name} post-step logits must be finite")

    model_root = Path(output_root) / model_name
    checkpoint_path = model_root / "SMOKE_ONLY_DO_NOT_USE.pt"
    model_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "purpose": "one-step model-side G2 save/reload smoke; not a trained model",
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config_version": config["config_version"],
            "seed": seed,
        },
        checkpoint_path,
    )
    reloaded = _model_from_config(model_name, config).to(resolved_device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(model_inputs)["logits"]
    reload_difference = float((trained_logits - reloaded_logits).abs().max().cpu())
    if not torch.allclose(trained_logits, reloaded_logits, rtol=0.0, atol=1e-7):
        raise RuntimeError(f"{model_name} checkpoint reload changed logits")

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else None
    )
    report = {
        "status": "PASS",
        "task": "distraction",
        "modalities": list(MODALITIES),
        "num_classes": 6,
        "model_name": model_name,
        "model_class": config["models"][model_name]["class"],
        "batch_source": batch_source,
        "real_batch": batch_source == "owner_provided_public_dataset_collate",
        "formal_result": False,
        "metrics_evaluated": False,
        "test_manifest_accessed": False,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shapes": {
            modality: list(stream["x"].shape) for modality, stream in model_inputs.items()
        },
        "logits_shape": list(trained_logits.shape),
        "one_step_loss_not_a_metric": float(loss.detach().cpu()),
        "gradient_l2_norm": gradient_norm,
        "tail_padding_max_logit_difference": padding_difference,
        "checkpoint_reload_max_logit_difference": reload_difference,
        "checkpoint": str(checkpoint_path.relative_to(Path(output_root))),
        "peak_cuda_memory_bytes": peak_bytes,
    }
    _write_json(model_root / "g2_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[3]
            / "configs"
            / "fusion"
            / "distraction_video_audio"
            / "g2_preflight_v1.json"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        _verify_config(config)
        raise ValueError("REAL_DCPT_BATCH_REQUIRED")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
