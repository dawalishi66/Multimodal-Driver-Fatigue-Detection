from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.fusion.distraction_video_audio.prepare import (
    MODEL_NAMES,
    _model_from_config,
    _resolve_device,
    _verify_config,
    run_model_g2,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPOSITORY
    / "configs"
    / "fusion"
    / "distraction_video_audio"
    / "g2_preflight_v1.json"
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def synthetic_batch() -> dict:
    generator = torch.Generator().manual_seed(123)
    video_mask = torch.tensor(
        [
            [True, False, True, True, False, True, True, False, False, False],
            [True, True, True, True, True, True, True, True, True, False],
            [True, False, True, True, True, False, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    audio_mask = torch.tensor(
        [
            [True, False, True, True, False],
            [True, True, True, True, True],
            [True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    video_x = torch.randn(3, 10, 512, generator=generator)
    audio_x = torch.randn(3, 5, 2048, generator=generator)
    video_x[~video_mask] = 12345.0
    audio_x[~audio_mask] = -9876.0
    return {
        "inputs": {
            "video": {
                "x": video_x,
                "valid_mask": video_mask,
                "time_s": torch.arange(10, dtype=torch.float64).repeat(3, 1),
                "support_s": torch.zeros(3, 10, 2, dtype=torch.float64),
                "observed_fraction": torch.ones(3, 10, dtype=torch.float32),
            },
            "audio": {
                "x": audio_x,
                "valid_mask": audio_mask,
                "time_s": torch.arange(5, dtype=torch.float64).repeat(3, 1),
                "support_s": torch.zeros(3, 5, 2, dtype=torch.float64),
                "observed_fraction": torch.ones(3, 5, dtype=torch.float32),
            },
        },
        "labels": torch.tensor([0, 1, 5], dtype=torch.int64),
        "sample_id": ["synthetic_0", "synthetic_1", "synthetic_2"],
        "subject_id": ["P01", "P02", "P03"],
    }


def run_synthetic(model_name: str, tmp_path: Path) -> dict:
    return run_model_g2(
        model_name=model_name,
        config=load_config(),
        batch=synthetic_batch(),
        batch_source="synthetic_test",
        device=torch.device("cpu"),
        output_root=tmp_path,
    )


def test_config_is_model_side_six_class_preflight_only():
    config = load_config()

    _verify_config(config)

    assert config["status"] == "model_side_g2_preflight_awaiting_real_public_batch"
    assert config["label_scheme"] == "dcpt_video_6c_v1"
    assert config["classes"] == ["01", "03", "04", "05", "07", "08"]
    assert config["num_classes"] == 6
    assert config["inputs"]["video"]["shape"] == [10, 512]
    assert config["inputs"]["video"]["dtype"] == "float32"
    assert config["inputs"]["video"]["fields"] == ["x", "valid_mask", "time_s"]
    assert config["inputs"]["audio"]["shape"] == [5, 2048]
    assert config["inputs"]["audio"]["dtype"] == "float32"
    assert config["inputs"]["audio"]["fields"] == ["x", "valid_mask", "time_s"]
    assert config["data"]["test_access"] == "forbidden"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_synthetic_batch_runs_model_side_g2_harness(tmp_path: Path, model_name: str):
    report = run_synthetic(model_name, tmp_path)

    assert report["status"] == "PASS"
    assert report["real_batch"] is False
    assert report["formal_result"] is False
    assert report["metrics_evaluated"] is False
    assert report["test_manifest_accessed"] is False
    assert "real_g2" not in report
    assert report["input_shapes"] == {
        "video": [3, 10, 512],
        "audio": [3, 5, 2048],
    }
    assert report["logits_shape"] == [3, 6]
    assert report["one_step_loss_not_a_metric"] >= 0.0
    assert report["gradient_l2_norm"] > 0.0
    assert report["tail_padding_max_logit_difference"] <= 1e-6
    assert report["checkpoint_reload_max_logit_difference"] <= 1e-7
    assert report["peak_cuda_memory_bytes"] is None
    assert report["parameters"] > 0

    expected_parameters = {
        "simple_fusion": 12884,
        "dual_modal_mult": 308526,
    }
    assert report["parameters"] == expected_parameters[model_name]

    checkpoint = tmp_path / report["checkpoint"]
    assert checkpoint.name == "SMOKE_ONLY_DO_NOT_USE.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(state) == {
        "purpose",
        "model_name",
        "model_state_dict",
        "config_version",
        "seed",
    }
    assert state["model_name"] == model_name


def test_model_construction_uses_dcpt_input_contract():
    config = load_config()

    simple = _model_from_config("simple_fusion", config)
    mult = _model_from_config("dual_modal_mult", config)

    assert simple.input_dims == {"video": 512, "audio": 2048}
    assert mult.input_dims == {"video": 512, "audio": 2048}
    assert simple.classifier[-1].out_features == 6
    assert mult.num_classes == 6


def _run_rejected_batch(tmp_path: Path, mutate, message: str) -> None:
    batch = synthetic_batch()
    mutate(batch)
    with pytest.raises(ValueError, match=message):
        run_model_g2(
            model_name="simple_fusion",
            config=load_config(),
            batch=batch,
            batch_source="synthetic_test",
            device=torch.device("cpu"),
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch: batch["inputs"]["video"].update(
                x=batch["inputs"]["video"]["x"][:, :9]
            ),
            r"video.x must have shape \[B,10,512\]",
        ),
        (
            lambda batch: batch["inputs"]["video"].update(
                x=batch["inputs"]["video"]["x"][:, :, :511]
            ),
            r"video.x must have shape \[B,10,512\]",
        ),
        (
            lambda batch: batch["inputs"]["audio"].update(
                x=batch["inputs"]["audio"]["x"][:, :4]
            ),
            r"audio.x must have shape \[B,5,2048\]",
        ),
        (
            lambda batch: batch["inputs"]["audio"].update(
                x=batch["inputs"]["audio"]["x"][:, :, :2047]
            ),
            r"audio.x must have shape \[B,5,2048\]",
        ),
        (
            lambda batch: batch.update(labels=torch.tensor([0, 1, 6], dtype=torch.int64)),
            "labels must contain only class ids 0..5",
        ),
        (
            lambda batch: batch["inputs"]["video"]["valid_mask"].fill_(False),
            "every video sample must have a valid token",
        ),
        (
            lambda batch: batch["inputs"]["audio"]["valid_mask"].fill_(False),
            "every audio sample must have a valid token",
        ),
        (
            lambda batch: batch["inputs"]["video"].pop("x"),
            "video must contain x, valid_mask, and time_s",
        ),
        (
            lambda batch: batch["inputs"]["audio"].pop("valid_mask"),
            "audio must contain x, valid_mask, and time_s",
        ),
        (
            lambda batch: batch["inputs"]["video"].pop("time_s"),
            "video must contain x, valid_mask, and time_s",
        ),
    ],
)
def test_runner_rejects_invalid_batch_contract(tmp_path: Path, mutate, message: str):
    _run_rejected_batch(tmp_path, mutate, message)


def test_runner_rejects_missing_real_public_batch(tmp_path: Path):
    with pytest.raises(ValueError, match="REAL_DCPT_BATCH_REQUIRED"):
        run_model_g2(
            model_name="simple_fusion",
            config=load_config(),
            batch=None,
            batch_source="owner_provided_public_dataset_collate",
            device=torch.device("cpu"),
            output_root=tmp_path,
        )


def test_runner_rejects_unknown_batch_source(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_source"):
        run_model_g2(
            model_name="simple_fusion",
            config=load_config(),
            batch=synthetic_batch(),
            batch_source="random_fallback",
            device=torch.device("cpu"),
            output_root=tmp_path,
        )


def test_auto_device_resolves_to_available_device():
    resolved = _resolve_device("auto")

    assert resolved.type == ("cuda" if torch.cuda.is_available() else "cpu")
