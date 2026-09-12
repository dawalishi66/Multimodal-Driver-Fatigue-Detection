from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from driver_state.data import MaskedStandardizer
from tools.fusion.fatigue_video_can.prepare import (
    MODEL_NAMES,
    _model_from_config,
    _run_model_smoke,
    _select_smoke_indices,
    _verify_config,
)
from tools.fusion.fatigue_video_can.train import (
    _build_model as build_training_model,
    run_seed,
    verify_train_val_config,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPOSITORY
    / "configs"
    / "fusion"
    / "fatigue_video_can"
    / "g2_preflight_v1.json"
)
TRAIN_CONFIG_DIR = REPOSITORY / "configs" / "fusion" / "fatigue_video_can"


class MiniDataset:
    def __init__(self, *, split: str = "train") -> None:
        self.split = split
        self.manifest_sha256 = "a" * 64
        self.samples = []
        self.records = []
        for index in range(18):
            label = index % 3
            video_x = torch.tensor(
                [[index, 1.0], [index + 2.0, 3.0]], dtype=torch.float32
            )
            can_x = torch.tensor(
                [[index], [index + 1.0], [9999.0]], dtype=torch.float32
            )
            self.samples.append(
                {
                    "inputs": {
                        "video": {
                            "x": video_x,
                            "valid_mask": torch.tensor([True, index % 2 == 0]),
                        },
                        "can": {
                            "x": can_x,
                            "valid_mask": torch.tensor([True, True, False]),
                        },
                    }
                }
            )
            self.records.append(SimpleNamespace(label_id=label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class TinyPairedDataset:
    def __init__(self, *, split: str, subject: str) -> None:
        self.split = split
        self.manifest_sha256 = "c" * 64
        self.samples = []
        self.records = []
        generator = torch.Generator().manual_seed(42 if split == "train" else 43)
        for index in range(8):
            sample_id = f"SYNTHETIC_{subject}_A_{index}"
            video_x = torch.randn(6, 96, generator=generator) + index / 10
            can_x = torch.randn(300, 9, generator=generator) + index / 10
            video_starts = torch.arange(6, dtype=torch.float64) * 5
            can_starts = torch.arange(300, dtype=torch.float64) / 10
            self.samples.append(
                {
                    "inputs": {
                        "video": {
                            "x": video_x,
                            "time_s": video_starts + 2.5,
                            "valid_mask": torch.ones(6, dtype=torch.bool),
                            "support_s": torch.column_stack(
                                (video_starts, video_starts + 5)
                            ),
                            "observed_fraction": torch.ones(6, dtype=torch.float32),
                        },
                        "can": {
                            "x": can_x,
                            "time_s": can_starts + 0.05,
                            "valid_mask": torch.ones(300, dtype=torch.bool),
                            "support_s": torch.column_stack(
                                (can_starts, can_starts + 0.1)
                            ),
                            "observed_fraction": torch.ones(300, dtype=torch.float32),
                        },
                    },
                    "label": torch.tensor(0, dtype=torch.int64),
                    "sample_id": sample_id,
                    "video_sample_id": f"video_{sample_id}",
                    "can_sample_id": sample_id,
                    "parent_id": f"SYNTHETIC_{subject}_A_PARENT0",
                    "subject_id": subject,
                    "session_id": f"{subject}_A",
                    "window_index": index,
                    "window_start_ms": index * 30_000,
                    "window_end_ms": (index + 1) * 30_000,
                    "split": split,
                }
            )
            self.records.append(
                SimpleNamespace(
                    label_id=0,
                    parent_id=f"SYNTHETIC_{subject}_A_PARENT0",
                    subject_id=subject,
                    session_id=f"{subject}_A",
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_generic_train_only_standardizer_handles_each_modality():
    dataset = MiniDataset()
    video = MaskedStandardizer.fit_dataset_modality(
        dataset, modality="video", feature_names=("v0", "v1")
    )
    can = MaskedStandardizer.fit_dataset_modality(
        dataset, modality="can", feature_names=("c0",)
    )

    assert video.feature_dim == 2
    assert video.valid_token_count == 27
    assert can.feature_dim == 1
    assert can.valid_token_count == 36
    assert video.source_manifest_sha256 == "a" * 64
    transformed = can.transform_tensor(
        dataset[0]["inputs"]["can"]["x"],
        dataset[0]["inputs"]["can"]["valid_mask"],
    )
    assert transformed[-1].item() == 0.0


def test_generic_standardizer_rejects_val_and_missing_modality():
    with pytest.raises(ValueError, match="train split"):
        MaskedStandardizer.fit_dataset_modality(
            MiniDataset(split="val"), modality="video", feature_names=("v0", "v1")
        )
    with pytest.raises(ValueError, match="does not contain modality"):
        MaskedStandardizer.fit_dataset_modality(
            MiniDataset(), modality="audio", feature_names=("a0",)
        )


def test_frozen_g2_config_has_no_test_path_and_validates():
    config = load_config()
    _verify_config(config)
    assert "test_windows" not in config["data"]
    assert config["preflight"]["access_test_manifest"] is False
    assert config["preflight"]["evaluate_metrics"] is False

    changed = copy.deepcopy(config)
    changed["data"]["test_windows"] = "manifests/test.csv"
    with pytest.raises(ValueError, match="only the frozen train/val"):
        _verify_config(changed)


@pytest.mark.parametrize(
    ("filename", "model_class"),
    (
        ("simple_fusion_v1_train_val.json", "SimpleFusion"),
        ("mult_v1_train_val.json", "DualModalMulT"),
    ),
)
def test_train_val_configs_are_frozen_and_build_models(filename, model_class):
    config = json.loads((TRAIN_CONFIG_DIR / filename).read_text(encoding="utf-8"))
    verify_train_val_config(config)
    assert not any("test" in key.lower() for key in config["data"])
    assert config["evaluation"]["test_access"] is False
    model = build_training_model(config)
    assert type(model).__name__ == model_class


def test_train_val_config_rejects_a_test_manifest_key():
    config = json.loads(
        (TRAIN_CONFIG_DIR / "simple_fusion_v1_train_val.json").read_text(
            encoding="utf-8"
        )
    )
    config["data"]["test_manifest"] = "manifests/forbidden.csv"
    with pytest.raises(ValueError, match="must not contain test data keys"):
        verify_train_val_config(config)


def test_smoke_selector_is_deterministic_and_covers_all_classes():
    dataset = MiniDataset()
    first = _select_smoke_indices(dataset, 12)
    second = _select_smoke_indices(dataset, 12)
    assert first == second
    assert len(first) == len(set(first)) == 12
    assert {dataset.records[index].label_id for index in first} == {0, 1, 2}


def synthetic_batch(batch_size: int = 3) -> dict:
    video_mask = torch.tensor(
        [[True, True, False, True], [True, True, True, True], [True, False, True, True]]
    )
    can_mask = torch.tensor(
        [
            [True, True, True, False, True, True, False, True],
            [True, True, True, True, True, True, True, True],
            [True, False, True, True, False, True, True, True],
        ]
    )
    generator = torch.Generator().manual_seed(123)
    video_x = torch.randn(batch_size, 4, 96, generator=generator)
    can_x = torch.randn(batch_size, 8, 9, generator=generator)
    video_x[~video_mask] = 0
    can_x[~can_mask] = 0
    return {
        "inputs": {
            "video": {
                "x": video_x,
                "valid_mask": video_mask,
                "time_s": torch.arange(4, dtype=torch.float64).repeat(batch_size, 1),
            },
            "can": {
                "x": can_x,
                "valid_mask": can_mask,
                "time_s": torch.arange(8, dtype=torch.float64).repeat(batch_size, 1),
            },
        },
        "labels": torch.tensor([0, 1, 2], dtype=torch.int64),
    }


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_each_fusion_model_completes_forward_backward_and_reload(tmp_path, model_name):
    config = load_config()
    model = _model_from_config(model_name, config)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0

    report = _run_model_smoke(
        model_name=model_name,
        config=config,
        train_batch=synthetic_batch(),
        val_batch=synthetic_batch(),
        device=torch.device("cpu"),
        output_root=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["train_logits_shape"] == [3, 3]
    assert report["val_logits_shape"] == [3, 3]
    assert report["gradient_l2_norm"] > 0
    assert report["tail_padding_max_logit_difference"] <= 1e-6
    assert report["checkpoint_reload_max_logit_difference"] <= 1e-7
    checkpoint = tmp_path / report["checkpoint"]
    assert checkpoint.is_file()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["purpose"].startswith("one-step G2")


def test_simple_fusion_seed_runner_writes_complete_validation_package(tmp_path):
    config = json.loads(
        (TRAIN_CONFIG_DIR / "simple_fusion_v1_train_val.json").read_text(
            encoding="utf-8"
        )
    )
    config["training"]["physical_batch_size"] = 4
    config["training"]["effective_batch_size"] = 4
    config["training"]["max_epochs"] = 2
    config["training"]["early_stopping_patience"] = 1
    train = TinyPairedDataset(split="train", subject="D")
    val = TinyPairedDataset(split="val", subject="A")
    standardizers = {
        "video": MaskedStandardizer.fit_dataset_modality(
            train,
            modality="video",
            feature_names=tuple(f"v{index}" for index in range(96)),
        ),
        "can": MaskedStandardizer.fit_dataset_modality(
            train,
            modality="can",
            feature_names=tuple(f"c{index}" for index in range(9)),
        ),
    }

    result = run_seed(
        experiment_id="synthetic_fusion",
        experiment_root=tmp_path,
        seed=11,
        config=config,
        config_sha256="d" * 64,
        train_dataset=train,
        val_dataset=val,
        standardizers=standardizers,
        input_hashes={"pair_manifest": "c" * 64},
        git_state={"commit": "e" * 40, "branch": "synthetic", "dirty": False},
        device=torch.device("cpu"),
    )

    assert result["status"] == "PASS"
    assert result["validation_windows"] == 8
    assert result["validation_parents"] == 1
    assert result["test_manifest_accessed"] is False
    assert result["checkpoint_reload_max_probability_difference"] <= 1e-12
    seed_root = tmp_path / "seed_11"
    assert (seed_root / "checkpoints" / "best.pt").is_file()
    assert (seed_root / "predictions" / "val_windows.csv").is_file()
    assert (seed_root / "parent_predictions" / "val_parents.csv").is_file()
    report = json.loads((seed_root / "metrics.json").read_text(encoding="utf-8"))
    assert report["formal_result"] is False
    assert report["test_manifest_accessed"] is False
