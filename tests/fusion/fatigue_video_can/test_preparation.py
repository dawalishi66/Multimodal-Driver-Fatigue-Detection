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


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPOSITORY
    / "configs"
    / "fusion"
    / "fatigue_video_can"
    / "g2_preflight_v1.json"
)


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
