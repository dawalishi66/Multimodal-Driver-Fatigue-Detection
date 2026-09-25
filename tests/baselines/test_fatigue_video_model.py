from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from driver_state.baselines.fatigue_video import VideoGruBaseline
from tools.fusion.fatigue_video_can.train import _build_model, verify_train_val_config


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY / "configs/baselines/fatigue_video/gru_v1_train_val.json"


def test_video_config_is_frozen_and_test_locked():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_train_val_config(config)
    assert config["modalities"] == ["video"]
    assert config["evaluation"]["test_access"] is False
    assert type(_build_model(config)).__name__ == "VideoGruBaseline"


def test_video_gru_ignores_masked_values_and_tail_padding():
    torch.manual_seed(7)
    model = VideoGruBaseline(dropout=0.0).eval()
    x = torch.randn(2, 6, 96)
    mask = torch.tensor(
        [[True, False, True, True, True, True], [True, True, True, True, True, True]]
    )
    reference = model({"video": {"x": x, "valid_mask": mask}})["logits"]

    changed = x.clone()
    changed[~mask] = 999_999
    changed_logits = model({"video": {"x": changed, "valid_mask": mask}})["logits"]
    padded_logits = model(
        {
            "video": {
                "x": torch.cat((changed, torch.randn(2, 3, 96)), dim=1),
                "valid_mask": torch.cat(
                    (mask, torch.zeros(2, 3, dtype=torch.bool)), dim=1
                ),
            }
        }
    )["logits"]

    assert torch.allclose(reference, changed_logits, atol=1e-7, rtol=0)
    assert torch.allclose(reference, padded_logits, atol=1e-7, rtol=0)


def test_video_gru_rejects_all_invalid_sample():
    model = VideoGruBaseline()
    with pytest.raises(ValueError, match="all-invalid"):
        model(
            {
                "video": {
                    "x": torch.zeros(1, 6, 96),
                    "valid_mask": torch.zeros(1, 6, dtype=torch.bool),
                }
            }
        )
