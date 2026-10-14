from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
import torch

from driver_state.baselines.fatigue_video import VideoGruBaseline
from tools.fusion.fatigue_video_can.train import _build_model, verify_train_val_config
from tools.baselines.fatigue_video.test_features import (
    export_features,
    validate_package,
    verify_unlock,
)


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


def test_test_feature_handoff_is_non_scoring_and_sample_relative(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    request = tmp_path / "request.csv"
    rows = []
    for index, subject in enumerate(("C", "H", "P")):
        start_ms = index * 30_000
        starts = start_ms / 1000.0 + np.arange(6, dtype=np.float64) * 5.0
        source_path = source / f"{subject}.npz"
        np.savez_compressed(
            source_path,
            x=np.ones((6, 96), dtype=np.float32) * index,
            time_s=starts + 2.5,
            valid_mask=np.ones(6, dtype=np.bool_),
            support_s=np.column_stack((starts, starts + 5.0)),
            observed_fraction=np.ones(6, dtype=np.float32),
        )
        rows.append(
            f"test_{subject},{subject},{subject}_A,{start_ms},{start_ms + 30000},"
            f"{subject}.npz,session_relative\n"
        )
    request.write_text(
        "sample_id,subject_id,session_id,window_start_ms,window_end_ms,"
        "source_feature,source_time_reference\n" + "".join(rows),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    report = export_features(request, source, output)

    assert report == validate_package(output)
    assert report["status"] == "PASS"
    assert report["subjects"] == ["C", "H", "P"]
    assert report["model_loaded"] is False
    assert report["predictions_generated"] is False
    assert report["metrics_evaluated"] is False
    with np.load(output / "features_30s/test_P.npz", allow_pickle=False) as archive:
        assert np.allclose(archive["support_s"][0], [0.0, 5.0])


def test_test_feature_unlock_forbids_model_evaluation(tmp_path):
    unlock = tmp_path / "unlock.json"
    unlock.write_text(
        json.dumps(
            {
                "authorized_by": "team lead",
                "issued_at": "2026-09-25T00:00:00Z",
                "scope": "test_feature_generation_and_validation_only",
                "allow_model_evaluation": True,
                "allow_threshold_or_model_changes": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_model_evaluation"):
        verify_unlock(unlock)
