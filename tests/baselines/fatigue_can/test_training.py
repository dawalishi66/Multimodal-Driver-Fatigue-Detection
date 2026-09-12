import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from driver_state.data import MaskedStandardizer
from driver_state.preprocessing.fatigue_can.pipeline import CAN_FEATURE_COLUMNS
from tools.baselines.fatigue_can.train import (
    _make_loader,
    _predict,
    run_seed,
    verify_train_val_config,
)


class MemoryCanDataset:
    def __init__(self, *, split: str, subject: str, parents: int) -> None:
        self.split = split
        self.manifest_sha256 = f"synthetic-{split}-manifest"
        self.samples = []
        self.records = []
        for parent_index in range(parents):
            label = parent_index % 3
            parent_id = f"SYNTHETIC_{split}_P{parent_index}"
            for window_index in range(8):
                sample_id = f"{parent_id}_W{window_index}"
                x = torch.zeros(12, 9, dtype=torch.float32)
                x[:, label] = 1.0
                x[:, 6] = float(window_index) / 7
                valid_mask = torch.ones(12, dtype=torch.bool)
                valid_mask[5] = False
                x[5] = 999_999
                self.samples.append({
                    "inputs": {
                        "can": {
                            "x": x,
                            "time_s": torch.arange(12, dtype=torch.float64),
                            "valid_mask": valid_mask,
                            "support_s": torch.zeros(12, 2, dtype=torch.float64),
                            "observed_fraction": valid_mask.to(torch.float32),
                        }
                    },
                    "label": torch.tensor(label, dtype=torch.int64),
                    "sample_id": sample_id,
                    "parent_id": parent_id,
                    "subject_id": subject,
                    "session_id": f"{subject}_A",
                    "window_index": window_index,
                    "split": split,
                })
                self.records.append(SimpleNamespace(
                    sample_id=sample_id,
                    parent_id=parent_id,
                    subject_id=subject,
                    session_id=f"{subject}_A",
                    label_id=label,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def training_config() -> dict:
    repository = Path(__file__).resolve().parents[3]
    return json.loads(
        (repository / "configs/baselines/fatigue_can/gru_v1_train_val.json")
        .read_text(encoding="utf-8")
    )


def test_training_config_has_no_test_path_and_rejects_one():
    config = training_config()
    verify_train_val_config(config)
    assert not any("test" in key.lower() for key in config["data"])
    changed = copy.deepcopy(config)
    changed["data"]["test_windows"] = "manifests/forbidden.csv"
    with pytest.raises(ValueError, match="must not contain test data keys"):
        verify_train_val_config(changed)


def test_seeded_training_loader_order_is_reproducible():
    dataset = MemoryCanDataset(split="train", subject="D", parents=2)
    normalizer = MaskedStandardizer.fit_can_dataset(
        dataset, feature_names=CAN_FEATURE_COLUMNS
    )

    def order(seed):
        loader = _make_loader(
            dataset,
            standardizer=normalizer,
            batch_size=5,
            shuffle=True,
            seed=seed,
        )
        return [sample_id for batch in loader for sample_id in batch["sample_id"]]

    assert order(11) == order(11)


def test_run_seed_writes_reloadable_train_val_package(tmp_path):
    config = training_config()
    config["training"]["max_epochs"] = 3
    config["training"]["early_stopping_patience"] = 1
    train_dataset = MemoryCanDataset(split="train", subject="D", parents=3)
    val_dataset = MemoryCanDataset(split="val", subject="A", parents=3)
    normalizer = MaskedStandardizer.fit_can_dataset(
        train_dataset, feature_names=CAN_FEATURE_COLUMNS
    )
    result = run_seed(
        experiment_id="synthetic_experiment",
        experiment_root=tmp_path,
        seed=11,
        config=config,
        config_sha256="synthetic-config",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        standardizer=normalizer,
        manifest_hashes={"train": "synthetic", "val": "synthetic"},
        git_state={"commit": "synthetic-commit", "branch": "test", "dirty": False},
        device=torch.device("cpu"),
    )
    seed_root = tmp_path / "seed_11"
    assert result["status"] == "PASS"
    assert result["validation_windows"] == 24
    assert result["validation_parents"] == 3
    assert result["test_manifest_accessed"] is False
    assert result["checkpoint_reload_max_probability_difference"] == 0.0
    assert (seed_root / "checkpoints/best.pt").is_file()
    assert (seed_root / "predictions/val_windows.csv").is_file()
    assert (seed_root / "parent_predictions/val_parents.csv").is_file()
    manifest = json.loads((seed_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_result"] is False
    assert manifest["test_manifest_accessed"] is False


def test_prediction_rejects_non_finite_logits():
    dataset = MemoryCanDataset(split="val", subject="A", parents=1)
    train_dataset = MemoryCanDataset(split="train", subject="D", parents=1)
    normalizer = MaskedStandardizer.fit_can_dataset(
        train_dataset, feature_names=CAN_FEATURE_COLUMNS
    )
    loader = _make_loader(
        dataset,
        standardizer=normalizer,
        batch_size=8,
        shuffle=False,
        seed=11,
    )

    class NonFiniteModel(torch.nn.Module):
        def forward(self, inputs):
            batch_size = inputs["can"]["x"].shape[0]
            return {"logits": torch.full((batch_size, 3), float("nan"))}

    with pytest.raises(RuntimeError, match="validation logits contain NaN or Inf"):
        _predict(
            NonFiniteModel(), loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
        )
