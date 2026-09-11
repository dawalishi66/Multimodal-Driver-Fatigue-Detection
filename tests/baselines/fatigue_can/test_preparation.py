import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from driver_state.data import CanWindowDataset, MaskedStandardizer, make_can_collate
from driver_state.data.can import file_sha256
from driver_state.baselines.fatigue_can import CanGruBaseline
from driver_state.preprocessing.fatigue_can.pipeline import (
    CAN_FEATURE_COLUMNS,
    CAN_METADATA_FIELDS,
    CAN_PARENT_FIELDS,
)


def test_can_baseline_files_are_isolated_from_fusion_models():
    repository = Path(__file__).resolve().parents[3]
    assert (
        repository
        / "src"
        / "driver_state"
        / "baselines"
        / "fatigue_can"
        / "model.py"
    ).is_file()
    assert not (
        repository
        / "src"
        / "driver_state"
        / "models"
        / "unimodal"
        / "can_gru.py"
    ).exists()
    assert not list(
        (repository / "src" / "driver_state" / "models" / "mult").glob("*can_gru*")
    )
    assert not list(
        (repository / "src" / "driver_state" / "models" / "simple_fusion").glob(
            "*can_gru*"
        )
    )


def write_complete_parent(tmp_path, *, split="train", subject="D"):
    root = tmp_path / split
    feature_dir = root / "features"
    manifest_dir = root / "manifests"
    feature_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    parent_id = f"SYNTHETIC_{subject}_A_P0"
    rows = []
    for window_index in range(8):
        sample_id = f"SYNTHETIC_{subject}_A_{window_index}"
        feature = feature_dir / f"{sample_id}.npz"
        x = np.tile(np.arange(9, dtype=np.float32), (300, 1))
        x[:, 6] += window_index
        valid_mask = np.ones(300, dtype=bool)
        valid_mask[10] = False
        x[10] = 999_999
        edges = np.arange(301, dtype=np.float64) / 10
        np.savez(
            feature,
            x=x,
            time_s=(edges[:-1] + edges[1:]) / 2,
            valid_mask=valid_mask,
            support_s=np.column_stack((edges[:-1], edges[1:])),
            observed_fraction=np.where(valid_mask, 1.0, 0.0).astype(np.float32),
        )
        row = {field: "" for field in CAN_METADATA_FIELDS}
        row.update({
            "sample_id": sample_id,
            "modality": "can",
            "subject_id": subject,
            "session_id": f"{subject}_A",
            "split": split,
            "source_file": f"synthetic/{subject}_Telemetry_A.csv",
            "window_index": str(window_index),
            "window_start_ms": str(window_index * 30_000),
            "window_end_ms": str((window_index + 1) * 30_000),
            "duration_ms": "30000",
            "label_start_ms": "0",
            "label_end_ms": "240000",
            "kss_score": "3",
            "label_class": "low",
            "label_id": "0",
            "valid": "true",
            "valid_ratio": f"{299 / 300:.6f}",
            "mask": f"features/{feature.name}::valid_mask",
            "feature_path": f"features/{feature.name}",
            "feature_shape": "[300,9]",
            "feature_dtype": "float32",
            "extractor_name": "synthetic_can",
            "extractor_version": "1.0.0",
            "error": "",
            "parent_id": parent_id,
            "feature_columns": json.dumps(list(CAN_FEATURE_COLUMNS)),
            "feature_sha256": file_sha256(feature),
            "qc_status": "PASS",
            "qc_reason_codes": "",
        })
        rows.append(row)
    windows = manifest_dir / f"can_{split}_windows_v1.csv"
    with windows.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAN_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    parents = manifest_dir / f"can_{split}_parents_240s_v1.csv"
    parent = {field: "" for field in CAN_PARENT_FIELDS}
    parent.update({
        "parent_id": parent_id,
        "subject_id": subject,
        "session_id": f"{subject}_A",
        "split": split,
        "label_start_ms": "0",
        "label_end_ms": "240000",
        "kss_score": "3",
        "label_class": "low",
        "label_id": "0",
        "window_count": "8",
        "valid_window_count": "8",
        "complete_for_240s": "true",
        "qc_reason_codes": "",
    })
    with parents.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAN_PARENT_FIELDS)
        writer.writeheader()
        writer.writerow(parent)
    return root, windows, parents


def test_complete_parent_dataset_normalizer_and_collate(tmp_path):
    root, windows, parents = write_complete_parent(tmp_path)
    dataset = CanWindowDataset(
        windows, feature_root=root, split="train", parent_manifest=parents
    )
    assert len(dataset) == 8
    assert dataset[0]["inputs"]["can"]["x"].shape == (300, 9)
    standardizer = MaskedStandardizer.fit_can_dataset(
        dataset, feature_names=CAN_FEATURE_COLUMNS
    )
    assert standardizer.fitted_split == "train"
    assert standardizer.valid_token_count == 8 * 299
    assert standardizer.mean[0] == pytest.approx(0.0)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        collate_fn=make_can_collate(standardizer),
    )
    batch = next(iter(loader))
    assert batch["inputs"]["can"]["x"].shape == (4, 300, 9)
    assert batch["labels"].dtype == torch.int64
    assert torch.all(batch["inputs"]["can"]["x"][:, 10] == 0)
    assert batch["sample_id"] == [record.sample_id for record in dataset.records[:4]]


def test_normalizer_refuses_validation_fit(tmp_path):
    root, windows, parents = write_complete_parent(
        tmp_path, split="val", subject="A"
    )
    dataset = CanWindowDataset(
        windows, feature_root=root, split="val", parent_manifest=parents
    )
    with pytest.raises(ValueError, match="only be fitted on the train"):
        MaskedStandardizer.fit_can_dataset(dataset, feature_names=CAN_FEATURE_COLUMNS)


def test_dataset_locks_test_by_default(tmp_path):
    with pytest.raises(PermissionError, match="test manifest access is locked"):
        CanWindowDataset(
            tmp_path / "not_opened.csv",
            feature_root=tmp_path,
            split="test",
        )


def test_standardizer_round_trip(tmp_path):
    root, windows, parents = write_complete_parent(tmp_path)
    dataset = CanWindowDataset(
        windows, feature_root=root, split="train", parent_manifest=parents
    )
    original = MaskedStandardizer.fit_can_dataset(
        dataset, feature_names=CAN_FEATURE_COLUMNS
    )
    path = tmp_path / "normalizer.json"
    original.save_json(path)
    restored = MaskedStandardizer.from_json(path)
    assert restored == original


def test_can_gru_is_mask_and_padding_invariant():
    torch.manual_seed(7)
    model = CanGruBaseline(dropout=0.2).eval()
    x = torch.randn(3, 6, 9)
    mask = torch.tensor([
        [True, True, False, True, False, False],
        [True, False, True, True, True, False],
        [True, False, False, False, False, False],
    ])
    inputs = {"can": {"x": x, "valid_mask": mask}}
    with torch.no_grad():
        reference = model(inputs)
        changed = x.clone()
        changed[~mask] = 1_000_000
        changed_logits = model(
            {"can": {"x": changed, "valid_mask": mask}}
        )["logits"]
        tail_x = torch.cat((x, torch.randn(3, 4, 9) * 1000), dim=1)
        tail_mask = torch.cat((mask, torch.zeros(3, 4, dtype=torch.bool)), dim=1)
        tail_logits = model(
            {"can": {"x": tail_x, "valid_mask": tail_mask}}
        )["logits"]
    assert reference["logits"].shape == (3, 3)
    assert reference["embedding"].shape == (3, 64)
    assert torch.allclose(reference["logits"], changed_logits, rtol=0, atol=1e-6)
    assert torch.allclose(reference["logits"], tail_logits, rtol=0, atol=1e-6)
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        permuted = model({
            "can": {
                "x": x[permutation],
                "valid_mask": mask[permutation],
            }
        })["logits"]
    assert torch.allclose(
        reference["logits"][permutation], permuted, rtol=0, atol=1e-6
    )


def test_can_gru_rejects_all_invalid_sample():
    model = CanGruBaseline()
    with pytest.raises(ValueError, match="all-invalid"):
        model({
            "can": {
                "x": torch.zeros(2, 4, 9),
                "valid_mask": torch.tensor([
                    [True, False, False, False],
                    [False, False, False, False],
                ]),
            }
        })
