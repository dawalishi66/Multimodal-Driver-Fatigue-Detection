from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from driver_state.data import (
    FatigueVideoCanDataset,
    collate_fatigue_video_can_batch,
    make_fusion_model_inputs,
)


PAIR_FIELDS = [
    "paired_sample_id", "parent_id", "subject_id", "session_id", "split",
    "window_index", "window_start_ms", "window_end_ms", "duration_ms",
    "label_start_ms", "label_end_ms", "kss_score", "label_class", "label_id",
    "paired_valid", "exclude_reason", "complete8_eligible",
    "complete8_exclude_reason", "sync_status", "offset_ms",
    "hardware_clock_verified", "video_sample_id", "video_subwindow_ids",
    "video_feature_archive", "video_feature_member", "video_feature_shape",
    "video_feature_dtype", "video_feature_sha256", "video_source_time_reference",
    "video_time_transform", "can_sample_id", "can_feature_path",
    "can_feature_shape", "can_feature_dtype", "can_feature_sha256",
    "can_valid_ratio",
]


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _video_arrays(window_start_s: float) -> dict[str, np.ndarray]:
    starts = window_start_s + np.arange(6, dtype=np.float64) * 5.0
    return {
        "x": np.arange(6 * 96, dtype=np.float32).reshape(6, 96),
        "time_s": starts + 2.5,
        "valid_mask": np.ones(6, dtype=np.bool_),
        "support_s": np.column_stack((starts, starts + 5.0)),
        "observed_fraction": np.ones(6, dtype=np.float32),
    }


def _can_arrays() -> dict[str, np.ndarray]:
    starts = np.arange(300, dtype=np.float64) / 10.0
    return {
        "x": np.arange(300 * 9, dtype=np.float32).reshape(300, 9),
        "time_s": starts + 0.05,
        "valid_mask": np.ones(300, dtype=np.bool_),
        "support_s": np.column_stack((starts, starts + 0.1)),
        "observed_fraction": np.ones(300, dtype=np.float32),
    }


def _build_fixture(
    root: Path,
    *,
    subject: str = "D",
    split: str = "train",
    window_count: int = 8,
    mutate_row=None,
    mutate_video=None,
) -> Path:
    session = f"{subject}_A"
    parent_id = f"ULDD_{subject}_A_000000000_000240000"
    can_dir = root / "can"
    can_dir.mkdir(parents=True)
    rows = []
    video_payloads: dict[str, bytes] = {}
    for index in range(window_count):
        start_ms = index * 30_000
        end_ms = start_ms + 30_000
        sample_id = f"ULDD_{subject}_A_{start_ms:09d}_{end_ms:09d}"
        video_member = f"features/{sample_id}.npz"
        video_arrays = _video_arrays(start_ms / 1000.0)
        if mutate_video is not None and index == 0:
            mutate_video(video_arrays)
        video_payload = _npz_bytes(video_arrays)
        video_payloads[video_member] = video_payload
        can_path = can_dir / f"{sample_id}.npz"
        np.savez(can_path, **_can_arrays())
        row = {
            "paired_sample_id": sample_id,
            "parent_id": parent_id,
            "subject_id": subject,
            "session_id": session,
            "split": split,
            "window_index": str(index),
            "window_start_ms": str(start_ms),
            "window_end_ms": str(end_ms),
            "duration_ms": "30000",
            "label_start_ms": "0",
            "label_end_ms": "240000",
            "kss_score": "3.0",
            "label_class": "low",
            "label_id": "0",
            "paired_valid": "true",
            "exclude_reason": "",
            "complete8_eligible": "true",
            "complete8_exclude_reason": "",
            "sync_status": "provider_documented_aligned",
            "offset_ms": "0",
            "hardware_clock_verified": "false",
            "video_sample_id": f"video30__{session}__{start_ms}_{end_ms}",
            "video_subwindow_ids": "|".join(
                f"video5__{session}__{index}_{subindex}" for subindex in range(6)
            ),
            "video_feature_archive": "video.zip",
            "video_feature_member": video_member,
            "video_feature_shape": "[6,96]",
            "video_feature_dtype": "float32",
            "video_feature_sha256": hashlib.sha256(video_payload).hexdigest(),
            "video_source_time_reference": "session_relative",
            "video_time_transform": "subtract_window_start_ms_div_1000",
            "can_sample_id": sample_id,
            "can_feature_path": can_path.relative_to(root).as_posix(),
            "can_feature_shape": "[300,9]",
            "can_feature_dtype": "float32",
            "can_feature_sha256": hashlib.sha256(can_path.read_bytes()).hexdigest(),
            "can_valid_ratio": "1.000000",
        }
        if mutate_row is not None and index == 0:
            mutate_row(row)
        rows.append(row)

    with zipfile.ZipFile(root / "video.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for member, payload in video_payloads.items():
            archive.writestr(member, payload)
    manifest = root / "pairs.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_loads_complete_parent_and_converts_video_time(tmp_path: Path):
    manifest = _build_fixture(tmp_path)
    dataset = FatigueVideoCanDataset(
        manifest,
        dataset_root=tmp_path,
        split="train",
        verify_feature_hashes=True,
    )

    assert len(dataset) == 8
    assert dataset.parent_count == 1
    sample = dataset[1]
    assert sample["sample_id"] == "ULDD_D_A_000030000_000060000"
    assert sample["inputs"]["video"]["x"].shape == (6, 96)
    assert sample["inputs"]["can"]["x"].shape == (300, 9)
    torch.testing.assert_close(
        sample["inputs"]["video"]["time_s"],
        torch.tensor([2.5, 7.5, 12.5, 17.5, 22.5, 27.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        sample["inputs"]["video"]["support_s"][0],
        torch.tensor([0.0, 5.0], dtype=torch.float64),
    )


def test_cached_features_are_returned_as_independent_tensors(tmp_path: Path):
    dataset = FatigueVideoCanDataset(
        _build_fixture(tmp_path), dataset_root=tmp_path, split="train"
    )
    first = dataset[0]
    first["inputs"]["video"]["x"].fill_(999.0)
    second = dataset[0]
    assert not torch.all(second["inputs"]["video"]["x"] == 999.0)


def test_collate_retains_audit_fields_and_pads_modalities_independently(tmp_path: Path):
    dataset = FatigueVideoCanDataset(
        _build_fixture(tmp_path), dataset_root=tmp_path, split="train"
    )
    short = dataset[0]
    full = dataset[1]
    for name in ("x", "time_s", "valid_mask", "support_s", "observed_fraction"):
        short["inputs"]["video"][name] = short["inputs"]["video"][name][:-1]
        short["inputs"]["can"][name] = short["inputs"]["can"][name][:-10]
    short["inputs"]["video"]["valid_mask"][1] = False
    short["inputs"]["video"]["x"][1] = 123.0

    batch = collate_fatigue_video_can_batch([short, full])

    assert batch["inputs"]["video"]["x"].shape == (2, 6, 96)
    assert batch["inputs"]["can"]["x"].shape == (2, 300, 9)
    assert batch["labels"].shape == (2,)
    assert set(batch["inputs"]["video"]) == {
        "x", "time_s", "valid_mask", "support_s", "observed_fraction"
    }
    assert not batch["inputs"]["video"]["valid_mask"][0, -1]
    assert not batch["inputs"]["can"]["valid_mask"][0, -10:].any()
    assert torch.all(batch["inputs"]["video"]["x"][0, 1] == 0.0)


def test_model_input_adapter_removes_audit_only_arrays(tmp_path: Path):
    dataset = FatigueVideoCanDataset(
        _build_fixture(tmp_path), dataset_root=tmp_path, split="train"
    )
    batch = collate_fatigue_video_can_batch([dataset[0], dataset[1]])

    model_inputs = make_fusion_model_inputs(batch)

    assert set(model_inputs) == {"video", "can"}
    assert set(model_inputs["video"]) == {"x", "valid_mask", "time_s"}
    assert set(model_inputs["can"]) == {"x", "valid_mask", "time_s"}
    assert model_inputs["video"]["x"] is batch["inputs"]["video"]["x"]


def test_batch_permutation_preserves_sample_tensor_pairing(tmp_path: Path):
    dataset = FatigueVideoCanDataset(
        _build_fixture(tmp_path), dataset_root=tmp_path, split="train"
    )
    samples = [dataset[0], dataset[1], dataset[2]]
    forward = collate_fatigue_video_can_batch(samples)
    reverse = collate_fatigue_video_can_batch(list(reversed(samples)))

    assert reverse["sample_id"] == list(reversed(forward["sample_id"]))
    torch.testing.assert_close(
        reverse["inputs"]["video"]["x"],
        forward["inputs"]["video"]["x"].flip(0),
    )
    torch.testing.assert_close(
        reverse["inputs"]["can"]["x"],
        forward["inputs"]["can"]["x"].flip(0),
    )


def test_test_split_is_locked(tmp_path: Path):
    manifest = _build_fixture(tmp_path, subject="C", split="test")
    with pytest.raises(PermissionError):
        FatigueVideoCanDataset(manifest, dataset_root=tmp_path, split="test")


def test_incomplete_parent_is_rejected(tmp_path: Path):
    manifest = _build_fixture(tmp_path, window_count=7)
    with pytest.raises(ValueError, match="window indices 0..7"):
        FatigueVideoCanDataset(manifest, dataset_root=tmp_path, split="train")


def test_hash_mismatch_is_rejected_when_verification_is_enabled(tmp_path: Path):
    manifest = _build_fixture(
        tmp_path,
        mutate_row=lambda row: row.update(video_feature_sha256="0" * 64),
    )
    dataset = FatigueVideoCanDataset(
        manifest,
        dataset_root=tmp_path,
        split="train",
        verify_feature_hashes=True,
    )
    with pytest.raises(ValueError, match="video feature hash mismatch"):
        dataset[0]


def test_nonfinite_feature_is_rejected(tmp_path: Path):
    def add_nan(arrays: dict[str, np.ndarray]) -> None:
        arrays["x"][0, 0] = np.nan

    dataset = FatigueVideoCanDataset(
        _build_fixture(tmp_path, mutate_video=add_nan),
        dataset_root=tmp_path,
        split="train",
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        dataset[0]


def test_unsafe_feature_path_is_rejected(tmp_path: Path):
    manifest = _build_fixture(
        tmp_path,
        mutate_row=lambda row: row.update(can_feature_path="../escape.npz"),
    )
    with pytest.raises(ValueError, match="safe, non-empty relative path"):
        FatigueVideoCanDataset(manifest, dataset_root=tmp_path, split="train")


def test_collate_rejects_unknown_standardizer_modality(tmp_path: Path):
    dataset = FatigueVideoCanDataset(
        _build_fixture(tmp_path), dataset_root=tmp_path, split="train"
    )
    with pytest.raises(ValueError, match="unknown modalities"):
        collate_fatigue_video_can_batch(
            [dataset[0]], standardizers={"audio": object()}
        )
