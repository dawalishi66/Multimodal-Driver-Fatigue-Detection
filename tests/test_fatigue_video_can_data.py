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
from tools.build_fatigue_video_can_pairs import validate_video_features


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


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


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
    extra_video_source: tuple[str, str, str, int] | None = None,
    mutate_source5=None,
) -> Path:
    session = f"{subject}_A"
    parent_id = f"ULDD_{subject}_A_000000000_000240000"
    can_dir = root / "can"
    can_dir.mkdir(parents=True)
    rows = []
    video_payloads: dict[str, bytes] = {}
    source30_rows: list[dict[str, str]] = []
    source5_rows: list[dict[str, str]] = []
    feature_rows: list[dict[str, str]] = []
    for index in range(window_count):
        start_ms = index * 30_000
        end_ms = start_ms + 30_000
        sample_id = f"ULDD_{subject}_A_{start_ms:09d}_{end_ms:09d}"
        video_sample_id = f"video30__{session}__{start_ms:010d}_{end_ms:010d}"
        video_member = f"features_30s/{video_sample_id}.npz"
        subwindow_ids = tuple(
            f"video5__{session}_IR_{(start_ms + subindex * 5_000) * 1000:013d}_"
            f"{(start_ms + (subindex + 1) * 5_000) * 1000:013d}"
            for subindex in range(6)
        )
        video_arrays = _video_arrays(start_ms / 1000.0)
        if mutate_video is not None and index == 0:
            mutate_video(video_arrays)
        video_payload = _npz_bytes(video_arrays)
        video_payloads[video_member] = video_payload
        video_sha = hashlib.sha256(video_payload).hexdigest()
        source30_rows.append({
            "sample_id": video_sample_id,
            "subject_id": subject,
            "session_id": session,
            "split": split,
            "window_start_ms": str(start_ms),
            "window_end_ms": str(end_ms),
            "label_start_ms": "0",
            "label_end_ms": "240000",
            "kss_score": "3.0",
            "label_class": "low",
            "label_id": "0",
            "valid": "true",
            "video_subwindow_ids": "|".join(subwindow_ids),
        })
        feature_rows.append({
            "sample_id": video_sample_id,
            "relative_path": video_member,
            "feature_shape": "[6,96]",
            "Dv": "96",
            "feature_dtype": "float32",
            "feature_sha256": video_sha,
            "video_subwindow_ids": "|".join(subwindow_ids),
        })
        for subindex, subwindow_id in enumerate(subwindow_ids):
            source5_rows.append({
                "sample_id": subwindow_id,
                "subject_id": subject,
                "session_id": session,
                "split": split,
                "window_start_ms": str(start_ms + subindex * 5_000),
                "window_end_ms": str(start_ms + (subindex + 1) * 5_000),
                "label_start_ms": "0",
                "label_end_ms": "240000",
                "kss_score": "3.0",
                "label_class": "low",
                "label_id": "0",
                "valid": "true",
                "feature_path": video_member,
                "feature_sha256": video_sha,
            })
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
            "video_sample_id": video_sample_id,
            "video_subwindow_ids": "|".join(subwindow_ids),
            "video_feature_archive": "video.zip",
            "video_feature_member": video_member,
            "video_feature_shape": "[6,96]",
            "video_feature_dtype": "float32",
            "video_feature_sha256": video_sha,
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

    if extra_video_source is not None:
        other_subject, other_session, other_split, other_label_id = extra_video_source
        other_class = ("low", "medium", "high")[other_label_id]
        other_score = (3.0, 5.0, 7.0)[other_label_id]
        other_id = f"video30__{other_session}__0000000000_0000030000"
        other_member = f"features_30s/{other_id}.npz"
        other_ids = tuple(
            f"video5__{other_session}_IR_{subindex * 5_000_000:013d}_"
            f"{(subindex + 1) * 5_000_000:013d}"
            for subindex in range(6)
        )
        other_arrays = _video_arrays(0.0)
        other_arrays["x"] += 100.0
        other_payload = _npz_bytes(other_arrays)
        other_sha = hashlib.sha256(other_payload).hexdigest()
        video_payloads[other_member] = other_payload
        source30_rows.append({
            "sample_id": other_id,
            "subject_id": other_subject,
            "session_id": other_session,
            "split": other_split,
            "window_start_ms": "0",
            "window_end_ms": "30000",
            "label_start_ms": "0",
            "label_end_ms": "240000",
            "kss_score": str(other_score),
            "label_class": other_class,
            "label_id": str(other_label_id),
            "valid": "true",
            "video_subwindow_ids": "|".join(other_ids),
        })
        feature_rows.append({
            "sample_id": other_id,
            "relative_path": other_member,
            "feature_shape": "[6,96]",
            "Dv": "96",
            "feature_dtype": "float32",
            "feature_sha256": other_sha,
            "video_subwindow_ids": "|".join(other_ids),
        })
        for subindex, other_subwindow_id in enumerate(other_ids):
            source5_rows.append({
                "sample_id": other_subwindow_id,
                "subject_id": other_subject,
                "session_id": other_session,
                "split": other_split,
                "window_start_ms": str(subindex * 5_000),
                "window_end_ms": str((subindex + 1) * 5_000),
                "label_start_ms": "0",
                "label_end_ms": "240000",
                "kss_score": str(other_score),
                "label_class": other_class,
                "label_id": str(other_label_id),
                "valid": "true",
                "feature_path": other_member,
                "feature_sha256": other_sha,
            })

    if mutate_source5 is not None:
        mutate_source5(source5_rows[0])

    with zipfile.ZipFile(root / "video.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for member, payload in video_payloads.items():
            archive.writestr(member, payload)
        archive.writestr("video_windows_30s_v2.csv", _csv_bytes(source30_rows))
        archive.writestr("video_windows_5s_v2.csv", _csv_bytes(source5_rows))
        archive.writestr("video_feature_index_v2.csv", _csv_bytes(feature_rows))
    manifest = root / "pairs.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _rewrite_first_pair(manifest: Path, update) -> None:
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    update(rows[0])
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _point_first_pair_at_video(manifest: Path, video_id: str) -> None:
    with zipfile.ZipFile(manifest.parent / "video.zip") as archive:
        features = list(csv.DictReader(io.StringIO(
            archive.read("video_feature_index_v2.csv").decode("utf-8")
        )))
    feature = next(row for row in features if row["sample_id"] == video_id)
    _rewrite_first_pair(manifest, lambda row: row.update({
        "video_sample_id": video_id,
        "video_subwindow_ids": feature["video_subwindow_ids"],
        "video_feature_member": feature["relative_path"],
        "video_feature_sha256": feature["feature_sha256"],
    }))


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


def test_pair_hash_mismatch_is_rejected_against_source_index(tmp_path: Path):
    manifest = _build_fixture(
        tmp_path,
        mutate_row=lambda row: row.update(video_feature_sha256="0" * 64),
    )
    with pytest.raises(ValueError, match="pair_video_feature_sha256"):
        FatigueVideoCanDataset(
            manifest,
            dataset_root=tmp_path,
            split="train",
            verify_feature_hashes=True,
        )


@pytest.mark.parametrize(
    ("foreign", "expected_error"),
    [
        (("A", "A_D", "val", 1), "pair_video_sample_id"),
        (("D", "D_D", "train", 1), "pair_video_session_id"),
        (("D", "D_D", "val", 1), "pair_video_split"),
    ],
)
def test_real_source_swap_cannot_cross_subject_session_or_split(
    tmp_path: Path, foreign, expected_error: str
):
    manifest = _build_fixture(tmp_path, extra_video_source=foreign)
    foreign_id = f"video30__{foreign[1]}__0000000000_0000030000"
    _point_first_pair_at_video(manifest, foreign_id)
    with pytest.raises(ValueError, match=expected_error):
        FatigueVideoCanDataset(
            manifest,
            dataset_root=tmp_path,
            split="train",
            verify_feature_hashes=True,
        )


def test_reversed_six_video_sources_are_rejected(tmp_path: Path):
    manifest = _build_fixture(
        tmp_path,
        mutate_row=lambda row: row.update(
            video_subwindow_ids="|".join(reversed(row["video_subwindow_ids"].split("|")))
        ),
    )
    with pytest.raises(ValueError, match="pair_video_5s_order_or_identity"):
        FatigueVideoCanDataset(manifest, dataset_root=tmp_path, split="train")


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("window_end_ms", "4000", "source_5s_time_0"),
        ("split", "val", "source_5s_split_0"),
        ("label_end_ms", "480000", "source_5s_label_end_ms_0"),
    ],
)
def test_five_second_source_index_must_match_time_split_and_parent(
    tmp_path: Path, field: str, value: str, expected_error: str
):
    manifest = _build_fixture(
        tmp_path,
        mutate_source5=lambda row: row.update({field: value}),
    )
    with pytest.raises(ValueError, match=expected_error):
        FatigueVideoCanDataset(manifest, dataset_root=tmp_path, split="train")


@pytest.mark.parametrize("bad_support", ["gaps", "short", "overlap"])
def test_video_support_must_be_six_contiguous_five_second_segments(
    tmp_path: Path, bad_support: str
):
    def mutate(arrays: dict[str, np.ndarray]) -> None:
        if bad_support == "gaps":
            arrays["support_s"][:, 0] += 1.0
            arrays["support_s"][:, 1] -= 1.0
        elif bad_support == "short":
            arrays["support_s"][0, 1] = 4.0
        else:
            arrays["support_s"][1, 0] = 4.0

    manifest = _build_fixture(tmp_path, mutate_video=mutate)
    dataset = FatigueVideoCanDataset(
        manifest, dataset_root=tmp_path, split="train", verify_feature_hashes=True
    )
    with pytest.raises(ValueError, match="video_support_not_contiguous_5s_grid"):
        dataset[0]


def test_video_centers_must_match_source_five_second_grid(tmp_path: Path):
    def move_center(arrays: dict[str, np.ndarray]) -> None:
        arrays["time_s"][0] = 2.0

    manifest = _build_fixture(tmp_path, mutate_video=move_center)
    dataset = FatigueVideoCanDataset(manifest, dataset_root=tmp_path, split="train")
    with pytest.raises(ValueError, match="video_time_not_5s_centers"):
        dataset[0]


@pytest.mark.parametrize("failure", ["source_time", "feature_grid"])
def test_pair_builder_rejects_bad_five_second_source_or_feature_grid(
    tmp_path: Path, failure: str
):
    def short_support(arrays: dict[str, np.ndarray]) -> None:
        arrays["support_s"][0, 1] = 4.0

    _build_fixture(
        tmp_path,
        mutate_source5=(
            (lambda row: row.update(window_end_ms="4000"))
            if failure == "source_time" else None
        ),
        mutate_video=short_support if failure == "feature_grid" else None,
    )
    with zipfile.ZipFile(tmp_path / "video.zip") as archive:
        windows = list(csv.DictReader(io.StringIO(
            archive.read("video_windows_30s_v2.csv").decode("utf-8")
        )))
        features = list(csv.DictReader(io.StringIO(
            archive.read("video_feature_index_v2.csv").decode("utf-8")
        )))
        _, errors = validate_video_features(archive, windows, features)
    expected = (
        "source_5s_time_0" if failure == "source_time"
        else "video_support_not_contiguous_5s_grid"
    )
    assert any(expected in error for error in errors)


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
