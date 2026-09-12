"""Synthetic tests for video metadata building, validation and pair checking."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from driver_state.preprocessing.distraction_video.build_metadata import (
    VIDEO_6C_CSV_FIELDS,
    build_6c_rows,
    load_feature_index,
)
from driver_state.preprocessing.distraction_video.check_pairs import check_pairs
from driver_state.preprocessing.distraction_video.fusion_labels import (
    SIX_CLASS_NAMES,
    SIX_CLASS_TASKS,
    VIDEO_LABEL_SCHEME_NAME,
    load_label_scheme,
    load_subject_splits,
)
from driver_state.preprocessing.distraction_video.validate_video_6c import validate_video_6c

SAMPLE_ID = "01_P01_20231111_09_31_43_12"
SESSION_ID = "P01_20231111_0931_43"
SPLIT_MAP = {"P01": "val"}


def _scheme(tmp: Path) -> Path:
    path = tmp / "label_scheme.json"
    path.write_text(
        json.dumps({
            "label_scheme": VIDEO_LABEL_SCHEME_NAME,
            "class_names": list(SIX_CLASS_NAMES),
            "task_to_class": {task: index for index, task in enumerate(SIX_CLASS_TASKS)},
        }),
        encoding="utf-8",
    )
    return path


def _splits(tmp: Path) -> Path:
    path = tmp / "splits.json"
    path.write_text(json.dumps({"splits": SPLIT_MAP}), encoding="utf-8")
    return path


def _manifest() -> list[dict]:
    return [{
        "sample_id": SAMPLE_ID,
        "subject_id": "P01",
        "session_id": SESSION_ID,
        "label_raw": "01",
        "source_refs": {"video_file": f"Upper_body_video_01/{SAMPLE_ID}.mp4"},
    }]


def _feature_root(tmp: Path) -> Path:
    root = tmp / "processed"
    path = root / "video_features_v1" / f"{SAMPLE_ID}.npz"
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        x=np.zeros((10, 8), dtype=np.float32),
        time_s=np.arange(10, dtype=np.float64) + 0.5,
        valid_mask=np.ones(10, dtype=bool),
        support_s=np.array([[index, index + 1] for index in range(10)], dtype=np.float64),
        observed_fraction=np.ones(10, dtype=np.float32),
    )
    return root


def _feature_index(tmp: Path) -> Path:
    path = tmp / "feature_index.jsonl"
    path.write_text(
        json.dumps({
            "sample_id": SAMPLE_ID,
            "path": f"processed/video_features_v1/{SAMPLE_ID}.npz",
        }) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_6C_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_and_validate_video_6c(tmp_path: Path) -> None:
    scheme_path = _scheme(tmp_path)
    splits_path = _splits(tmp_path)
    feature_root = _feature_root(tmp_path)
    rows = build_6c_rows(
        _manifest(),
        load_label_scheme(scheme_path),
        load_subject_splits(splits_path),
        load_feature_index(_feature_index(tmp_path)),
        feature_root,
    )
    assert len(rows) == 1
    assert rows[0]["label_scheme"] == VIDEO_LABEL_SCHEME_NAME
    assert rows[0]["label_id"] == "0"
    assert rows[0]["modality"] == "video"
    assert rows[0]["valid"] == "true"
    assert rows[0]["feature_shape"] == "[10, 8]"

    csv_path = tmp_path / "video.csv"
    _write_csv(csv_path, rows)
    report = validate_video_6c(
        csv_path,
        label_scheme_path=scheme_path,
        subject_splits_path=splits_path,
        feature_root=feature_root,
    )
    assert report["status"] == "PASS", report["errors"]
    assert report["checked_feature_count"] == 1


def test_pair_check_passes_for_matching_video_and_audio(tmp_path: Path) -> None:
    video_row = {
        "sample_id": SAMPLE_ID,
        "subject_id": "P01",
        "session_id": SESSION_ID,
        "split": "val",
        "label_id": "0",
        "label_class": "No task",
        "source_file": json.dumps([f"Upper_body_video_01/{SAMPLE_ID}.mp4"]),
    }
    audio_row = dict(
        video_row,
        source_file=json.dumps([f"First_person_view_audio/{SAMPLE_ID}.wav"]),
    )
    fields = list(video_row)
    video_csv = tmp_path / "video_pair.csv"
    audio_csv = tmp_path / "audio_pair.csv"
    _write_csv_with_fields(video_csv, video_row, fields)
    _write_csv_with_fields(audio_csv, audio_row, fields)
    report = check_pairs(audio_csv, video_csv)
    assert report["status"] == "PASS"
    assert report["common_samples"] == 1


def _write_csv_with_fields(path: Path, row: dict[str, str], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
