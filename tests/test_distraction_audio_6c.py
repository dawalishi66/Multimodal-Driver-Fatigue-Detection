"""Tests for the video-aligned 6-class audio metadata (fusion interface)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from driver_state.preprocessing.distraction_audio.build_audio_6c_metadata import (
    AUDIO_6C_CSV_FIELDS,
    ERROR_NO_FEATURES,
    build_6c_rows,
)
from driver_state.preprocessing.distraction_audio.check_fusion_pairs import check_fusion_pairs
from driver_state.preprocessing.distraction_audio.fusion_labels import (
    AUDIO_LABEL_SCHEME_NAME,
    AUDIO_SPLIT_VERSION,
    SIX_CLASS_NAMES,
    is_six_class_task,
    load_label_scheme,
    load_subject_splits,
    six_class_label,
)
from driver_state.preprocessing.distraction_audio.naming import parse_clip_filename
from driver_state.preprocessing.distraction_audio.validate_audio_6c import validate_audio_6c

TASKS = ["01", "03", "04", "05", "07", "08"]
# (stem_task, subject, session) -> use unique stems
CLIPS = [
    ("01", "P01", "20231111_093143"),
    ("03", "P02", "20231114_135917"),
    ("04", "P40", "20240101_102030"),
    ("05", "P01", "20231111_100000"),
    ("07", "P02", "20231114_140000"),
    ("08", "P40", "20240101_110000"),
]
SPLIT_MAP = {"P01": "val", "P02": "train", "P40": "test"}


def _stem(task: str, subject: str, session: str) -> str:
    # real stems look like 01_P01_20231111_09_31_43_12 ; reuse task+subject+date/time
    date, time = session.split("_")
    hh, mm, ss = time[:2], time[2:4], time[4:6]
    return f"{task}_{subject}_{date}_{hh}_{mm}_{ss}_12"


def _write_scheme(tmp: Path) -> Path:
    path = tmp / "label_scheme.json"
    path.write_text(json.dumps({
        "label_scheme": AUDIO_LABEL_SCHEME_NAME,
        "class_names": list(SIX_CLASS_NAMES),
        "task_to_class": {t: i for i, t in enumerate(TASKS)},
    }), encoding="utf-8")
    return path


def _write_splits(tmp: Path) -> Path:
    path = tmp / "splits.json"
    path.write_text(json.dumps({
        "split_version": AUDIO_SPLIT_VERSION,
        "splits": SPLIT_MAP,
    }), encoding="utf-8")
    return path


def _audit_rows() -> list[dict[str, str]]:
    rows = []
    for task, subject, session in CLIPS:
        stem = _stem(task, subject, session)
        rows.append({
            "sample_id": stem,
            "modality": "audio",
            "subject_id": subject,
            "session_id": session,
            "split": "",
            "source_file": f"First_person_view_audio/{stem}.wav",
            "window_index": "0", "window_start_ms": "0", "window_end_ms": "10000",
            "duration_ms": "10000", "label_class": "", "label_id": "",
            "valid": "false", "valid_ratio": "0", "mask": "", "feature_path": "",
            "feature_shape": "", "feature_dtype": "",
            "extractor_name": "dcpt_audio_qc_audit", "extractor_version": "0.1.0",
            "error": "NO_FEATURES_YET",
        })
    # include two non-6c tasks that must be filtered out
    rows.append({
        "sample_id": _stem("02", "P01", "20231111_093502"), "modality": "audio",
        "subject_id": "P01", "session_id": "20231111_093502", "split": "",
        "source_file": "First_person_view_audio/x.wav", "window_index": "0",
        "window_start_ms": "0", "window_end_ms": "10000", "duration_ms": "10000",
        "label_class": "", "label_id": "", "valid": "false", "valid_ratio": "0",
        "mask": "", "feature_path": "", "feature_shape": "", "feature_dtype": "",
        "extractor_name": "dcpt_audio_qc_audit", "extractor_version": "0.1.0",
        "error": "NO_FEATURES_YET",
    })
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIO_6C_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _feature_npz(root: Path, stem: str) -> str:
    rel = f"audio_features_v1/{stem}.npz"
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    length = 5
    np.savez(out,
             x=np.zeros((length, 8), dtype=np.float32),
             time_s=np.array([1, 3, 5, 7, 9], dtype=np.float64),
             valid_mask=np.ones(length, dtype=bool),
             support_s=np.array([[i * 2, i * 2 + 2] for i in range(length)], dtype=np.float64),
             observed_fraction=np.ones(length, dtype=np.float32))
    return rel


def _feature_index(tmp: Path) -> Path:
    path = tmp / "feature_index.jsonl"
    lines = []
    for task, subject, session in CLIPS:
        stem = _stem(task, subject, session)
        lines.append(json.dumps({
            "sample_id": stem, "modality": "audio",
            "path": f"audio_features_v1/{stem}.npz", "status": "ok"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_six_class_label_mapping() -> None:
    for i, task in enumerate(TASKS):
        assert six_class_label(int(task)) == (i, SIX_CLASS_NAMES[i])
    assert is_six_class_task(1) and not is_six_class_task(2)
    assert is_six_class_task(9) is False
    with pytest.raises(ValueError):
        six_class_label(2)


def test_load_scheme_and_splits(tmp_path: Path) -> None:
    scheme = load_label_scheme(_write_scheme(tmp_path))
    assert scheme["task_to_class"] == {t: i for i, t in enumerate(TASKS)}
    splits = load_subject_splits(_write_splits(tmp_path))
    assert splits == SPLIT_MAP


def test_builder_core(tmp_path: Path) -> None:
    scheme = json.loads(_write_scheme(tmp_path).read_text(encoding="utf-8"))
    splits = load_subject_splits(_write_splits(tmp_path))
    rows = build_6c_rows(_audit_rows(), scheme, splits)
    assert len(rows) == 6  # 8 candidates minus the two non-6c tasks
    assert all(r["label_scheme"] == AUDIO_LABEL_SCHEME_NAME for r in rows)
    # session_id is re-derived from the sample_id stem to mirror the video module.
    assert all(r["session_id"] for r in rows)
    assert rows[0]["session_id"] == "P01_20231111_0931_43"
    # source_file mirrors the video format: a single-element JSON array.
    assert json.loads(rows[0]["source_file"]) == [
        f"First_person_view_audio/{rows[0]['sample_id']}.wav"]
    for r in rows:
        task = r["sample_id"][:2]
        assert (int(r["label_id"]), r["label_class"]) == six_class_label(int(task))
        assert r["split"] == SPLIT_MAP[r["subject_id"]]
        assert r["valid"] == "false"
        assert r["error"] == ERROR_NO_FEATURES


def test_validate_6c_pass_with_features(tmp_path: Path) -> None:
    scheme_path = _write_scheme(tmp_path)
    splits_path = _write_splits(tmp_path)
    scheme = json.loads(scheme_path.read_text(encoding="utf-8"))
    splits = load_subject_splits(splits_path)
    feature_root = tmp_path / "processed"
    feature_root.mkdir()
    rows = []
    for task, subject, session in CLIPS:
        stem = _stem(task, subject, session)
        rel = _feature_npz(feature_root, stem)
        label_id, label_class = six_class_label(int(task))
        ref = parse_clip_filename(f"{stem}.wav")
        assert ref is not None
        rows.append({
            "sample_id": stem, "modality": "audio", "subject_id": subject,
            "session_id": ref.session_id, "split": SPLIT_MAP[subject],
            "source_file": f"First_person_view_audio/{stem}.wav",
            "window_index": "0", "window_start_ms": "0", "window_end_ms": "10000",
            "duration_ms": "10000", "label_class": label_class, "label_id": str(label_id),
            "label_scheme": AUDIO_LABEL_SCHEME_NAME, "valid": "true", "valid_ratio": "1",
            "mask": f"{rel}::valid_mask", "feature_path": rel,
            "feature_shape": "[5, 8]", "feature_dtype": "float32",
            "extractor_name": "panns_cnn14_16k",
            "extractor_version": "cnn14_16k_mAP0.438_v1", "error": "",
        })
    csv_path = tmp_path / "audio_6c.csv"
    _write_csv(csv_path, rows)
    report = validate_audio_6c(csv_path, label_scheme_path=scheme_path,
                               subject_splits_path=splits_path, feature_root=feature_root)
    assert report["status"] == "PASS", report["errors"]
    assert report["checked_feature_count"] == len(rows)


def test_validate_6c_fails_wrong_label(tmp_path: Path) -> None:
    scheme_path = _write_scheme(tmp_path)
    splits_path = _write_splits(tmp_path)
    feature_root = tmp_path / "processed"
    feature_root.mkdir()
    task, subject, session = CLIPS[0]
    stem = _stem(task, subject, session)
    rel = _feature_npz(feature_root, stem)
    label_id, _ = six_class_label(int(task))
    wrong_class = SIX_CLASS_NAMES[(label_id + 1) % 6]
    ref = parse_clip_filename(f"{stem}.wav")
    assert ref is not None
    row = {
        "sample_id": stem, "modality": "audio", "subject_id": subject,
        "session_id": ref.session_id, "split": SPLIT_MAP[subject],
        "source_file": f"First_person_view_audio/{stem}.wav",
        "window_index": "0", "window_start_ms": "0", "window_end_ms": "10000",
        "duration_ms": "10000", "label_class": wrong_class, "label_id": str(label_id),
        "label_scheme": AUDIO_LABEL_SCHEME_NAME, "valid": "true", "valid_ratio": "1",
        "mask": f"{rel}::valid_mask", "feature_path": rel,
        "feature_shape": "[5, 8]", "feature_dtype": "float32",
        "extractor_name": "panns_cnn14_16k",
        "extractor_version": "cnn14_16k_mAP0.438_v1", "error": "",
    }
    csv_path = tmp_path / "audio_6c_bad.csv"
    _write_csv(csv_path, [row])
    report = validate_audio_6c(csv_path, label_scheme_path=scheme_path,
                               subject_splits_path=splits_path, feature_root=feature_root)
    assert report["status"] == "FAIL"
    assert any("label_id/class must match" in e["message"] for e in report["errors"])


def test_check_fusion_pairs_pass_and_fail(tmp_path: Path) -> None:
    stems = [_stem(*c) for c in CLIPS]
    audio_rows = []
    video_rows = []
    for (task, subject, session), stem in zip(CLIPS, stems):
        label_id, label_class = six_class_label(int(task))
        ref = parse_clip_filename(f"{stem}.wav")
        assert ref is not None
        base = {
            "sample_id": stem, "subject_id": subject, "session_id": ref.session_id,
            "split": SPLIT_MAP[subject], "label_id": str(label_id),
            "label_class": label_class,
        }
        audio_rows.append(dict(base, modality="audio",
                               source_file=f'["First_person_view_audio/{stem}.wav"]'))
        video_rows.append(dict(base, modality="video",
                               source_file=f'["Upper_body_video_01/{stem}.mp4"]'))
    ok = check_fusion_pairs(audio_csv=_write_csv_tmp(tmp_path, "audio.csv", audio_rows),
                            video_csv=_write_csv_tmp(tmp_path, "video.csv", video_rows))
    assert ok["status"] == "PASS"
    assert ok["common_samples"] == 6
    assert ok["session_mismatch_count"] == 0
    assert ok["source_mismatch_count"] == 0

    video_rows[0]["label_class"] = "No task"  # may or may not be wrong; force mismatch
    task0 = stems[0][:2]
    _, correct_class = six_class_label(int(task0))
    video_rows[0]["label_class"] = SIX_CLASS_NAMES[(int(video_rows[0]["label_id"]) + 1) % 6]
    video_rows[0]["label_id"] = str((int(video_rows[0]["label_id"]) + 1) % 6)
    bad = check_fusion_pairs(audio_csv=_write_csv_tmp(tmp_path, "audio.csv", audio_rows),
                             video_csv=_write_csv_tmp(tmp_path, "video2.csv", video_rows))
    assert bad["status"] == "FAIL"
    assert bad["label_mismatch_count"] >= 1

    # A session_id drift must be caught.
    drift_rows = [dict(r) for r in video_rows]
    drift_rows[0]["session_id"] = "P01_20231111_9999_99"
    drift = check_fusion_pairs(audio_csv=_write_csv_tmp(tmp_path, "audio.csv", audio_rows),
                               video_csv=_write_csv_tmp(tmp_path, "video3.csv", drift_rows))
    assert drift["status"] == "FAIL"
    assert drift["session_mismatch_count"] == 1

    # A source_file whose basename does not equal sample_id must be caught.
    bad_src_rows = [dict(r) for r in video_rows]
    bad_src_rows[0]["source_file"] = '["Upper_body_video_01/9999_wrong_stem.mp4"]'
    bad_src = check_fusion_pairs(audio_csv=_write_csv_tmp(tmp_path, "audio.csv", audio_rows),
                                 video_csv=_write_csv_tmp(tmp_path, "video4.csv", bad_src_rows))
    assert bad_src["status"] == "FAIL"
    assert bad_src["source_mismatch_count"] >= 1


def _write_csv_tmp(tmp: Path, name: str, rows: list[dict[str, str]]) -> Path:
    path = tmp / name
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
