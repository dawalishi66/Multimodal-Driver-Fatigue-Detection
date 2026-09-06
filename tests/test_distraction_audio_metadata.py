"""End-to-end tests for the DCPT audio audit -> metadata + QC sidecar builder."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from driver_state.preprocessing.distraction_audio.build_metadata import (
    ERROR_NO_FEATURES,
    build_audio_metadata,
)
from driver_state.schemas import COMMON_METADATA_FIELDS
from driver_state.validation.metadata import validate_metadata

from wav_utils import make_wav_bytes

CLIPS = [
    "01_P01_20231111_09_31_43_12.wav",
    "02_P01_20231111_09_35_02_12.wav",
    "03_P02_20231114_13_59_17_11.wav",
    "09_P40_20240101_10_20_30_25.wav",
]
SPLITS = {"P01": "train", "P02": "train", "P40": "test"}


def _write_clips(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in CLIPS:
        (directory / name).write_bytes(make_wav_bytes(duration_s=9.95))
    (directory / "notes.txt").write_text("not an audio clip", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_splits(path: Path) -> None:
    path.write_text(json.dumps(SPLITS), encoding="utf-8")


def test_directory_source_builds_metadata_and_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    out = tmp_path / "out"
    _write_clips(source)
    result = build_audio_metadata(audio_source=source, output_dir=out)

    rows = _read_csv(Path(result["metadata_path"]))
    assert [name for name in COMMON_METADATA_FIELDS] == list(rows[0].keys())
    assert len(rows) == len(CLIPS)

    sidecar = json.loads(Path(result["qc_path"]).read_text(encoding="utf-8"))
    agg = sidecar["aggregates"]
    assert agg["files_total"] == len(CLIPS) + 1  # + notes.txt
    assert agg["rows_total"] == len(CLIPS)
    assert agg["unparsed_total"] == 1
    assert agg["unparsed_files"] == ["notes.txt"]
    assert agg["rows_by_qc_status"] == {"pass_with_flags": len(CLIPS)}
    assert len(sidecar["clips"]) == len(CLIPS)

    first = next(row for row in rows if row["sample_id"].startswith("01_"))
    assert first["sample_id"] == "01_P01_20231111_09_31_43_12"
    assert first["subject_id"] == "P01"
    assert first["session_id"] == "20231111_093143"
    assert first["split"] == ""
    assert first["valid"] == "false"
    assert first["valid_ratio"] == "0"
    assert first["error"] == ERROR_NO_FEATURES
    assert first["label_id"] == "0"
    assert first["label_class"] == "No task"
    assert first["duration_ms"] == "10000"
    assert first["feature_path"] == ""
    assert Path(result["audit_path"]).is_file()


def test_zip_source_builds_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "First_person_view_audio.zip"
    out = tmp_path / "out"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in CLIPS:
            zf.writestr(f"First_person_view_audio/{name}", make_wav_bytes(duration_s=9.98))
    result = build_audio_metadata(audio_source=archive, output_dir=out)
    rows = _read_csv(Path(result["metadata_path"]))
    assert len(rows) == len(CLIPS)
    assert all(row["source_file"].startswith("First_person_view_audio/") for row in rows)


def test_provided_splits_allow_validator_pass(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    out = tmp_path / "out"
    _write_clips(source)
    _write_splits(tmp_path / "splits.json")
    result = build_audio_metadata(
        audio_source=source,
        output_dir=out,
        subject_splits=json.loads((tmp_path / "splits.json").read_text(encoding="utf-8")),
    )
    rows = _read_csv(Path(result["metadata_path"]))
    assert {row["split"] for row in rows} == {"train", "test"}

    report = validate_metadata(Path(result["metadata_path"]), task="distraction",
                               feature_root=out)
    assert report["status"] == "PASS"
    assert any("DCPT fixed subject manifest is not checked" in w for w in report["warnings"])
    assert report["checked_feature_count"] == 0


def test_missing_split_fails_validator(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    out = tmp_path / "out"
    _write_clips(source)
    result = build_audio_metadata(audio_source=source, output_dir=out)
    report = validate_metadata(Path(result["metadata_path"]), task="distraction",
                               feature_root=out)
    assert report["status"] == "FAIL"
    assert any("split must be train, val or test" in e["message"] for e in report["errors"])


def test_wrong_label_fails_validator(tmp_path: Path) -> None:
    source = tmp_path / "audio"
    out = tmp_path / "out"
    _write_clips(source)
    result = build_audio_metadata(
        audio_source=source, output_dir=out, subject_splits=dict(SPLITS))
    path = Path(result["metadata_path"])
    rows = _read_csv(path)
    target = next(row for row in rows if row["label_id"] == "1")
    target["label_class"] = "No task"  # Watching video -> No task mismatch
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMMON_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = validate_metadata(path, task="distraction", feature_root=out)
    assert report["status"] == "FAIL"
    assert any("label_id/class must match" in e["message"] for e in report["errors"])