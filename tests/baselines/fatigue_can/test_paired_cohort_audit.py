from __future__ import annotations

import csv
import json
from pathlib import Path

from driver_state.data.can import file_sha256
from tools.baselines.fatigue_can.audit_paired_cohort import audit_paired_cohort


CAN_FIELDS = [
    "sample_id", "parent_id", "subject_id", "session_id", "split",
    "window_index", "window_start_ms", "window_end_ms", "duration_ms",
    "label_start_ms", "label_end_ms", "kss_score", "label_class", "label_id",
    "valid", "valid_ratio", "feature_path", "feature_shape", "feature_dtype",
    "feature_sha256", "clock_status", "qc_status", "error",
]
PARENT_FIELDS = [
    "parent_id", "split", "window_count", "valid_window_count",
    "complete_for_240s",
]
PAIR_FIELDS = [
    "paired_sample_id", "parent_id", "subject_id", "session_id", "split",
    "window_index", "window_start_ms", "window_end_ms", "duration_ms",
    "label_start_ms", "label_end_ms", "kss_score", "label_class", "label_id",
    "paired_valid", "complete8_eligible", "can_sample_id", "can_feature_path",
    "can_feature_shape", "can_feature_dtype", "can_feature_sha256",
    "can_valid_ratio", "can_clock_status",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(root: Path) -> tuple[Path, Path, Path]:
    processed = root / "Processed_CAN_v1"
    pair_manifest = root / "pairs.csv"
    run_dir = processed / "run"
    manifest_hashes: dict[str, str] = {}
    all_pairs: list[dict[str, str]] = []
    for split, subject in (("train", "D"), ("val", "A")):
        session = f"{subject}_A"
        parent_id = f"ULDD_{subject}_A_000000000_000240000"
        can_rows: list[dict[str, str]] = []
        for index in range(8):
            start = index * 30_000
            end = start + 30_000
            sample_id = f"ULDD_{subject}_A_{start:09d}_{end:09d}"
            can_row = {
                "sample_id": sample_id,
                "parent_id": parent_id,
                "subject_id": subject,
                "session_id": session,
                "split": split,
                "window_index": str(index),
                "window_start_ms": str(start),
                "window_end_ms": str(end),
                "duration_ms": "30000",
                "label_start_ms": "0",
                "label_end_ms": "240000",
                "kss_score": "3",
                "label_class": "low",
                "label_id": "0",
                "valid": "true",
                "valid_ratio": "1.000000",
                "feature_path": f"features/{sample_id}.npz",
                "feature_shape": "[300,9]",
                "feature_dtype": "float32",
                "feature_sha256": f"{index + 1:064x}",
                "clock_status": "stable",
                "qc_status": "PASS",
                "error": "",
            }
            can_rows.append(can_row)
            all_pairs.append(
                {
                    "paired_sample_id": sample_id,
                    "parent_id": parent_id,
                    "subject_id": subject,
                    "session_id": session,
                    "split": split,
                    "window_index": str(index),
                    "window_start_ms": str(start),
                    "window_end_ms": str(end),
                    "duration_ms": "30000",
                    "label_start_ms": "0",
                    "label_end_ms": "240000",
                    "kss_score": "3.0",
                    "label_class": "low",
                    "label_id": "0",
                    "paired_valid": "true",
                    "complete8_eligible": "true",
                    "can_sample_id": sample_id,
                    "can_feature_path": f"Processed_CAN_v1/features/{sample_id}.npz",
                    "can_feature_shape": "[300,9]",
                    "can_feature_dtype": "float32",
                    "can_feature_sha256": f"{index + 1:064x}",
                    "can_valid_ratio": "1.000000",
                    "can_clock_status": "stable",
                }
            )
        if split == "train":
            extra = dict(can_rows[0])
            extra["sample_id"] = "ULDD_D_A_000240000_000270000"
            extra["parent_id"] = "ULDD_D_A_000240000_000480000"
            extra["window_start_ms"] = "240000"
            extra["window_end_ms"] = "270000"
            extra["feature_path"] = "features/extra.npz"
            can_rows.append(extra)
        window_relative = f"manifests/can_{split}_windows_v1.csv"
        parent_relative = f"manifests/can_{split}_parents_240s_v1.csv"
        _write_csv(processed / window_relative, CAN_FIELDS, can_rows)
        _write_csv(
            processed / parent_relative,
            PARENT_FIELDS,
            [{
                "parent_id": parent_id,
                "split": split,
                "window_count": "8",
                "valid_window_count": "8",
                "complete_for_240s": "true",
            }],
        )
        manifest_hashes[window_relative] = file_sha256(processed / window_relative)
        manifest_hashes[parent_relative] = file_sha256(processed / parent_relative)

    _write_csv(pair_manifest, PAIR_FIELDS, all_pairs)
    run_dir.mkdir(parents=True)
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps({
            "experiment_id": "synthetic",
            "test_manifest_accessed": False,
            "completed_seeds": [11, 22, 33],
            "manifest_hashes": manifest_hashes,
            "git": {"commit": "abc123"},
        }),
        encoding="utf-8",
    )
    return processed, pair_manifest, run_dir


def test_exact_effective_cohort_passes_without_retraining(tmp_path: Path):
    processed, pair_manifest, run_dir = _build_fixture(tmp_path)

    report = audit_paired_cohort(
        processed_root=processed,
        pair_manifest=pair_manifest,
        run_dir=run_dir,
        code_version="test",
    )

    assert report["status"] == "PASS"
    assert report["equivalent_for_train_val_fair_comparison"] is True
    assert report["can_training_was_rerun"] is False
    assert report["splits"]["train"]["effective_can_windows"] == 8
    assert report["splits"]["train"]["can_windows_excluded_by_complete8_parent_filter"] == 1


def test_label_mismatch_fails(tmp_path: Path):
    processed, pair_manifest, run_dir = _build_fixture(tmp_path)
    rows = list(csv.DictReader(pair_manifest.open(encoding="utf-8")))
    rows[0]["label_id"] = "1"
    _write_csv(pair_manifest, PAIR_FIELDS, rows)

    report = audit_paired_cohort(
        processed_root=processed,
        pair_manifest=pair_manifest,
        run_dir=run_dir,
        code_version="test",
    )

    assert report["status"] == "FAIL"
    assert report["splits"]["train"]["identity_label_time_mismatch_count"] == 1


def test_run_bound_manifest_hash_change_fails(tmp_path: Path):
    processed, pair_manifest, run_dir = _build_fixture(tmp_path)
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text())
    manifest["manifest_hashes"]["manifests/can_train_windows_v1.csv"] = "0" * 64
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest))

    report = audit_paired_cohort(
        processed_root=processed,
        pair_manifest=pair_manifest,
        run_dir=run_dir,
        code_version="test",
    )

    assert report["status"] == "FAIL"
    assert any("hash changed" in error for error in report["errors"])


def test_test_row_in_pair_manifest_fails(tmp_path: Path):
    processed, pair_manifest, run_dir = _build_fixture(tmp_path)
    rows = list(csv.DictReader(pair_manifest.open(encoding="utf-8")))
    extra = dict(rows[0])
    extra["split"] = "test"
    extra["paired_sample_id"] = "test-row"
    extra["can_sample_id"] = "test-row"
    _write_csv(pair_manifest, PAIR_FIELDS, [*rows, extra])

    report = audit_paired_cohort(
        processed_root=processed,
        pair_manifest=pair_manifest,
        run_dir=run_dir,
        code_version="test",
    )

    assert report["status"] == "FAIL"
    assert any("test rows" in error for error in report["errors"])
