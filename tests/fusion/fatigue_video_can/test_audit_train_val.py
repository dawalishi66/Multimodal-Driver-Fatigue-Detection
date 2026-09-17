from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.fusion.fatigue_video_can.audit_train_val import audit_run


METRICS = {
    "accuracy": 1.0,
    "balanced_accuracy_supported_classes": 1.0,
    "macro_f1_fixed_classes": 1.0,
}
SUMMARY_METRICS = (
    "val_window_macro_f1",
    "val_window_balanced_accuracy",
    "val_window_accuracy",
    "val_parent_macro_f1",
    "val_parent_balanced_accuracy",
    "val_parent_accuracy",
    "val_parent_class2_recall",
    "majority_val_parent_macro_f1",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_predictions(path: Path, *, seed: int, count: int, parent: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "seed", "split", "parent_id", "subject_id", "session_id", "label_id", "prob_low", "prob_medium", "prob_high", "prediction"]
    if parent:
        fields.insert(7, "window_count")
    else:
        fields[3:3] = ["sample_id", "video_sample_id", "can_sample_id"]
        fields.insert(9, "window_index")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            label = index % 3
            probabilities = [0.0, 0.0, 0.0]
            probabilities[label] = 1.0
            row = {
                "run_id": f"run_seed{seed}", "seed": seed, "split": "val",
                "parent_id": f"parent_{index:03d}", "subject_id": "A", "session_id": "A_D",
                "label_id": label, "prob_low": probabilities[0], "prob_medium": probabilities[1],
                "prob_high": probabilities[2], "prediction": label,
            }
            if parent:
                row["window_count"] = 8
            else:
                row.update(sample_id=f"sample_{index:03d}", video_sample_id=f"video_{index:03d}", can_sample_id=f"sample_{index:03d}", window_index=index % 8)
                row["parent_id"] = f"parent_{index // 8:03d}"
            writer.writerow(row)


def _records(root: Path, excluded: set[Path]) -> list[dict]:
    return [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _hash(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path not in excluded
    ]


def _build_run(root: Path) -> Path:
    seed_results = []
    for seed in (11, 22, 33):
        seed_root = root / f"seed_{seed}"
        _write_predictions(seed_root / "predictions" / "val_windows.csv", seed=seed, count=328, parent=False)
        _write_predictions(seed_root / "parent_predictions" / "val_parents.csv", seed=seed, count=41, parent=True)
        _write_json(seed_root / "metrics.json", {"formal_result": False, "test_manifest_accessed": False, "validation": {"window_30s": METRICS, "parent_240s": METRICS}})
        run_manifest = seed_root / "run_manifest.json"
        _write_json(run_manifest, {"formal_result": False, "test_manifest_accessed": False, "files": _records(seed_root, {run_manifest})})
        seed_results.append({
            "seed": seed, "best_epoch": 1, "epochs_ran": 16,
            "val_window_macro_f1": 1.0, "val_window_balanced_accuracy": 1.0, "val_window_accuracy": 1.0,
            "val_parent_macro_f1": 1.0, "val_parent_balanced_accuracy": 1.0, "val_parent_accuracy": 1.0,
            "val_parent_class2_recall": 1.0, "majority_val_parent_macro_f1": 0.2,
            "test_manifest_accessed": False,
        })
    statistics = {
        name: {"values": [item[name] for item in seed_results], "mean": seed_results[0][name], "sample_std_ddof_1": 0.0, "seed_count": 3}
        for name in SUMMARY_METRICS
    }
    _write_json(root / "seed_summary.json", {
        "status": "PASS", "formal_result": False, "test_manifest_accessed": False,
        "model_id": "simple_fusion_v1", "experiment_id": "synthetic", "cohort": "synthetic",
        "seed_results": seed_results, "statistics": statistics,
    })
    manifest = root / "experiment_manifest.json"
    _write_json(manifest, {
        "status": "DEVELOPMENT_TRAIN_VAL_COMPLETE", "formal_result": False,
        "test_manifest_accessed": False, "completed_seeds": [11, 22, 33],
        "files": _records(root, {manifest}),
    })
    return root


def test_audit_run_recomputes_predictions_and_hashes(tmp_path: Path):
    report = audit_run(_build_run(tmp_path / "run"))
    assert report["status"] == "PASS"
    assert report["test_manifest_accessed"] is False
    assert report["parent_macro_f1_mean"] == 1.0
    assert [item["seed"] for item in report["seed_reports"]] == [11, 22, 33]


def test_audit_run_rejects_artifact_hash_drift(tmp_path: Path):
    root = _build_run(tmp_path / "run")
    path = root / "seed_11" / "predictions" / "val_windows.csv"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("sample_000", "sample_999", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_run(root)
