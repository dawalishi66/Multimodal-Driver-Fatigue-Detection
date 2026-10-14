"""Independently audit frozen fatigue video/CAN train/val run packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SEEDS = (11, 22, 33)
EXPECTED_WINDOWS = 328
EXPECTED_PARENTS = 41
CLASS_IDS = (0, 1, 2)
PROBABILITY_FIELDS = ("prob_low", "prob_medium", "prob_high")
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _safe_child(root: Path, relative: str) -> Path:
    value = Path(relative)
    if not relative or value.is_absolute() or ".." in value.parts:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes run root: {relative!r}") from exc
    return resolved


def _verify_file_records(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    excluded: Iterable[Path] = (),
) -> int:
    excluded_paths = {path.resolve() for path in excluded}
    recorded: set[str] = set()
    for record in records:
        relative = str(record.get("path", ""))
        if relative in recorded:
            raise ValueError(f"duplicate file record: {relative}")
        recorded.add(relative)
        path = _safe_child(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"recorded artifact is missing: {relative}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"artifact size mismatch: {relative}")
        if file_sha256(path) != str(record.get("sha256", "")):
            raise ValueError(f"artifact hash mismatch: {relative}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded_paths and path.suffix != ".tmp"
    }
    if recorded != actual:
        missing_records = sorted(actual - recorded)
        missing_files = sorted(recorded - actual)
        raise ValueError(
            "artifact inventory mismatch: "
            f"unrecorded={missing_records}, missing={missing_files}"
        )
    return len(actual)


def _classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and equal in length")
    matrix = [[0 for _ in CLASS_IDS] for _ in CLASS_IDS]
    for label, prediction in zip(labels, predictions, strict=True):
        if label not in CLASS_IDS or prediction not in CLASS_IDS:
            raise ValueError("label or prediction is outside classes 0,1,2")
        matrix[label][prediction] += 1

    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id in CLASS_IDS:
        true_positive = matrix[class_id][class_id]
        support = sum(matrix[class_id])
        predicted = sum(row[class_id] for row in matrix)
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": sum(matrix[index][index] for index in CLASS_IDS) / len(labels),
        "balanced_accuracy_supported_classes": statistics.fmean(recalls),
        "macro_f1_fixed_classes": statistics.fmean(f1_values),
        "class_2_recall": recalls[2],
        "confusion_matrix": matrix,
    }


def _read_predictions(path: Path, *, seed: int, unit: str) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        rows = list(reader)
    expected_count = EXPECTED_WINDOWS if unit == "window_30s" else EXPECTED_PARENTS
    identifier = "sample_id" if unit == "window_30s" else "parent_id"
    if len(rows) != expected_count:
        raise ValueError(f"{unit} prediction count must be {expected_count}")
    if len({row.get(identifier, "") for row in rows}) != expected_count:
        raise ValueError(f"{unit} predictions contain duplicate or empty IDs")

    labels: list[int] = []
    predictions: list[int] = []
    for row in rows:
        if row.get("split") != "val" or int(row.get("seed", -1)) != seed:
            raise ValueError(f"{unit} prediction identity does not match validation seed {seed}")
        if unit == "parent_240s" and int(row.get("window_count", -1)) != 8:
            raise ValueError("parent prediction must aggregate exactly eight windows")
        probabilities = [float(row[field]) for field in PROBABILITY_FIELDS]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError(f"{unit} contains invalid probabilities")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{unit} probabilities do not sum to one")
        prediction = max(CLASS_IDS, key=lambda index: probabilities[index])
        if prediction != int(row["prediction"]):
            raise ValueError(f"{unit} stored prediction differs from probability argmax")
        labels.append(int(row["label_id"]))
        predictions.append(prediction)
    return _classification_metrics(labels, predictions)


def _assert_close(actual: float, expected: Any, *, name: str) -> None:
    if not math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"metric mismatch for {name}: {actual} != {expected}")


def audit_run(run_root: Path | str) -> dict[str, Any]:
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")
    manifest_path = root / "experiment_manifest.json"
    summary_path = root / "seed_summary.json"
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    if manifest.get("status") != "DEVELOPMENT_TRAIN_VAL_COMPLETE":
        raise ValueError("experiment manifest is not complete")
    if summary.get("status") != "PASS":
        raise ValueError("seed summary is not PASS")
    for value in (manifest, summary):
        if value.get("formal_result") is not False or value.get("test_manifest_accessed") is not False:
            raise ValueError("run package overstates formality or accessed test")
    if tuple(manifest.get("completed_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("experiment must complete seeds 11,22,33")
    if tuple(item.get("seed") for item in summary.get("seed_results", ())) != EXPECTED_SEEDS:
        raise ValueError("seed summary order must be 11,22,33")

    experiment_files = _verify_file_records(
        root,
        manifest.get("files", ()),
        excluded={manifest_path},
    )
    seed_reports: list[dict[str, Any]] = []
    for seed_result in summary["seed_results"]:
        seed = int(seed_result["seed"])
        seed_root = root / f"seed_{seed}"
        metrics = _read_json(seed_root / "metrics.json")
        run_manifest_path = seed_root / "run_manifest.json"
        run_manifest = _read_json(run_manifest_path)
        for value in (metrics, run_manifest, seed_result):
            if value.get("test_manifest_accessed") is not False:
                raise ValueError(f"seed {seed} does not explicitly lock test access")
        if metrics.get("formal_result") is not False or run_manifest.get("formal_result") is not False:
            raise ValueError(f"seed {seed} overstates a development result")
        if any("test" in path.name.lower() for path in seed_root.rglob("*") if path.is_file()):
            raise ValueError(f"seed {seed} contains a test-named artifact")
        seed_files = _verify_file_records(
            seed_root,
            run_manifest.get("files", ()),
            excluded={run_manifest_path},
        )

        window = _read_predictions(
            seed_root / "predictions" / "val_windows.csv",
            seed=seed,
            unit="window_30s",
        )
        parent = _read_predictions(
            seed_root / "parent_predictions" / "val_parents.csv",
            seed=seed,
            unit="parent_240s",
        )
        for unit, recomputed in (("window_30s", window), ("parent_240s", parent)):
            stored = metrics.get("validation", {}).get(unit, {})
            for name in ("accuracy", "balanced_accuracy_supported_classes", "macro_f1_fixed_classes"):
                _assert_close(recomputed[name], stored.get(name), name=f"seed{seed}.{unit}.{name}")
        _assert_close(parent["macro_f1_fixed_classes"], seed_result["val_parent_macro_f1"], name=f"seed{seed}.summary.parent_macro_f1")
        _assert_close(parent["class_2_recall"], seed_result["val_parent_class2_recall"], name=f"seed{seed}.summary.class2_recall")
        seed_reports.append(
            {
                "seed": seed,
                "best_epoch": int(seed_result["best_epoch"]),
                "epochs_ran": int(seed_result["epochs_ran"]),
                "files_verified": seed_files,
                "window_metrics": window,
                "parent_metrics": parent,
            }
        )

    for metric_name in SUMMARY_METRICS:
        values = [float(item[metric_name]) for item in summary["seed_results"]]
        stored = summary.get("statistics", {}).get(metric_name, {})
        _assert_close(statistics.fmean(values), stored.get("mean"), name=f"summary.{metric_name}.mean")
        _assert_close(statistics.stdev(values), stored.get("sample_std_ddof_1"), name=f"summary.{metric_name}.std")

    return {
        "status": "PASS",
        "formal_result": False,
        "test_manifest_accessed": False,
        "model_id": summary.get("model_id"),
        "experiment_id": summary.get("experiment_id"),
        "cohort": summary.get("cohort"),
        "experiment_files_verified": experiment_files,
        "total_files": sum(1 for path in root.rglob("*") if path.is_file()),
        "total_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "experiment_manifest_sha256": file_sha256(manifest_path),
        "seed_summary_sha256": file_sha256(summary_path),
        "seed_reports": seed_reports,
        "parent_macro_f1_mean": float(summary["statistics"]["val_parent_macro_f1"]["mean"]),
        "parent_macro_f1_sample_std_ddof_1": float(
            summary["statistics"]["val_parent_macro_f1"]["sample_std_ddof_1"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite audit report: {output}")
        report = {
            "schema_version": "fatigue_video_can_train_val_audit_v1",
            "status": "PASS",
            "formal_result": False,
            "test_manifest_accessed": False,
            "code_version": args.code_version,
            "runs": [audit_run(path) for path in args.run_root],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
