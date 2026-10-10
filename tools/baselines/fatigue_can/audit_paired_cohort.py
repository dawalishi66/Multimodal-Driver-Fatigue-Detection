"""Audit whether a completed CAN train/val run used the frozen paired cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from driver_state.data.can import file_sha256
from driver_state.data.fatigue_pairing import parse_bool


SPLITS = ("train", "val")
CAN_IDENTITY_FIELDS = (
    "parent_id",
    "subject_id",
    "session_id",
    "split",
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "duration_ms",
    "label_start_ms",
    "label_end_ms",
    "label_class",
    "label_id",
)
NUMERIC_FIELDS = {
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "duration_ms",
    "label_start_ms",
    "label_end_ms",
    "label_id",
}


def _load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path.name} is missing fields: {missing}")
        return list(reader)


def _same_value(field: str, can_value: str, pair_value: str) -> bool:
    if field in NUMERIC_FIELDS:
        return int(can_value) == int(pair_value)
    if field == "kss_score":
        return bool(np.isclose(float(can_value), float(pair_value), rtol=0, atol=1e-9))
    return can_value == pair_value


def _audit_split(
    *,
    split: str,
    processed_root: Path,
    pair_rows: list[dict[str, str]],
    manifest_hashes: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    window_relative = f"manifests/can_{split}_windows_v1.csv"
    parent_relative = f"manifests/can_{split}_parents_240s_v1.csv"
    window_path = processed_root / window_relative
    parent_path = processed_root / parent_relative
    for relative, path in (
        (window_relative, window_path),
        (parent_relative, parent_path),
    ):
        expected_hash = manifest_hashes.get(relative)
        if expected_hash is None:
            errors.append(f"run manifest does not bind {relative}")
        elif not path.is_file():
            errors.append(f"missing CAN manifest: {relative}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"CAN manifest hash changed: {relative}")
    if errors:
        return {"split": split}, errors

    can_rows = _load_csv(
        window_path,
        {
            "sample_id", "parent_id", "subject_id", "session_id", "split",
            "window_index", "window_start_ms", "window_end_ms", "duration_ms",
            "label_start_ms", "label_end_ms", "kss_score", "label_class",
            "label_id", "valid", "valid_ratio", "feature_path",
            "feature_shape", "feature_dtype", "feature_sha256", "clock_status",
            "qc_status", "error",
        },
    )
    parent_rows = _load_csv(
        parent_path,
        {
            "parent_id", "split", "window_count", "valid_window_count",
            "complete_for_240s",
        },
    )
    complete_parent_ids: set[str] = set()
    for row in parent_rows:
        if row["split"] != split:
            errors.append(f"CAN parent {row['parent_id']} has wrong split")
        if (
            not parse_bool(row["complete_for_240s"])
            or int(row["window_count"]) != 8
            or int(row["valid_window_count"]) != 8
        ):
            errors.append(f"CAN parent {row['parent_id']} is not complete-8")
        if row["parent_id"] in complete_parent_ids:
            errors.append(f"duplicate CAN parent_id: {row['parent_id']}")
        complete_parent_ids.add(row["parent_id"])

    effective_can = [row for row in can_rows if row["parent_id"] in complete_parent_ids]
    selected_pairs = [row for row in pair_rows if row["split"] == split]
    can_by_id: dict[str, dict[str, str]] = {}
    for row in effective_can:
        sample_id = row["sample_id"]
        if sample_id in can_by_id:
            errors.append(f"duplicate effective CAN sample_id: {sample_id}")
        can_by_id[sample_id] = row
        if not parse_bool(row["valid"]) or row["qc_status"] != "PASS" or row["error"]:
            errors.append(f"effective CAN sample is not valid PASS: {sample_id}")

    pair_by_id: dict[str, dict[str, str]] = {}
    for row in selected_pairs:
        sample_id = row["can_sample_id"]
        if sample_id in pair_by_id:
            errors.append(f"duplicate paired CAN sample_id: {sample_id}")
        pair_by_id[sample_id] = row
        if row["paired_sample_id"] != sample_id:
            errors.append(f"paired and CAN sample IDs differ: {sample_id}")
        if not parse_bool(row["paired_valid"]) or not parse_bool(row["complete8_eligible"]):
            errors.append(f"pair is not valid complete-8: {sample_id}")

    only_can = sorted(set(can_by_id) - set(pair_by_id))
    only_pair = sorted(set(pair_by_id) - set(can_by_id))
    if only_can:
        errors.append(f"{split} has {len(only_can)} effective CAN-only sample IDs")
    if only_pair:
        errors.append(f"{split} has {len(only_pair)} pair-only CAN sample IDs")

    identity_mismatches: list[str] = []
    feature_mismatches: list[str] = []
    for sample_id in sorted(set(can_by_id) & set(pair_by_id)):
        can_row = can_by_id[sample_id]
        pair_row = pair_by_id[sample_id]
        mismatched_fields = [
            field
            for field in (*CAN_IDENTITY_FIELDS, "kss_score")
            if not _same_value(field, can_row[field], pair_row[field])
        ]
        if mismatched_fields:
            identity_mismatches.append(f"{sample_id}:{','.join(mismatched_fields)}")

        expected_pair_path = (
            PurePosixPath(processed_root.name) / PurePosixPath(can_row["feature_path"])
        ).as_posix()
        feature_equal = (
            pair_row["can_feature_path"] == expected_pair_path
            and pair_row["can_feature_shape"] == can_row["feature_shape"]
            and pair_row["can_feature_dtype"] == can_row["feature_dtype"]
            and pair_row["can_feature_sha256"] == can_row["feature_sha256"]
            and pair_row["can_clock_status"] == can_row["clock_status"]
            and np.isclose(
                float(pair_row["can_valid_ratio"]),
                float(can_row["valid_ratio"]),
                rtol=0,
                atol=1e-6,
            )
        )
        if not feature_equal:
            feature_mismatches.append(sample_id)

    if identity_mismatches:
        errors.append(f"{split} has {len(identity_mismatches)} identity/label/time mismatches")
    if feature_mismatches:
        errors.append(f"{split} has {len(feature_mismatches)} CAN feature-reference mismatches")

    pair_parent_ids = {row["parent_id"] for row in selected_pairs}
    parent_only_can = sorted(complete_parent_ids - pair_parent_ids)
    parent_only_pair = sorted(pair_parent_ids - complete_parent_ids)
    if parent_only_can or parent_only_pair:
        errors.append(f"{split} complete parent sets differ")

    return {
        "split": split,
        "can_manifest_windows": len(can_rows),
        "can_windows_excluded_by_complete8_parent_filter": len(can_rows) - len(effective_can),
        "effective_can_windows": len(effective_can),
        "paired_windows": len(selected_pairs),
        "effective_can_parents": len(complete_parent_ids),
        "paired_parents": len(pair_parent_ids),
        "sample_id_difference_count": len(only_can) + len(only_pair),
        "parent_id_difference_count": len(parent_only_can) + len(parent_only_pair),
        "identity_label_time_mismatch_count": len(identity_mismatches),
        "feature_reference_mismatch_count": len(feature_mismatches),
        "mismatch_examples": (identity_mismatches + feature_mismatches)[:10],
    }, errors


def audit_paired_cohort(
    *,
    processed_root: Path,
    pair_manifest: Path,
    run_dir: Path,
    code_version: str,
) -> dict[str, Any]:
    experiment_path = run_dir / "experiment_manifest.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if experiment.get("test_manifest_accessed") is not False:
        errors.append("CAN experiment does not prove test_manifest_accessed=false")
    if experiment.get("completed_seeds") != [11, 22, 33]:
        errors.append("CAN experiment did not complete the frozen three seeds")
    manifest_hashes = experiment.get("manifest_hashes")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("CAN experiment manifest_hashes must be a mapping")

    pair_rows = _load_csv(
        pair_manifest,
        {
            "paired_sample_id", "parent_id", "subject_id", "session_id", "split",
            "window_index", "window_start_ms", "window_end_ms", "duration_ms",
            "label_start_ms", "label_end_ms", "kss_score", "label_class",
            "label_id", "paired_valid", "complete8_eligible", "can_sample_id",
            "can_feature_path", "can_feature_shape", "can_feature_dtype",
            "can_feature_sha256", "can_valid_ratio", "can_clock_status",
        },
    )
    if any(row["split"] == "test" for row in pair_rows):
        errors.append("train/val pair manifest unexpectedly contains test rows")
    unexpected_splits = sorted({row["split"] for row in pair_rows} - set(SPLITS))
    if unexpected_splits:
        errors.append(f"pair manifest contains unexpected splits: {unexpected_splits}")

    split_reports: dict[str, Any] = {}
    for split in SPLITS:
        split_report, split_errors = _audit_split(
            split=split,
            processed_root=processed_root,
            pair_rows=pair_rows,
            manifest_hashes=manifest_hashes,
        )
        split_reports[split] = split_report
        errors.extend(split_errors)

    equivalent = not errors
    return {
        "schema_version": "can_paired_cohort_equivalence_v1",
        "status": "PASS" if equivalent else "FAIL",
        "formal_result": False,
        "test_manifest_accessed": False,
        "equivalent_for_train_val_fair_comparison": equivalent,
        "can_training_was_rerun": False,
        "historical_run_manifest_unchanged": True,
        "code_version": code_version,
        "can_experiment_id": experiment.get("experiment_id"),
        "can_experiment_code_commit": experiment.get("git", {}).get("commit"),
        "pair_manifest": pair_manifest.name,
        "pair_manifest_sha256": file_sha256(pair_manifest),
        "experiment_manifest_sha256": file_sha256(experiment_path),
        "comparison_rule": "CAN windows filtered by the run-bound complete-8 parent manifests",
        "splits": split_reports,
        "errors": errors,
        "limitations": [
            "This audit establishes train/val cohort equivalence only.",
            "It does not unlock or evaluate test.",
            "It does not validate the provenance of video feature extractor weights.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-version", default="uncommitted")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_paired_cohort(
            processed_root=args.processed_root,
            pair_manifest=args.pair_manifest,
            run_dir=args.run_dir,
            code_version=args.code_version,
        )
    except Exception as exc:
        report = {
            "schema_version": "can_paired_cohort_equivalence_v1",
            "status": "FAIL",
            "formal_result": False,
            "test_manifest_accessed": False,
            "equivalent_for_train_val_fair_comparison": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
