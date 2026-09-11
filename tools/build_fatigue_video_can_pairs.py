"""Build the frozen UL-DD train/val video-CAN pairing manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from driver_state.data.fatigue_pairing import (
    PAIRING_VERSION,
    canonical_pair_id,
    compare_labels,
    normalize_video_time_arrays,
    pairing_key,
    parse_bool,
    select_complete_parents,
    validate_normalized_video_time,
)


SYNC_EVIDENCE = (
    "UL-DD paper Section 4.2 Synchronization; local IR duration versus "
    "Telemetry 60-Hz row-count audit"
)
SYNC_SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13039290/"
PAIR_FIELDS = (
    "paired_sample_id",
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
    "kss_score",
    "label_class",
    "label_id",
    "paired_valid",
    "exclude_reason",
    "complete8_eligible",
    "complete8_exclude_reason",
    "sync_status",
    "sync_evidence",
    "offset_ms",
    "hardware_clock_verified",
    "video_sample_id",
    "video_subwindow_ids",
    "video_feature_archive",
    "video_feature_member",
    "video_feature_shape",
    "video_feature_dtype",
    "video_feature_sha256",
    "video_source_time_reference",
    "video_time_transform",
    "can_sample_id",
    "can_feature_path",
    "can_feature_shape",
    "can_feature_dtype",
    "can_feature_sha256",
    "can_valid_ratio",
    "can_clock_status",
)
PARENT_FIELDS = (
    "parent_id",
    "split",
    "subject_id",
    "session_id",
    "label_start_ms",
    "label_end_ms",
    "kss_score",
    "label_class",
    "label_id",
    "window_count",
    "paired_sample_ids",
)
SYNC_FIELDS = (
    "subject_id",
    "session_id",
    "split",
    "video_member",
    "video_duration_s",
    "telemetry_source_rows",
    "telemetry_nominal_duration_s",
    "duration_difference_ms",
    "offset_ms",
    "sync_status",
    "sync_evidence",
    "hardware_clock_verified",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_zip_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    text = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mp4_duration_s(data: bytes) -> float:
    position = data.find(b"mvhd")
    if position < 0:
        raise ValueError("MP4 mvhd box not found")
    version = data[position + 4]
    if version == 0:
        timescale, duration = struct.unpack(">II", data[position + 16 : position + 24])
    elif version == 1:
        timescale = struct.unpack(">I", data[position + 24 : position + 28])[0]
        duration = struct.unpack(">Q", data[position + 28 : position + 36])[0]
    else:
        raise ValueError(f"unsupported mvhd version: {version}")
    if not timescale:
        raise ValueError("MP4 mvhd timescale is zero")
    return duration / timescale


def session_condition(session_id: str) -> str:
    return session_id.rsplit("_", 1)[-1]


def audit_timebase(
    raw_video_archive: Path,
    can_rows: list[dict[str, str]],
    *,
    duration_tolerance_ms: float,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    session_info: dict[tuple[str, str], dict[str, str]] = {}
    for row in can_rows:
        key = (row["subject_id"], row["session_id"])
        session_info.setdefault(key, row)

    output: list[dict[str, object]] = []
    status_by_session: dict[tuple[str, str], dict[str, object]] = {}
    with zipfile.ZipFile(raw_video_archive) as archive:
        names = set(archive.namelist())
        for (subject, session_id), can_row in sorted(session_info.items()):
            condition = session_condition(session_id)
            pattern = re.compile(
                rf"(?:^|/){re.escape(subject)}/{condition}/"
                rf"{re.escape(subject)}_IR_{condition}\.mp4$"
            )
            matches = sorted(name for name in names if pattern.search(name))
            if len(matches) != 1:
                status = "unverified_missing_unique_ir_video"
                record = {
                    "subject_id": subject,
                    "session_id": session_id,
                    "split": can_row["split"],
                    "video_member": "",
                    "video_duration_s": "",
                    "telemetry_source_rows": can_row["source_rows"],
                    "telemetry_nominal_duration_s": "",
                    "duration_difference_ms": "",
                    "offset_ms": "",
                    "sync_status": status,
                    "sync_evidence": SYNC_EVIDENCE,
                    "hardware_clock_verified": "false",
                }
            else:
                video_duration = mp4_duration_s(archive.read(matches[0]))
                source_rows = int(can_row["source_rows"])
                telemetry_duration = source_rows / 60.0
                difference_ms = (video_duration - telemetry_duration) * 1000.0
                if abs(difference_ms) <= duration_tolerance_ms:
                    status = "provider_documented_aligned"
                    offset_ms: int | str = 0
                else:
                    status = "unverified_duration_mismatch"
                    offset_ms = ""
                record = {
                    "subject_id": subject,
                    "session_id": session_id,
                    "split": can_row["split"],
                    "video_member": matches[0],
                    "video_duration_s": f"{video_duration:.6f}",
                    "telemetry_source_rows": source_rows,
                    "telemetry_nominal_duration_s": f"{telemetry_duration:.6f}",
                    "duration_difference_ms": f"{difference_ms:.6f}",
                    "offset_ms": offset_ms,
                    "sync_status": status,
                    "sync_evidence": SYNC_EVIDENCE,
                    "hardware_clock_verified": "false",
                }
            output.append(record)
            status_by_session[(subject, session_id)] = record
    return output, status_by_session


def validate_video_features(
    archive: zipfile.ZipFile,
    video_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    errors: list[str] = []
    feature_by_id: dict[str, dict[str, str]] = {}
    for row in feature_rows:
        sample_id = row["sample_id"]
        if sample_id in feature_by_id:
            errors.append(f"duplicate video feature index: {sample_id}")
        feature_by_id[sample_id] = row
    video_by_id = {row["sample_id"]: row for row in video_rows}
    if len(video_by_id) != len(video_rows):
        errors.append("duplicate video 30-second sample_id")
    if set(video_by_id) != set(feature_by_id):
        errors.append("video window IDs and feature index IDs differ")

    expected_keys = {"x", "time_s", "valid_mask", "support_s", "observed_fraction"}
    for sample_id, row in feature_by_id.items():
        member = row["relative_path"]
        try:
            payload = archive.read(member)
        except KeyError:
            errors.append(f"missing video feature member: {sample_id}")
            continue
        if sha256_bytes(payload) != row["feature_sha256"]:
            errors.append(f"video feature hash mismatch: {sample_id}")
            continue
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                if set(arrays.files) != expected_keys:
                    errors.append(f"video feature arrays differ: {sample_id}")
                    continue
                x = arrays["x"]
                time_s = arrays["time_s"]
                support_s = arrays["support_s"]
                valid_mask = arrays["valid_mask"]
                observed_fraction = arrays["observed_fraction"]
                if x.shape != (6, int(row["Dv"])) or x.dtype != np.float32:
                    errors.append(f"video x shape/dtype: {sample_id}")
                if (
                    valid_mask.shape != (6,)
                    or valid_mask.dtype != np.bool_
                    or not valid_mask.all()
                ):
                    errors.append(f"video valid_mask: {sample_id}")
                if (
                    observed_fraction.shape != (6,)
                    or observed_fraction.dtype != np.float32
                    or not np.allclose(observed_fraction, 1.0)
                ):
                    errors.append(f"video observed_fraction: {sample_id}")
                if not all(
                    np.isfinite(value).all()
                    for value in (x, time_s, support_s, observed_fraction)
                ):
                    errors.append(f"video non-finite values: {sample_id}")
                start_ms = int(video_by_id[sample_id]["window_start_ms"])
                normalized_time, normalized_support = normalize_video_time_arrays(
                    time_s, support_s, window_start_ms=start_ms
                )
                for error in validate_normalized_video_time(
                    normalized_time, normalized_support
                ):
                    errors.append(f"video normalized time {error}: {sample_id}")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"video feature load error {sample_id}: {type(exc).__name__}")
    return feature_by_id, errors


def validate_can_feature(can_root: Path, row: dict[str, str]) -> tuple[str, ...]:
    errors: list[str] = []
    path = can_root / row["feature_path"]
    if not path.is_file():
        return ("missing_can_feature",)
    if sha256_path(path) != row["feature_sha256"]:
        errors.append("can_feature_hash_mismatch")
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["x"].shape != (300, 9) or arrays["x"].dtype != np.float32:
                errors.append("can_x_shape_dtype")
            if not all(np.isfinite(arrays[name]).all() for name in arrays.files):
                errors.append("can_nonfinite")
            if arrays["time_s"].min() < 0 or arrays["support_s"].max() > 30:
                errors.append("can_time_not_sample_relative")
    except (OSError, ValueError, KeyError):
        errors.append("can_feature_load_error")
    return tuple(errors)


def build(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = args.dataset_root.resolve()
    video_package = args.video_package.resolve()
    can_root = args.can_root.resolve()
    raw_video_archive = args.raw_video_archive.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    output_root.mkdir(parents=True)

    can_paths = {
        "train": can_root / "manifests" / "can_train_windows_v1.csv",
        "val": can_root / "manifests" / "can_val_windows_v1.csv",
    }
    can_rows = [row for split in ("train", "val") for row in read_csv(can_paths[split])]
    if any(row["split"] == "test" for row in can_rows):
        raise ValueError("test row found in train/val CAN manifests")
    can_by_key: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in can_rows:
        key = pairing_key(row)
        if key in can_by_key:
            raise ValueError(f"duplicate CAN pairing key: {key}")
        can_by_key[key] = row

    sync_rows, sync_by_session = audit_timebase(
        raw_video_archive,
        can_rows,
        duration_tolerance_ms=args.duration_tolerance_ms,
    )
    write_csv(output_root / "reports" / "video_can_sync_audit_v1.csv", SYNC_FIELDS, sync_rows)

    with zipfile.ZipFile(video_package) as archive:
        if archive.testzip() is not None:
            raise ValueError("video handoff ZIP integrity check failed")
        video_rows = read_zip_csv(archive, "video_windows_30s_v2.csv")
        feature_rows = read_zip_csv(archive, "video_feature_index_v2.csv")
        if any(row["split"] == "test" for row in video_rows):
            raise ValueError("test row found in video train/val handoff")
        feature_by_id, feature_errors = validate_video_features(
            archive, video_rows, feature_rows
        )

    pair_rows: list[dict[str, object]] = []
    matched_can_ids: set[str] = set()
    for video in video_rows:
        key = pairing_key(video)
        can = can_by_key.get(key)
        feature = feature_by_id[video["sample_id"]]
        reasons: list[str] = []
        if not parse_bool(video["valid"]):
            reasons.append("video_invalid")
        if can is None:
            reasons.append("can_missing_or_invalid")
            sync = None
        else:
            matched_can_ids.add(can["sample_id"])
            if not parse_bool(can["valid"]):
                reasons.append("can_invalid")
            mismatches = compare_labels(video, can)
            reasons.extend(f"label_mismatch_{field}" for field in mismatches)
            sync = sync_by_session.get((video["subject_id"], video["session_id"]))
            if sync is None or sync["sync_status"] != "provider_documented_aligned":
                reasons.append("sync_unverified")
            reasons.extend(validate_can_feature(can_root, can))

        start_ms = int(video["window_start_ms"])
        label_start_ms = int(video["label_start_ms"])
        row = {
            "paired_sample_id": canonical_pair_id(video),
            "parent_id": can["parent_id"] if can else "",
            "subject_id": video["subject_id"],
            "session_id": video["session_id"],
            "split": video["split"],
            "window_index": (start_ms - label_start_ms) // 30_000,
            "window_start_ms": start_ms,
            "window_end_ms": int(video["window_end_ms"]),
            "duration_ms": int(video["window_end_ms"]) - start_ms,
            "label_start_ms": label_start_ms,
            "label_end_ms": int(video["label_end_ms"]),
            "kss_score": video["kss_score"],
            "label_class": video["label_class"],
            "label_id": video["label_id"],
            "paired_valid": "false" if reasons else "true",
            "exclude_reason": ";".join(sorted(set(reasons))),
            "complete8_eligible": "false",
            "complete8_exclude_reason": "not_evaluated",
            "sync_status": sync["sync_status"] if sync else "not_applicable_missing_can",
            "sync_evidence": sync["sync_evidence"] if sync else "CAN unavailable",
            "offset_ms": sync["offset_ms"] if sync else "",
            "hardware_clock_verified": "false",
            "video_sample_id": video["sample_id"],
            "video_subwindow_ids": video["video_subwindow_ids"],
            "video_feature_archive": video_package.relative_to(dataset_root).as_posix(),
            "video_feature_member": feature["relative_path"],
            "video_feature_shape": feature["feature_shape"],
            "video_feature_dtype": feature["feature_dtype"],
            "video_feature_sha256": feature["feature_sha256"],
            "video_source_time_reference": "session_relative",
            "video_time_transform": "subtract_window_start_ms_div_1000",
            "can_sample_id": can["sample_id"] if can else "",
            "can_feature_path": (
                (can_root.relative_to(dataset_root) / can["feature_path"]).as_posix()
                if can
                else ""
            ),
            "can_feature_shape": can["feature_shape"] if can else "",
            "can_feature_dtype": can["feature_dtype"] if can else "",
            "can_feature_sha256": can["feature_sha256"] if can else "",
            "can_valid_ratio": can["valid_ratio"] if can else "",
            "can_clock_status": can["clock_status"] if can else "",
        }
        pair_rows.append(row)

    unmatched_can = set(row["sample_id"] for row in can_rows) - matched_can_ids
    eligible_ids, parents = select_complete_parents(pair_rows)
    complete_rows = [row for row in pair_rows if row["paired_sample_id"] in eligible_ids]
    parent_id_by_sample = {
        sample_id: parent.parent_id
        for parent in parents
        for sample_id in parent.sample_ids
    }
    for row in complete_rows:
        row["parent_id"] = parent_id_by_sample[str(row["paired_sample_id"])]
    for row in pair_rows:
        if str(row["paired_sample_id"]) in eligible_ids:
            row["complete8_eligible"] = "true"
            row["complete8_exclude_reason"] = ""
        elif str(row["paired_valid"]).lower() == "true":
            row["complete8_exclude_reason"] = "incomplete_240s_parent"
        else:
            row["complete8_exclude_reason"] = "pair_invalid"

    parent_rows = [
        {
            "parent_id": parent.parent_id,
            "split": parent.split,
            "subject_id": parent.subject_id,
            "session_id": parent.session_id,
            "label_start_ms": parent.label_start_ms,
            "label_end_ms": parent.label_end_ms,
            "kss_score": parent.kss_score,
            "label_class": parent.label_class,
            "label_id": parent.label_id,
            "window_count": len(parent.sample_ids),
            "paired_sample_ids": "|".join(parent.sample_ids),
        }
        for parent in parents
    ]

    write_csv(
        output_root / "manifests" / "fatigue_video_can_pairs_all_v1.csv",
        PAIR_FIELDS,
        pair_rows,
    )
    write_csv(
        output_root
        / "manifests"
        / "fatigue_video_can_complete8_train_val_v1.csv",
        PAIR_FIELDS,
        complete_rows,
    )
    write_csv(
        output_root
        / "manifests"
        / "fatigue_video_can_parents_train_val_v1.csv",
        PARENT_FIELDS,
        parent_rows,
    )

    window_counts = Counter(str(row["split"]) for row in complete_rows)
    parent_counts = Counter(parent.split for parent in parents)
    reason_counts = Counter(
        reason
        for row in pair_rows
        for reason in str(row["exclude_reason"]).split(";")
        if reason
    )
    checks = {
        "video_feature_validation": len(feature_errors) == 0,
        "all_valid_can_windows_matched": len(unmatched_can) == 0,
        "train_windows": window_counts["train"] == 1272,
        "val_windows": window_counts["val"] == 328,
        "train_parents": parent_counts["train"] == 159,
        "val_parents": parent_counts["val"] == 41,
        "all_parents_have_8_windows": all(
            len(parent.sample_ids) == 8 for parent in parents
        ),
        "all_shared_sessions_provider_aligned": all(
            row["sync_status"] == "provider_documented_aligned" for row in sync_rows
        ),
    }
    report: dict[str, object] = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pairing_version": PAIRING_VERSION,
        "formal_result": False,
        "test_manifest_accessed": False,
        "test_media_loaded": False,
        "sync_source_url": SYNC_SOURCE_URL,
        "hardware_clock_verified": False,
        "video_source_time_reference": "session_relative",
        "model_input_time_reference": "sample_relative",
        "video_time_transform": "subtract window_start_ms / 1000 from time_s and support_s",
        "counts": {
            "video_candidates": len(video_rows),
            "valid_can_windows": len(can_rows),
            "direct_valid_pairs": sum(
                str(row["paired_valid"]).lower() == "true" for row in pair_rows
            ),
            "valid_pairs_excluded_by_complete8": sum(
                row["complete8_exclude_reason"] == "incomplete_240s_parent"
                for row in pair_rows
            ),
            "complete8_windows": dict(sorted(window_counts.items())),
            "complete8_parents": dict(sorted(parent_counts.items())),
            "sync_sessions": len(sync_rows),
        },
        "class_counts_windows": {
            f"{split}:{label}": count
            for (split, label), count in sorted(
                Counter(
                    (str(row["split"]), str(row["label_class"]))
                    for row in complete_rows
                ).items()
            )
        },
        "class_counts_parents": {
            f"{split}:{label}": count
            for (split, label), count in sorted(
                Counter((parent.split, parent.label_class) for parent in parents).items()
            )
        },
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "checks": checks,
        "errors": feature_errors,
        "input_sha256": {
            video_package.relative_to(dataset_root).as_posix(): sha256_path(video_package),
            can_paths["train"].relative_to(dataset_root).as_posix(): sha256_path(
                can_paths["train"]
            ),
            can_paths["val"].relative_to(dataset_root).as_posix(): sha256_path(
                can_paths["val"]
            ),
        },
    }
    report_path = output_root / "reports" / "fatigue_video_can_pair_validation_v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "# UL-DD Video + CAN paired data v1\n\n"
        "This derived directory does not contain raw media or copied feature arrays. "
        "It freezes train/val identity, labels, time ranges, feature references and "
        "strict complete-8 eligibility.\n\n"
        "Video source times are session-relative. Loaders must subtract "
        "`window_start_ms / 1000` from video `time_s` and `support_s` so both "
        "modalities use sample-relative 0-30 second coordinates.\n\n"
        "Synchronization is provider-documented session alignment, supported by "
        "the local duration audit. It is not hardware-clock verification. Test is "
        "not included in this version.\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--video-package", type=Path, required=True)
    result.add_argument("--can-root", type=Path, required=True)
    result.add_argument("--raw-video-archive", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--duration-tolerance-ms", type=float, default=100.0)
    return result


def main() -> int:
    try:
        report = build(parser().parse_args())
    except (FileExistsError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
