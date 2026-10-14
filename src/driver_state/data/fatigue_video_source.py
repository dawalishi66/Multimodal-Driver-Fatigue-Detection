"""Bind UL-DD 30-second video features to their six recorded 5-second sources."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from driver_state.constants import FATIGUE_VIDEO_SUBWINDOW_MS, FATIGUE_WINDOW_MS
from driver_state.data.fatigue_pairing import parse_bool


@dataclass(frozen=True)
class VideoSourceIndex:
    windows30: Mapping[str, Mapping[str, str]]
    windows5: Mapping[str, Mapping[str, str]]
    features: Mapping[str, Mapping[str, str]]


def _read_index(
    archive: zipfile.ZipFile, member: str, required: set[str]
) -> dict[str, dict[str, str]]:
    try:
        payload = archive.read(member).decode("utf-8-sig")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError(f"video source index is missing or invalid: {member}") from exc
    reader = csv.DictReader(io.StringIO(payload), strict=True)
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"video source index {member} lacks fields: {sorted(missing)}")
    result: dict[str, dict[str, str]] = {}
    for row in reader:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in result:
            raise ValueError(f"blank or duplicate video source ID in {member}: {sample_id!r}")
        result[sample_id] = row
    return result


def load_video_source_index(archive: zipfile.ZipFile) -> VideoSourceIndex:
    """Read the provider's three independent identity/feature tables from a ZIP."""
    identity = {
        "sample_id", "subject_id", "session_id", "split",
        "window_start_ms", "window_end_ms", "label_start_ms", "label_end_ms",
        "kss_score", "label_class", "label_id", "valid",
    }
    return VideoSourceIndex(
        windows30=_read_index(
            archive, "video_windows_30s_v2.csv", identity | {"video_subwindow_ids"}
        ),
        windows5=_read_index(
            archive, "video_windows_5s_v2.csv", identity | {"feature_path", "feature_sha256"}
        ),
        features=_read_index(
            archive,
            "video_feature_index_v2.csv",
            {"sample_id", "relative_path", "feature_shape", "feature_dtype",
             "feature_sha256", "video_subwindow_ids"},
        ),
    )


def canonical_video30_id(session_id: str, start_ms: int, end_ms: int) -> str:
    return f"video30__{session_id}__{start_ms:010d}_{end_ms:010d}"


def canonical_video5_ids(session_id: str, start_ms: int) -> tuple[str, ...]:
    def one_id(position: int) -> str:
        start_us = (start_ms + position * FATIGUE_VIDEO_SUBWINDOW_MS) * 1000
        end_us = (start_ms + (position + 1) * FATIGUE_VIDEO_SUBWINDOW_MS) * 1000
        return f"video5__{session_id}_IR_{start_us:013d}_{end_us:013d}"

    return tuple(one_id(position) for position in range(6))


def _same_value(left: str, right: str, field: str) -> bool:
    if field == "kss_score":
        return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-6))
    if field.endswith("_ms") or field == "label_id":
        return int(left) == int(right)
    return left == right


def video_source_window_errors(
    window: Mapping[str, str], feature: Mapping[str, str], index: VideoSourceIndex
) -> tuple[str, ...]:
    """Check one provider 30-second row against its ordered 5-second rows."""
    errors: list[str] = []
    sample_id = window["sample_id"]
    session = window["session_id"]
    start_ms = int(window["window_start_ms"])
    end_ms = int(window["window_end_ms"])
    expected_ids = canonical_video5_ids(session, start_ms)
    ids = tuple(window["video_subwindow_ids"].split("|"))
    if end_ms - start_ms != FATIGUE_WINDOW_MS:
        errors.append("source_30s_duration")
    if sample_id != canonical_video30_id(session, start_ms, end_ms):
        errors.append("source_30s_id")
    if not session.startswith(f"{window['subject_id']}_"):
        errors.append("source_subject_session")
    if ids != expected_ids:
        errors.append("source_5s_order_or_identity")
    if feature["sample_id"] != sample_id:
        errors.append("source_feature_sample_id")
    if feature["video_subwindow_ids"] != window["video_subwindow_ids"]:
        errors.append("source_feature_subwindow_ids")
    if feature["relative_path"] != f"features_30s/{sample_id}.npz":
        errors.append("source_feature_member")
    if feature["feature_shape"] != "[6,96]" or feature["feature_dtype"] != "float32":
        errors.append("source_feature_shape_or_dtype")

    for position, child_id in enumerate(expected_ids):
        child = index.windows5.get(child_id)
        if child is None:
            errors.append(f"source_5s_missing_{position}")
            continue
        for field in (
            "subject_id", "session_id", "split", "label_start_ms", "label_end_ms",
            "kss_score", "label_class", "label_id",
        ):
            if not _same_value(child[field], window[field], field):
                errors.append(f"source_5s_{field}_{position}")
        if (
            int(child["window_start_ms"]) != start_ms + position * FATIGUE_VIDEO_SUBWINDOW_MS
            or int(child["window_end_ms"]) != start_ms + (position + 1) * FATIGUE_VIDEO_SUBWINDOW_MS
        ):
            errors.append(f"source_5s_time_{position}")
        if not parse_bool(child["valid"]):
            errors.append(f"source_5s_invalid_{position}")
        if child["feature_path"] != feature["relative_path"]:
            errors.append(f"source_5s_feature_member_{position}")
        if child["feature_sha256"] != feature["feature_sha256"]:
            errors.append(f"source_5s_feature_hash_{position}")
    return tuple(errors)


def video_pair_binding_errors(
    pair: Mapping[str, str], index: VideoSourceIndex
) -> tuple[str, ...]:
    """Reject a pair row that points at another provider video sample."""
    errors: list[str] = []
    start_ms = int(pair["window_start_ms"])
    end_ms = int(pair["window_end_ms"])
    video_id = pair["video_sample_id"]
    if video_id != canonical_video30_id(pair["session_id"], start_ms, end_ms):
        errors.append("pair_video_sample_id")
    if tuple(pair["video_subwindow_ids"].split("|")) != canonical_video5_ids(
        pair["session_id"], start_ms
    ):
        errors.append("pair_video_5s_order_or_identity")
    window = index.windows30.get(video_id)
    feature = index.features.get(video_id)
    if window is None or feature is None:
        return (*errors, "pair_video_source_missing")
    for field in (
        "subject_id", "session_id", "split", "window_start_ms", "window_end_ms",
        "label_start_ms", "label_end_ms", "kss_score", "label_class", "label_id",
    ):
        if not _same_value(pair[field], window[field], field):
            errors.append(f"pair_video_{field}")
    if pair["video_subwindow_ids"] != window["video_subwindow_ids"]:
        errors.append("pair_video_subwindow_ids")
    if not parse_bool(window["valid"]):
        errors.append("pair_video_source_invalid")
    for pair_field, source_field in (
        ("video_feature_member", "relative_path"),
        ("video_feature_shape", "feature_shape"),
        ("video_feature_dtype", "feature_dtype"),
        ("video_feature_sha256", "feature_sha256"),
    ):
        if pair[pair_field] != feature[source_field]:
            errors.append(f"pair_{pair_field}")
    errors.extend(video_source_window_errors(window, feature, index))
    return tuple(errors)


def validate_video_subwindow_grid(
    time_s: np.ndarray, support_s: np.ndarray
) -> tuple[str, ...]:
    """Require six exact, contiguous 5-second supports in sample time."""
    step_s = FATIGUE_VIDEO_SUBWINDOW_MS / 1000.0
    expected_starts = np.arange(6, dtype=np.float64) * step_s
    expected_support = np.column_stack((expected_starts, expected_starts + step_s))
    errors: list[str] = []
    if support_s.shape != (6, 2) or not np.allclose(
        support_s, expected_support, rtol=0, atol=1e-6
    ):
        errors.append("video_support_not_contiguous_5s_grid")
    if time_s.shape != (6,) or not np.allclose(
        time_s, expected_starts + step_s / 2.0, rtol=0, atol=1e-6
    ):
        errors.append("video_time_not_5s_centers")
    return tuple(errors)
