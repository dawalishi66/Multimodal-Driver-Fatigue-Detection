"""Build versioned 30-second UL-DD CAN features without changing raw files.

The source archive is read in place. Each output feature contains a 10 Hz token
sequence suitable for a single-modality baseline and later Video+CAN fusion.
Timestamp discontinuities are never repaired from row order: only complete
windows before the first discontinuity are eligible for release.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from driver_state.constants import (
    FATIGUE_PARENT_MS,
    FATIGUE_WINDOW_MS,
    ULDD_SPLIT_BY_SUBJECT,
    ULDD_SUBJECT_SPLITS,
    kss_label,
)
from driver_state.schemas import FATIGUE_METADATA_FIELDS

EXPECTED_TELEMETRY_HEADER = (
    "timestamp[us]",
    "raw rendering timestamp[us]",
    "raw simulation timestamp[us]",
    "raw paused simulation timestamp[us]",
    "heading[deg]",
    "pitch[deg]",
    "roll[deg]",
    "speed[m/s]",
    "rpm",
    "gear",
)
CAN_FEATURE_COLUMNS = (
    "heading_sin",
    "heading_cos",
    "pitch_sin",
    "pitch_cos",
    "roll_sin",
    "roll_cos",
    "speed_mps",
    "rpm",
    "gear",
)
CAN_EXTRA_METADATA_FIELDS = (
    "parent_id",
    "label_index",
    "session_condition",
    "source_member",
    "source_crc32",
    "source_rows",
    "raw_start_timestamp_us",
    "median_dt_ms",
    "clock_anomaly_count",
    "first_clock_anomaly_row",
    "first_clock_anomaly_time_s",
    "clock_status",
    "raw_coverage_ratio",
    "max_raw_gap_ms",
    "max_unusable_gap_ms",
    "interpolated_token_count",
    "feature_columns",
    "feature_sha256",
    "qc_status",
    "qc_reason_codes",
)
CAN_METADATA_FIELDS = (*FATIGUE_METADATA_FIELDS, *CAN_EXTRA_METADATA_FIELDS)
CAN_PARENT_FIELDS = (
    "parent_id", "subject_id", "session_id", "split", "label_start_ms",
    "label_end_ms", "kss_score", "label_class", "label_id", "window_count",
    "valid_window_count", "complete_for_240s", "qc_reason_codes",
)

TELEMETRY_PATTERN = re.compile(
    r"^(?P<subject>[A-Z])_Telemetry_(?P<condition>[AD])\.csv$"
)
LABEL_PATTERN = re.compile(r"^(?P<subject>[A-Z])_Labels_(?P<condition>[AD])\.csv$")
LABEL_ROW_PATTERN = re.compile(r"^(?P<subject>[A-Z])_(?P<condition>Alert|Drowsy)$")
CONDITION_NAME = {"A": "Alert", "D": "Drowsy"}
CONDITION_CODE = {value: key for key, value in CONDITION_NAME.items()}


@dataclass(frozen=True)
class CanConfig:
    extractor_name: str = "uldd_can_10hz_boxcar"
    extractor_version: str = "1.0.0"
    target_hz: float = 10.0
    min_raw_coverage_ratio: float = 0.95
    min_valid_ratio: float = 0.95
    max_continuous_unusable_s: float = 1.0
    max_interpolation_gap_s: float = 0.10
    clock_jump_threshold_s: float = 1.0
    smoke_window_count: int = 16

    @property
    def token_s(self) -> float:
        return 1.0 / self.target_hz

    @property
    def tokens_per_window(self) -> int:
        value = FATIGUE_WINDOW_MS / 1000 * self.target_hz
        if not float(value).is_integer():
            raise ValueError("target_hz must divide the 30-second window exactly")
        return int(value)

    @classmethod
    def from_json(cls, path: Path | str) -> "CanConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = data.get("preprocessing", data)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(f"unknown preprocessing settings: {unknown}")
        config = cls(**settings)
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("target_hz", "max_continuous_unusable_s", "max_interpolation_gap_s",
                     "clock_jump_threshold_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_raw_coverage_ratio", "min_valid_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.max_interpolation_gap_s > self.token_s + 1e-12:
            raise ValueError("max_interpolation_gap_s cannot exceed one target token")
        if self.smoke_window_count < 1:
            raise ValueError("smoke_window_count must be positive")
        _ = self.tokens_per_window


@dataclass(frozen=True)
class LabelSession:
    subject_id: str
    condition: str
    scores: tuple[float, ...]

    @property
    def session_id(self) -> str:
        return f"{self.subject_id}_{self.condition}"


@dataclass(frozen=True)
class ClockAudit:
    median_dt_s: float
    anomaly_indices: tuple[int, ...]
    trust_end_s: float
    first_anomaly_row: int | None
    first_anomaly_time_s: float | None
    anomaly_deltas_s: tuple[float, ...]

    @property
    def status(self) -> str:
        return "stable" if not self.anomaly_indices else "pending_after_discontinuity"


@dataclass(frozen=True)
class WindowFeature:
    x: np.ndarray
    time_s: np.ndarray
    valid_mask: np.ndarray
    support_s: np.ndarray
    observed_fraction: np.ndarray
    raw_coverage_ratio: float
    valid_ratio: float
    max_raw_gap_s: float
    max_unusable_gap_s: float
    interpolated_token_count: int


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_npz(path: Path, feature: WindowFeature) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        np.savez_compressed(
            stream,
            x=feature.x,
            time_s=feature.time_s,
            valid_mask=feature.valid_mask,
            support_s=feature.support_s,
            observed_fraction=feature.observed_fraction,
        )
        temporary = Path(stream.name)
    temporary.replace(path)
    return _sha256(path)


def _archive_members(archive: zipfile.ZipFile, pattern: re.Pattern[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for member in archive.namelist():
        match = pattern.match(Path(member).name)
        if not match:
            continue
        session_id = f"{match.group('subject')}_{match.group('condition')}"
        if session_id in result:
            raise ValueError(f"duplicate archive member for session {session_id}")
        result[session_id] = member
    return result


def read_label_sessions(labels_csv: Path, archive: zipfile.ZipFile) -> list[LabelSession]:
    sessions: list[LabelSession] = []
    with labels_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) != 11:
                raise ValueError(f"Labels.csv row {row_number} must contain one ID and ten KSS scores")
            match = LABEL_ROW_PATTERN.fullmatch(row[0].strip())
            if not match:
                raise ValueError(f"Labels.csv row {row_number} has an invalid session ID")
            scores = tuple(float(value) for value in row[1:])
            for score in scores:
                kss_label(score)
            sessions.append(LabelSession(
                subject_id=match.group("subject"),
                condition=CONDITION_CODE[match.group("condition")],
                scores=scores,
            ))
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError("Labels.csv contains duplicate sessions")

    label_members = _archive_members(archive, LABEL_PATTERN)
    for session in sessions:
        member = label_members.get(session.session_id)
        if not member:
            raise ValueError(f"archive label member is missing for {session.session_id}")
        archive_scores = tuple(
            float(value) for value in archive.read(member).decode("utf-8-sig").strip().split(",")
        )
        if archive_scores != session.scores:
            raise ValueError(f"Labels.csv and archive label disagree for {session.session_id}")
    return sessions


def load_telemetry(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, tuple[str, ...]]:
    with archive.open(member) as stream:
        header = tuple(stream.readline().decode("utf-8-sig").strip().split(","))
        if header != EXPECTED_TELEMETRY_HEADER:
            raise ValueError(f"unexpected telemetry header in {member}: {header}")
        data = np.loadtxt(stream, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != len(EXPECTED_TELEMETRY_HEADER):
        raise ValueError(f"telemetry data in {member} must have ten columns")
    if data.shape[0] < 2:
        raise ValueError(f"telemetry data in {member} has fewer than two rows")
    return data, header


def detect_clock_discontinuities(timestamp_us: np.ndarray, threshold_s: float = 1.0) -> ClockAudit:
    timestamp_us = np.asarray(timestamp_us, dtype=np.float64)
    if timestamp_us.ndim != 1 or timestamp_us.size < 2:
        raise ValueError("timestamp_us must be a one-dimensional array with at least two values")
    if not np.isfinite(timestamp_us).all():
        raise ValueError("timestamp_us contains NaN or Inf")
    deltas_s = np.diff(timestamp_us) / 1_000_000.0
    positive = deltas_s[deltas_s > 0]
    if positive.size == 0:
        raise ValueError("timestamp_us has no positive intervals")
    median_dt_s = float(np.median(positive))
    anomaly_indices = np.flatnonzero((deltas_s <= 0) | (deltas_s > threshold_s))
    if anomaly_indices.size:
        index = int(anomaly_indices[0])
        trust_end_s = float((timestamp_us[index] - timestamp_us[0]) / 1_000_000.0)
        first_row = index + 2
        first_time = trust_end_s
    else:
        trust_end_s = math.inf
        first_row = None
        first_time = None
    return ClockAudit(
        median_dt_s=median_dt_s,
        anomaly_indices=tuple(int(value) for value in anomaly_indices),
        trust_end_s=trust_end_s,
        first_anomaly_row=first_row,
        first_anomaly_time_s=first_time,
        anomaly_deltas_s=tuple(float(deltas_s[value]) for value in anomaly_indices),
    )


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def resample_can_window(
    timestamp_s: np.ndarray,
    signals: np.ndarray,
    *,
    window_start_s: float,
    config: CanConfig,
) -> WindowFeature:
    """Aggregate a trusted 30-second raw window to anti-aliased 10 Hz features."""
    timestamp_s = np.asarray(timestamp_s, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    if timestamp_s.ndim != 1 or signals.shape != (timestamp_s.size, 6):
        raise ValueError("signals must have shape [N,6] for heading,pitch,roll,speed,rpm,gear")
    if timestamp_s.size < 2 or not np.isfinite(timestamp_s).all():
        raise ValueError("timestamp_s must contain at least two finite values")
    if np.any(np.diff(timestamp_s) <= 0):
        raise ValueError("resampling requires a strictly increasing trusted clock segment")

    duration_s = FATIGUE_WINDOW_MS / 1000.0
    token_s = config.token_s
    token_count = config.tokens_per_window
    relative_edges = np.linspace(0.0, duration_s, token_count + 1, dtype=np.float64)
    support_s = np.column_stack((relative_edges[:-1], relative_edges[1:])).astype(np.float64)
    time_s = ((relative_edges[:-1] + relative_edges[1:]) / 2).astype(np.float64)
    x = np.zeros((token_count, len(CAN_FEATURE_COLUMNS)), dtype=np.float32)
    valid_mask = np.zeros(token_count, dtype=bool)
    observed_fraction = np.zeros(token_count, dtype=np.float32)

    required = np.isfinite(signals).all(axis=1)
    usable_times = timestamp_s[required]
    usable_signals = signals[required]
    if usable_times.size >= 2:
        positive_dt = np.diff(usable_times)
        median_dt_s = float(np.median(positive_dt[positive_dt > 0]))
    else:
        median_dt_s = token_s

    window_end_s = window_start_s + duration_s
    in_window = (usable_times >= window_start_s) & (usable_times < window_end_s)
    local_times = usable_times[in_window]
    local_signals = usable_signals[in_window]
    interpolated = 0

    for token_index in range(token_count):
        left = window_start_s + relative_edges[token_index]
        right = window_start_s + relative_edges[token_index + 1]
        start = int(np.searchsorted(local_times, left, side="left"))
        end = int(np.searchsorted(local_times, right, side="left"))
        token_times = local_times[start:end]
        token_values = local_signals[start:end]
        observed_fraction[token_index] = np.float32(
            min(1.0, token_times.size * median_dt_s / token_s)
        )

        if token_times.size:
            boundary_times = np.concatenate(([left], token_times, [right]))
            if float(np.max(np.diff(boundary_times))) <= config.max_interpolation_gap_s + 1e-9:
                angles = np.deg2rad(token_values[:, :3])
                x[token_index] = np.asarray((
                    np.mean(np.sin(angles[:, 0])), np.mean(np.cos(angles[:, 0])),
                    np.mean(np.sin(angles[:, 1])), np.mean(np.cos(angles[:, 1])),
                    np.mean(np.sin(angles[:, 2])), np.mean(np.cos(angles[:, 2])),
                    np.mean(token_values[:, 3]), np.mean(token_values[:, 4]),
                    token_values[-1, 5],
                ), dtype=np.float32)
                valid_mask[token_index] = True
            continue

        before = int(np.searchsorted(usable_times, left, side="left")) - 1
        after = before + 1
        if before < 0 or after >= usable_times.size:
            continue
        gap = float(usable_times[after] - usable_times[before])
        if gap <= config.max_interpolation_gap_s + 1e-9:
            center = window_start_s + time_s[token_index]
            weight = float((center - usable_times[before]) / gap)
            values = usable_signals[before] + weight * (usable_signals[after] - usable_signals[before])
            angles = np.deg2rad(values[:3])
            x[token_index] = np.asarray((
                math.sin(angles[0]), math.cos(angles[0]),
                math.sin(angles[1]), math.cos(angles[1]),
                math.sin(angles[2]), math.cos(angles[2]),
                values[3], values[4], usable_signals[before, 5],
            ), dtype=np.float32)
            valid_mask[token_index] = True
            interpolated += 1

    if local_times.size:
        max_raw_gap_s = float(np.max(np.diff(np.concatenate((
            [window_start_s], local_times, [window_end_s]
        )))))
    else:
        max_raw_gap_s = duration_s
    valid_ratio = float(np.mean(valid_mask))
    raw_coverage_ratio = float(np.mean(observed_fraction))
    max_unusable_gap_s = _longest_false_run(valid_mask) * token_s
    return WindowFeature(
        x=x,
        time_s=time_s,
        valid_mask=valid_mask,
        support_s=support_s,
        observed_fraction=observed_fraction,
        raw_coverage_ratio=raw_coverage_ratio,
        valid_ratio=valid_ratio,
        max_raw_gap_s=max_raw_gap_s,
        max_unusable_gap_s=max_unusable_gap_s,
        interpolated_token_count=interpolated,
    )


def _feature_reasons(feature: WindowFeature, config: CanConfig) -> list[str]:
    reasons: list[str] = []
    if feature.raw_coverage_ratio + 1e-9 < config.min_raw_coverage_ratio:
        reasons.append("RAW_COVERAGE_BELOW_95_PERCENT")
    if feature.valid_ratio + 1e-9 < config.min_valid_ratio:
        reasons.append("FEATURE_COVERAGE_BELOW_95_PERCENT")
    if feature.max_unusable_gap_s > config.max_continuous_unusable_s + 1e-9:
        reasons.append("CONTINUOUS_UNUSABLE_GAP_EXCEEDS_1S")
    return reasons


def _base_row(session: LabelSession, label_index: int, window_index: int) -> dict[str, str]:
    label_start_ms = label_index * FATIGUE_PARENT_MS
    label_end_ms = label_start_ms + FATIGUE_PARENT_MS
    start_ms = label_start_ms + window_index * FATIGUE_WINDOW_MS
    end_ms = start_ms + FATIGUE_WINDOW_MS
    score = session.scores[label_index]
    label_id, label_class = kss_label(score)
    sample_id = f"ULDD_{session.session_id}_{start_ms:09d}_{end_ms:09d}"
    return {
        "sample_id": sample_id,
        "modality": "can",
        "subject_id": session.subject_id,
        "session_id": session.session_id,
        "split": ULDD_SPLIT_BY_SUBJECT[session.subject_id],
        "window_index": str(window_index),
        "window_start_ms": str(start_ms),
        "window_end_ms": str(end_ms),
        "duration_ms": str(FATIGUE_WINDOW_MS),
        "label_start_ms": str(label_start_ms),
        "label_end_ms": str(label_end_ms),
        "kss_score": format(score, "g"),
        "label_class": label_class,
        "label_id": str(label_id),
        "parent_id": f"ULDD_{session.session_id}_{label_start_ms:09d}_{label_end_ms:09d}",
        "label_index": str(label_index),
        "session_condition": CONDITION_NAME[session.condition],
    }


def _missing_row(base: dict[str, str], source_file: str, source_member: str,
                 config: CanConfig) -> dict[str, str]:
    row = {field: "" for field in CAN_METADATA_FIELDS}
    row.update(base)
    row.update({
        "source_file": source_file,
        "source_member": source_member,
        "valid": "false",
        "valid_ratio": "0",
        "extractor_name": config.extractor_name,
        "extractor_version": config.extractor_version,
        "error": "SOURCE_TELEMETRY_MISSING",
        "clock_status": "source_missing",
        "raw_coverage_ratio": "0",
        "max_raw_gap_ms": str(FATIGUE_WINDOW_MS),
        "max_unusable_gap_ms": str(FATIGUE_WINDOW_MS),
        "interpolated_token_count": "0",
        "feature_columns": json.dumps(CAN_FEATURE_COLUMNS, separators=(",", ":")),
        "qc_status": "FAIL",
        "qc_reason_codes": "SOURCE_TELEMETRY_MISSING",
    })
    return row


def _pending_clock_row(base: dict[str, str], *, source_file: str, source_member: str,
                       info: zipfile.ZipInfo, data: np.ndarray, audit: ClockAudit,
                       config: CanConfig) -> dict[str, str]:
    row = {field: "" for field in CAN_METADATA_FIELDS}
    row.update(base)
    row.update({
        "source_file": source_file,
        "source_member": source_member,
        "source_crc32": f"{info.CRC:08x}",
        "source_rows": str(data.shape[0]),
        "raw_start_timestamp_us": format(data[0, 0], ".0f"),
        "median_dt_ms": format(audit.median_dt_s * 1000, ".6f"),
        "clock_anomaly_count": str(len(audit.anomaly_indices)),
        "first_clock_anomaly_row": str(audit.first_anomaly_row or ""),
        "first_clock_anomaly_time_s": (
            "" if audit.first_anomaly_time_s is None else format(audit.first_anomaly_time_s, ".6f")
        ),
        "clock_status": audit.status,
        "valid": "false",
        "valid_ratio": "0",
        "extractor_name": config.extractor_name,
        "extractor_version": config.extractor_version,
        "error": "CLOCK_MAPPING_PENDING_AFTER_DISCONTINUITY",
        "raw_coverage_ratio": "0",
        "max_raw_gap_ms": str(FATIGUE_WINDOW_MS),
        "max_unusable_gap_ms": str(FATIGUE_WINDOW_MS),
        "interpolated_token_count": "0",
        "feature_columns": json.dumps(CAN_FEATURE_COLUMNS, separators=(",", ":")),
        "qc_status": "PENDING",
        "qc_reason_codes": "CLOCK_MAPPING_PENDING_AFTER_DISCONTINUITY",
    })
    return row


def _trusted_arrays(data: np.ndarray, audit: ClockAudit) -> tuple[np.ndarray, np.ndarray]:
    end = audit.anomaly_indices[0] + 1 if audit.anomaly_indices else data.shape[0]
    relative_s = (data[:end, 0] - data[0, 0]) / 1_000_000.0
    return relative_s, data[:end, 4:10]


def _build_smoke(
    archive: zipfile.ZipFile,
    telemetry_members: dict[str, str],
    sessions: Sequence[LabelSession],
    config: CanConfig,
) -> dict[str, Any]:
    checked = 0
    session_ids: list[str] = []
    for session in sessions:
        member = telemetry_members.get(session.session_id)
        if not member:
            continue
        data, _ = load_telemetry(archive, member)
        audit = detect_clock_discontinuities(data[:, 0], config.clock_jump_threshold_s)
        trusted_time, trusted_signals = _trusted_arrays(data, audit)
        for label_index in range(len(session.scores)):
            for window_index in range(FATIGUE_PARENT_MS // FATIGUE_WINDOW_MS):
                start_s = (label_index * FATIGUE_PARENT_MS + window_index * FATIGUE_WINDOW_MS) / 1000
                end_s = start_s + FATIGUE_WINDOW_MS / 1000
                if end_s > audit.trust_end_s + 1e-9:
                    continue
                feature = resample_can_window(
                    trusted_time, trusted_signals, window_start_s=start_s, config=config
                )
                if feature.x.shape != (config.tokens_per_window, len(CAN_FEATURE_COLUMNS)):
                    raise AssertionError("smoke feature shape mismatch")
                if feature.x.dtype != np.float32 or feature.time_s.dtype != np.float64:
                    raise AssertionError("smoke feature dtype mismatch")
                checked += 1
                session_ids.append(session.session_id)
                if checked >= config.smoke_window_count:
                    return {"status": "PASS", "window_count": checked,
                            "session_ids": sorted(set(session_ids))}
    raise ValueError("not enough trusted windows for the configured smoke test")


def _write_metadata(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CAN_METADATA_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def _write_subject_splits(path: Path, rows: Sequence[dict[str, str]]) -> None:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)
    output: list[dict[str, str]] = []
    for split, subjects in ULDD_SUBJECT_SPLITS.items():
        for subject in subjects:
            subject_rows = by_subject.get(subject, [])
            sessions = sorted({row["session_id"] for row in subject_rows})
            output.append({
                "subject_id": subject,
                "split": split,
                "session_count": str(len(sessions)),
                "sessions": json.dumps(sessions, separators=(",", ":")),
                "window_count": str(len(subject_rows)),
                "valid_window_count": str(sum(row["valid"] == "true" for row in subject_rows)),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "subject_id", "split", "session_count", "sessions", "window_count",
            "valid_window_count",
        ))
        writer.writeheader()
        writer.writerows(output)
        temporary = Path(stream.name)
    temporary.replace(path)


def write_parent_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write one row per 240-second parent for reproducible interval evaluation."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(row)
    output: list[dict[str, str]] = []
    for parent_id, group in sorted(groups.items(), key=lambda item: (
        ("train", "val", "test").index(item[1][0]["split"]),
        item[1][0]["subject_id"], item[1][0]["session_id"],
        int(item[1][0]["label_start_ms"]),
    )):
        first = group[0]
        indices = sorted(int(row["window_index"]) for row in group)
        valid_count = sum(row["valid"] == "true" for row in group)
        complete = len(group) == 8 and indices == list(range(8)) and valid_count == 8
        reasons = sorted({
            reason
            for row in group
            for reason in row["qc_reason_codes"].split(";")
            if reason
        })
        output.append({
            "parent_id": parent_id,
            "subject_id": first["subject_id"],
            "session_id": first["session_id"],
            "split": first["split"],
            "label_start_ms": first["label_start_ms"],
            "label_end_ms": first["label_end_ms"],
            "kss_score": first["kss_score"],
            "label_class": first["label_class"],
            "label_id": first["label_id"],
            "window_count": str(len(group)),
            "valid_window_count": str(valid_count),
            "complete_for_240s": "true" if complete else "false",
            "qc_reason_codes": ";".join(reasons),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CAN_PARENT_FIELDS)
        writer.writeheader()
        writer.writerows(output)
        temporary = Path(stream.name)
    temporary.replace(path)


def _audit_summary(
    rows: Sequence[dict[str, str]],
    *,
    candidate_sessions: Sequence[LabelSession],
    telemetry_members: dict[str, str],
    excluded_sessions: Sequence[LabelSession],
    clock_audits: dict[str, ClockAudit],
    input_records: list[dict[str, Any]],
    smoke: dict[str, Any],
    output_root: Path,
    config: CanConfig,
) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ULDD_SUBJECT_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        split_counts[split] = {
            "subjects": len({row["subject_id"] for row in split_rows}),
            "sessions": len({row["session_id"] for row in split_rows}),
            "windows": len(split_rows),
            "valid_windows": sum(row["valid"] == "true" for row in split_rows),
        }
    reason_counts = Counter()
    for row in rows:
        reason_counts.update(code for code in row["qc_reason_codes"].split(";") if code)
    label_counts = Counter(row["label_class"] for row in rows)
    valid_label_counts = Counter(row["label_class"] for row in rows if row["valid"] == "true")
    anomalies = {}
    for session_id, audit in sorted(clock_audits.items()):
        if audit.anomaly_indices:
            anomalies[session_id] = {
                "count": len(audit.anomaly_indices),
                "first_csv_data_row": audit.first_anomaly_row,
                "first_time_s": audit.first_anomaly_time_s,
                "delta_s": list(audit.anomaly_deltas_s),
                "policy": "windows ending after the first discontinuity are pending",
            }
    valid_windows = sum(row["valid"] == "true" for row in rows)
    parent_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        parent_groups[row["parent_id"]].append(row)
    complete_parents = [
        group for group in parent_groups.values()
        if len(group) == 8 and all(row["valid"] == "true" for row in group)
    ]
    return {
        "status": "COMPLETED_WITH_PENDING_CLOCK_SEGMENTS" if anomalies else "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UL-DD",
        "modality": "can",
        "inputs": input_records,
        "output_root": str(output_root),
        "config": config.__dict__,
        "feature_columns": list(CAN_FEATURE_COLUMNS),
        "feature_contract": {
            "x": [config.tokens_per_window, len(CAN_FEATURE_COLUMNS)],
            "time_s": [config.tokens_per_window],
            "valid_mask": [config.tokens_per_window],
            "support_s": [config.tokens_per_window, 2],
            "observed_fraction": [config.tokens_per_window],
        },
        "smoke_test": smoke,
        "source": {
            "label_sessions": len(candidate_sessions) + len(excluded_sessions),
            "telemetry_sessions": len(telemetry_members),
            "project_candidate_sessions": len(candidate_sessions),
            "project_candidate_subjects": len({s.subject_id for s in candidate_sessions}),
            "missing_candidate_telemetry": sorted(
                s.session_id for s in candidate_sessions if s.session_id not in telemetry_members
            ),
            "excluded_unassigned_sessions": sorted(s.session_id for s in excluded_sessions),
        },
        "windows": {
            "total": len(rows),
            "valid": valid_windows,
            "invalid_or_pending": len(rows) - valid_windows,
            "valid_ratio": valid_windows / len(rows) if rows else 0,
            "by_split": split_counts,
            "label_counts_all": dict(sorted(label_counts.items())),
            "label_counts_valid": dict(sorted(valid_label_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "parents_240s": {
            "total": len(parent_groups),
            "complete": len(complete_parents),
            "by_split": {
                split: sum(group[0]["split"] == split for group in complete_parents)
                for split in ULDD_SUBJECT_SPLITS
            },
        },
        "clock_anomalies": anomalies,
        "limitations": [
            "The common zero is the first telemetry capture timestamp; cross-modal zero must still be confirmed with video metadata.",
            "Rows after the first timestamp discontinuity are kept as pending metadata and are not repaired from row order.",
            "Features are not normalized; normalization must be fitted on the training split only.",
            "A structural validator cannot prove that the test split was never used for model selection.",
        ],
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    windows = audit["windows"]
    source = audit["source"]
    lines = [
        "# UL-DD CAN 数据审计与预处理结果 v1",
        "",
        f"生成时间（UTC）：`{audit['created_utc']}`",
        "",
        "## 处理结论",
        "",
        f"- 项目固定驾驶员范围内共有 {source['project_candidate_sessions']} 个带标签 session，"
        f"其中 {len(source['missing_candidate_telemetry'])} 个缺少 CAN 源文件。",
        f"- 共建立 {windows['total']} 个 30 秒候选窗口；有效 {windows['valid']} 个，"
        f"无效或待确认 {windows['invalid_or_pending']} 个。",
        f"- 240 秒父区间共 {audit['parents_240s']['total']} 个，其中 "
        f"{audit['parents_240s']['complete']} 个具备完整 8 个有效窗口。",
        "- 所有候选行均保留；没有静默跳过缺失文件、尾部窗口或异常时钟区间。",
        "- 原始文件只读。所有输出均位于独立的 `Processed_CAN_v1` 目录。",
        "",
        "## 固定处理规则",
        "",
        "- KSS：`<4=low(0)`、`4<=KSS<7=medium(1)`、`KSS>=7=high(2)`。",
        "- 240 秒标签父区间，每个父区间 8 个 30 秒不重叠窗口。",
        "- CAN 以 100 ms 桶从约 60 Hz 聚合到 10 Hz，每个窗口输出 `[300,9]`。",
        "- 角度单位为度，分别转换为 sin/cos；速度、rpm 取桶均值；gear 取桶内最后观测值。",
        "- 时间戳回退或超过 1 秒的跳变不按行号修复。跳变后的窗口标记为 `PENDING`。",
        "- 特征未标准化；训练时只能用 train 驾驶员拟合标准化参数。",
        "",
        "## 数据范围",
        "",
        f"- 原始标签 session：{source['label_sessions']}",
        f"- 原始 Telemetry session：{source['telemetry_sessions']}",
        f"- 纳入固定驾驶员划分的 session：{source['project_candidate_sessions']}",
        f"- 缺失 CAN：{', '.join(source['missing_candidate_telemetry']) or '无'}",
        f"- 因不在固定划分而排除：{', '.join(source['excluded_unassigned_sessions']) or '无'}",
        "",
        "## 按 split 汇总",
        "",
        "| split | 驾驶员 | session | 窗口 | 有效窗口 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, values in windows["by_split"].items():
        lines.append(
            f"| {split} | {values['subjects']} | {values['sessions']} | "
            f"{values['windows']} | {values['valid_windows']} |"
        )
    lines.extend(["", "## QC 原因计数", ""])
    for reason, count in windows["reason_counts"].items():
        lines.append(f"- `{reason}`：{count}")
    if not windows["reason_counts"]:
        lines.append("- 无")
    lines.extend(["", "## 时间戳异常 session", ""])
    if audit["clock_anomalies"]:
        for session_id, values in audit["clock_anomalies"].items():
            deltas = ", ".join(format(value, ".6f") for value in values["delta_s"])
            lines.append(
                f"- `{session_id}`：首个异常约在 {values['first_time_s']:.3f}s，"
                f"异常 delta(s)=[{deltas}]。"
            )
    else:
        lines.append("- 未检测到。")
    lines.extend(["", "## 尚需团队确认", ""])
    for limitation in audit["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def build_can_dataset(
    dataset_root: Path | str,
    output_root: Path | str,
    *,
    config: CanConfig | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    config = config or CanConfig()
    config.validate()
    archive_path = dataset_root / "CSV_Files.zip"
    labels_path = dataset_root / "Labels.csv"
    info_path = dataset_root / "Info.xlsx"
    for path in (archive_path, labels_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root == dataset_root or dataset_root.is_relative_to(output_root):
        raise ValueError("output_root must be a dedicated child directory, not a raw-data ancestor")
    try:
        output_root.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("output_root must be inside the UL-DD dataset directory") from exc
    metadata_path = output_root / "metadata" / "can_windows_30s_v1.csv"
    if metadata_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing completed metadata file: {metadata_path}"
        )
    features_dir = output_root / "features"
    if features_dir.exists() and any(features_dir.iterdir()):
        raise FileExistsError(f"refusing to mix with existing feature files: {features_dir}")
    for directory in (features_dir, output_root / "metadata", output_root / "reports",
                      output_root / "manifests", output_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    input_records = [_file_record(path) for path in (archive_path, labels_path, info_path)]
    rows: list[dict[str, str]] = []
    clock_audits: dict[str, ClockAudit] = {}
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise zipfile.BadZipFile(f"CRC failure in {bad_member}")
        all_sessions = read_label_sessions(labels_path, archive)
        candidate_sessions = [s for s in all_sessions if s.subject_id in ULDD_SPLIT_BY_SUBJECT]
        excluded_sessions = [s for s in all_sessions if s.subject_id not in ULDD_SPLIT_BY_SUBJECT]
        telemetry_members = _archive_members(archive, TELEMETRY_PATTERN)
        smoke = _build_smoke(archive, telemetry_members, candidate_sessions, config)

        for session in candidate_sessions:
            member = telemetry_members.get(session.session_id)
            expected_member = (
                f"CSV_Files/CSV_Files/{session.subject_id}/{session.condition}/"
                f"{session.subject_id}_Telemetry_{session.condition}.csv"
            )
            logical_source = f"CSV_Files.zip/{member or expected_member}"
            if member is None:
                for label_index in range(len(session.scores)):
                    for window_index in range(FATIGUE_PARENT_MS // FATIGUE_WINDOW_MS):
                        rows.append(_missing_row(
                            _base_row(session, label_index, window_index),
                            logical_source, expected_member, config,
                        ))
                continue

            data, _ = load_telemetry(archive, member)
            info = archive.getinfo(member)
            clock = detect_clock_discontinuities(data[:, 0], config.clock_jump_threshold_s)
            clock_audits[session.session_id] = clock
            trusted_time, trusted_signals = _trusted_arrays(data, clock)
            for label_index in range(len(session.scores)):
                for window_index in range(FATIGUE_PARENT_MS // FATIGUE_WINDOW_MS):
                    base = _base_row(session, label_index, window_index)
                    start_s = int(base["window_start_ms"]) / 1000
                    end_s = int(base["window_end_ms"]) / 1000
                    if end_s > clock.trust_end_s + 1e-9:
                        rows.append(_pending_clock_row(
                            base, source_file=logical_source, source_member=member, info=info,
                            data=data, audit=clock, config=config,
                        ))
                        continue

                    feature = resample_can_window(
                        trusted_time, trusted_signals, window_start_s=start_s, config=config
                    )
                    reasons = _feature_reasons(feature, config)
                    feature_rel = Path("features") / f"{base['sample_id']}.npz"
                    feature_path = output_root / feature_rel
                    digest = _atomic_npz(feature_path, feature)
                    row = {field: "" for field in CAN_METADATA_FIELDS}
                    row.update(base)
                    row.update({
                        "source_file": logical_source,
                        "source_member": member,
                        "source_crc32": f"{info.CRC:08x}",
                        "source_rows": str(data.shape[0]),
                        "raw_start_timestamp_us": format(data[0, 0], ".0f"),
                        "median_dt_ms": format(clock.median_dt_s * 1000, ".6f"),
                        "clock_anomaly_count": str(len(clock.anomaly_indices)),
                        "first_clock_anomaly_row": str(clock.first_anomaly_row or ""),
                        "first_clock_anomaly_time_s": (
                            "" if clock.first_anomaly_time_s is None
                            else format(clock.first_anomaly_time_s, ".6f")
                        ),
                        "clock_status": (
                            "trusted_before_discontinuity"
                            if clock.anomaly_indices else clock.status
                        ),
                        "valid": "false" if reasons else "true",
                        "valid_ratio": format(feature.valid_ratio, ".6f"),
                        "mask": f"{feature_rel.as_posix()}::valid_mask",
                        "feature_path": feature_rel.as_posix(),
                        "feature_shape": json.dumps(list(feature.x.shape), separators=(",", ":")),
                        "feature_dtype": str(feature.x.dtype),
                        "extractor_name": config.extractor_name,
                        "extractor_version": config.extractor_version,
                        "error": ";".join(reasons),
                        "raw_coverage_ratio": format(feature.raw_coverage_ratio, ".6f"),
                        "max_raw_gap_ms": format(feature.max_raw_gap_s * 1000, ".3f"),
                        "max_unusable_gap_ms": format(feature.max_unusable_gap_s * 1000, ".3f"),
                        "interpolated_token_count": str(feature.interpolated_token_count),
                        "feature_columns": json.dumps(CAN_FEATURE_COLUMNS, separators=(",", ":")),
                        "feature_sha256": digest,
                        "qc_status": "FAIL" if reasons else "PASS",
                        "qc_reason_codes": ";".join(reasons),
                    })
                    rows.append(row)

    rows.sort(key=lambda row: (
        ("train", "val", "test").index(row["split"]),
        row["subject_id"], row["session_id"], int(row["window_start_ms"])
    ))
    _write_metadata(metadata_path, rows)
    _write_subject_splits(output_root / "metadata" / "can_subject_splits_v1.csv", rows)
    write_parent_manifest(output_root / "metadata" / "can_parents_240s_v1.csv", rows)
    audit = _audit_summary(
        rows,
        candidate_sessions=candidate_sessions,
        telemetry_members=telemetry_members,
        excluded_sessions=excluded_sessions,
        clock_audits=clock_audits,
        input_records=input_records,
        smoke=smoke,
        output_root=output_root,
        config=config,
    )
    _atomic_json(output_root / "reports" / "can_data_audit_v1.json", audit)
    _atomic_text(output_root / "reports" / "can_data_audit_v1.md", _audit_markdown(audit))
    _atomic_json(output_root / "manifests" / "can_run_manifest_v1.json", audit)
    _atomic_text(output_root / "README.md", (
        "# Processed CAN v1\n\n"
        "该目录由 UL-DD CAN 预处理工具生成。原始文件保持只读。\n\n"
        "- `metadata/`：30 秒窗口索引与固定驾驶员划分。\n"
        "- `features/`：标准 NPZ 特征，每个有效/可分析窗口一个文件。\n"
        "- `reports/`：数据审计和验证报告。\n"
        "- `manifests/`：输入哈希、参数、统计和限制。\n"
        "- `logs/`：命令运行日志。\n\n"
        "冻结划分后，训练读取 `manifests/can_train_windows_v1.csv`，调参与早停读取 "
        "`can_val_windows_v1.csv`。模型方案冻结前不要读取 `can_test_windows_v1.csv`。\n\n"
        "不得把本目录的 NPZ 特征整体提交到 GitHub。\n"
    ))
    return audit
