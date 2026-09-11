"""CAN-specific validation layered on the shared fatigue metadata validator."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from driver_state.preprocessing.fatigue_can.pipeline import (
    CAN_EXTRA_METADATA_FIELDS,
    CAN_FEATURE_COLUMNS,
)
from driver_state.validation.metadata import validate_metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_can_metadata(
    metadata: Path | str,
    *,
    feature_root: Path | str,
    min_valid_ratio: float = 0.95,
) -> dict[str, Any]:
    metadata = Path(metadata)
    feature_root = Path(feature_root)
    shared = validate_metadata(
        metadata, task="fatigue", feature_root=feature_root,
        min_valid_ratio=min_valid_ratio,
    )
    errors: list[dict[str, Any]] = []

    def error(code: str, message: str, row: int | None = None) -> None:
        errors.append({"code": code, "row": row, "message": message})

    try:
        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            missing = sorted(set(CAN_EXTRA_METADATA_FIELDS) - set(fields))
            if missing:
                error("MISSING_CAN_COLUMNS", f"Missing CAN columns: {missing}")
                rows = []
            else:
                rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        error("CAN_CSV_READ_FAILED", "Cannot read CAN metadata as UTF-8 CSV")
        rows = []

    session_starts: dict[str, list[int]] = {}
    feature_count = 0
    expected_columns = list(CAN_FEATURE_COLUMNS)
    expected_time = (np.arange(300, dtype=np.float64) + 0.5) / 10.0
    expected_support = np.column_stack((
        np.arange(300, dtype=np.float64) / 10.0,
        (np.arange(300, dtype=np.float64) + 1) / 10.0,
    ))
    for row_number, row in enumerate(rows, start=2):
        try:
            if row.get("modality") != "can":
                raise ValueError("CAN metadata rows must use modality=can")
            session_starts.setdefault(row["session_id"], []).append(int(row["window_start_ms"]))
            columns = json.loads(row["feature_columns"])
            if columns != expected_columns:
                raise ValueError("feature_columns do not match the locked CAN order")
            valid = row["valid"].strip().lower() in ("true", "1")
            if valid != (row["qc_status"] == "PASS"):
                raise ValueError("valid and qc_status disagree")
            if valid and row["qc_reason_codes"]:
                raise ValueError("valid CAN rows must have no QC reason codes")
            if row["clock_status"] == "pending_after_discontinuity" and valid:
                raise ValueError("pending clock segments cannot be valid")
            if not row["feature_path"]:
                continue
            feature_path = feature_root / Path(row["feature_path"])
            with np.load(feature_path, allow_pickle=False) as archive:
                x = archive["x"]
                time_s = archive["time_s"]
                valid_mask = archive["valid_mask"]
                support_s = archive["support_s"]
                observed = archive["observed_fraction"]
            feature_count += 1
            if x.shape != (300, 9):
                raise ValueError("CAN x must have shape [300,9]")
            if not np.allclose(time_s, expected_time, rtol=0, atol=1e-12):
                raise ValueError("CAN time_s must be the 10 Hz token centers")
            if not np.allclose(support_s, expected_support, rtol=0, atol=1e-12):
                raise ValueError("CAN support_s must be nonoverlapping 100 ms intervals")
            if np.any(np.abs(x[:, :6]) > 1 + 1e-6):
                raise ValueError("angle sin/cos features must stay within [-1,1]")
            if valid_mask.any() and not np.allclose(
                x[valid_mask, 8], np.rint(x[valid_mask, 8]), rtol=0, atol=1e-6
            ):
                raise ValueError("gear must remain discrete; linear interpolation is forbidden")
            if not math.isclose(float(row["raw_coverage_ratio"]), float(np.mean(observed)),
                                rel_tol=0, abs_tol=1e-6):
                raise ValueError("raw_coverage_ratio differs from observed_fraction mean")
            if row["feature_sha256"] != _sha256(feature_path):
                raise ValueError("feature_sha256 does not match the NPZ file")
        except (ValueError, KeyError, OSError, EOFError, json.JSONDecodeError) as exc:
            error("INVALID_CAN_ROW", str(exc) or type(exc).__name__, row_number)

    for session_id, starts in session_starts.items():
        expected = list(range(0, 2_400_000, 30_000))
        if sorted(starts) != expected:
            error("INCOMPLETE_CAN_SESSION_GRID",
                  f"{session_id} must retain all 80 candidate windows")

    return {
        "status": "FAIL" if shared["status"] != "PASS" or errors else "PASS",
        "scope": "shared_structure_plus_can_contract",
        "metadata": str(metadata),
        "feature_root": str(feature_root),
        "row_count": len(rows),
        "checked_feature_count": feature_count,
        "shared_report": shared,
        "can_errors": errors,
        "limitations": [
            "PASS does not prove cross-modal acquisition-zero synchronization.",
            "PASS does not prove train-only normalization or correct test-access history.",
        ],
    }
