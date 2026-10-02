import csv

import numpy as np

from driver_state.preprocessing.fatigue_can.pipeline import (
    CAN_FEATURE_COLUMNS,
    CanConfig,
    detect_clock_discontinuities,
    resample_can_window,
    write_parent_manifest,
)


def synthetic_can(duration_s=30.0, hz=60.0):
    timestamp_s = np.arange(0.0, duration_s, 1 / hz, dtype=np.float64)
    signals = np.column_stack((
        np.mod(timestamp_s * 20, 360),
        np.sin(timestamp_s) * 5,
        np.cos(timestamp_s) * 3,
        10 + timestamp_s / 10,
        1200 + timestamp_s,
        np.where(timestamp_s < 15, 4, 5),
    ))
    return timestamp_s, signals


def test_clock_discontinuity_stops_trust_at_first_jump():
    timestamps_us = np.arange(10, dtype=np.float64) * 16_666
    timestamps_us[6:] += 2_000_000
    audit = detect_clock_discontinuities(timestamps_us, threshold_s=1.0)
    assert audit.anomaly_indices == (5,)
    assert audit.first_anomaly_row == 7
    assert audit.status == "pending_after_discontinuity"


def test_stable_clock_has_infinite_trust_end():
    timestamps_us = np.arange(10, dtype=np.float64) * 16_666
    audit = detect_clock_discontinuities(timestamps_us)
    assert audit.anomaly_indices == ()
    assert np.isinf(audit.trust_end_s)


def test_resample_contract_and_discrete_gear():
    timestamp_s, signals = synthetic_can()
    feature = resample_can_window(
        timestamp_s, signals, window_start_s=0.0, config=CanConfig()
    )
    assert feature.x.shape == (300, len(CAN_FEATURE_COLUMNS))
    assert feature.x.dtype == np.float32
    assert feature.time_s.dtype == np.float64
    assert feature.valid_mask.dtype == np.bool_
    assert feature.support_s.dtype == np.float64
    assert feature.observed_fraction.dtype == np.float32
    assert feature.valid_mask.all()
    assert feature.valid_ratio == 1.0
    assert np.allclose(feature.x[:, 8], np.rint(feature.x[:, 8]))
    assert set(np.unique(feature.x[:, 8])) == {4.0, 5.0}


def test_parent_manifest_requires_all_eight_valid_windows(tmp_path):
    rows = []
    for index in range(8):
        rows.append({
            "parent_id": "P0", "subject_id": "D", "session_id": "D_A",
            "split": "train", "label_start_ms": "0", "label_end_ms": "240000",
            "kss_score": "3", "label_class": "low", "label_id": "0",
            "window_index": str(index), "valid": "true",
            "qc_reason_codes": "",
        })
    rows[-1]["valid"] = "false"
    rows[-1]["qc_reason_codes"] = "SYNTHETIC_FAILURE"
    path = tmp_path / "parents.csv"
    write_parent_manifest(path, rows)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        parent = next(csv.DictReader(stream))
    assert parent["window_count"] == "8"
    assert parent["valid_window_count"] == "7"
    assert parent["complete_for_240s"] == "false"
    assert parent["qc_reason_codes"] == "SYNTHETIC_FAILURE"
