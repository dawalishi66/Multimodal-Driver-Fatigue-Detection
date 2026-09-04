import json
import subprocess
import sys

import numpy as np
import pytest

from driver_state.constants import DCPT_CLASSES
from driver_state.validation.metadata import covered_seconds, validate_metadata


def test_valid_fatigue(fatigue_case):
    report = fatigue_case.validate()
    assert report["status"] == "PASS", report
    assert report["scope"] == "structure_and_features_only"
    assert report["row_count"] == report["checked_feature_count"] == 1
    assert report["limitations"]


@pytest.mark.parametrize("subject,split", [("D", "train"), ("A", "val"), ("P", "test")])
def test_each_split(fatigue_case, subject, split):
    fatigue_case.row.update(subject_id=subject, split=split)
    assert fatigue_case.validate()["status"] == "PASS"


@pytest.mark.parametrize("change", [
    {"subject_id": "B"}, {"split": "test"}, {"split": "training"},
    {"subject_id": ""}, {"session_id": ""}, {"sample_id": ""},
    {"duration_ms": "5000"}, {"window_end_ms": "5000", "duration_ms": "5000"},
    {"window_start_ms": "5000", "window_end_ms": "35000"},
    {"window_start_ms": "240000", "window_end_ms": "270000", "window_index": "0"},
    {"label_start_ms": "30000", "label_end_ms": "270000"},
    {"label_end_ms": "300000"}, {"window_index": "1"},
    {"window_start_ms": "0.0"}, {"label_id": "2"}, {"label_class": "high"},
    {"kss_score": "nan"}, {"kss_score": "10"}, {"modality": "audio"},
    {"valid": "false"}, {"valid": "yes"}, {"valid_ratio": "nan"},
    {"valid_ratio": "1.1"}, {"valid_ratio": "0.9"}, {"error": "EXTRA_ERROR"},
    {"feature_shape": "[6,5]"}, {"feature_shape": "[6.0,4]"},
    {"feature_shape": "garbage"}, {"feature_dtype": "float64"},
    {"mask": "[1,1,1]"}, {"mask": "other.npz::valid_mask"},
    {"mask": "synthetic.npz::other"}, {"extractor_version": "latest"},
    {"feature_path": "../synthetic.npz"}, {"source_file": "../source.bin"},
])
def test_invalid_metadata_fields(fatigue_case, change):
    fatigue_case.row.update(change)
    report = fatigue_case.validate()
    assert report["status"] == "FAIL", change
    assert report["errors"]


def test_no_absolute_feature_paths(fatigue_case):
    fatigue_case.row["feature_path"] = "Q:" + "/private/synthetic.npz"
    assert fatigue_case.validate()["status"] == "FAIL"


def test_missing_column(fatigue_case):
    report = fatigue_case.validate(omit=("kss_score",))
    assert report["status"] == "FAIL"
    assert report["errors"][0]["code"] == "MISSING_COLUMNS"


def test_duplicate_sample(fatigue_case):
    report = fatigue_case.validate(rows=[fatigue_case.row, fatigue_case.row.copy()])
    assert report["status"] == "FAIL"
    assert "DUPLICATE_SAMPLE_ID" in {error["code"] for error in report["errors"]}


def test_same_subject_cross_split(distraction_case):
    second = dict(distraction_case.row, sample_id="SYNTHETIC_sample_2", split="test")
    report = distraction_case.validate(rows=[distraction_case.row, second])
    assert report["status"] == "FAIL"
    assert "SUBJECT_SPLIT_LEAKAGE" in {error["code"] for error in report["errors"]}


def test_parent_label_consistency(fatigue_case):
    second = dict(fatigue_case.row, sample_id="SYNTHETIC_sample_2", window_start_ms="30000",
                  window_end_ms="60000", window_index="1", kss_score="6")
    report = fatigue_case.validate(rows=[fatigue_case.row, second])
    assert report["status"] == "FAIL"
    assert "PARENT_LABEL_CONFLICT" in {error["code"] for error in report["errors"]}


def test_contiguous_windows_and_wrong_stride(fatigue_case):
    second = dict(fatigue_case.row, sample_id="SYNTHETIC_sample_2", window_start_ms="30000",
                  window_end_ms="60000", window_index="1")
    assert fatigue_case.validate(rows=[fatigue_case.row, second])["status"] == "PASS"
    second.update(window_start_ms="60000", window_end_ms="90000", window_index="2")
    report = fatigue_case.validate(rows=[fatigue_case.row, second])
    assert report["status"] == "FAIL"
    assert "NONSTANDARD_STRIDE" in {error["code"] for error in report["errors"]}


def test_dcpt_short_coverage_not_stretched(distraction_case):
    distraction_case.arrays["support_s"][-1, 1] = 9.96
    distraction_case.row["valid_ratio"] = "0.996"
    assert distraction_case.validate()["status"] == "PASS"


@pytest.mark.parametrize("name", ["x", "time_s", "support_s", "observed_fraction"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_arrays(fatigue_case, name, value):
    fatigue_case.arrays[name].flat[0] = value
    assert fatigue_case.validate()["status"] == "FAIL"


def test_nonfinite_invalid_position_still_fails(fatigue_case):
    fatigue_case.arrays["valid_mask"][0] = False
    fatigue_case.arrays["x"][0, 0] = np.nan
    fatigue_case.row.update(valid="false", valid_ratio=str(5 / 6), error="LOW_COVERAGE")
    assert fatigue_case.validate()["status"] == "FAIL"


@pytest.mark.parametrize("name,dtype", [("x", "float64"), ("time_s", "float32"),
    ("valid_mask", "int64"), ("support_s", "float32"), ("observed_fraction", "float64")])
def test_wrong_array_dtype(fatigue_case, name, dtype):
    fatigue_case.arrays[name] = fatigue_case.arrays[name].astype(dtype)
    assert fatigue_case.validate()["status"] == "FAIL"


@pytest.mark.parametrize("name", ["x", "time_s", "valid_mask", "support_s", "observed_fraction"])
def test_wrong_array_shape(fatigue_case, name):
    fatigue_case.arrays[name] = fatigue_case.arrays[name][:-1]
    assert fatigue_case.validate()["status"] == "FAIL"


def test_mask_ratio_recomputed(fatigue_case):
    fatigue_case.arrays["valid_mask"][2] = False
    report = fatigue_case.validate()
    assert report["status"] == "FAIL"
    fatigue_case.row.update(valid="false", valid_ratio=str(5 / 6), error="MISSING_TOKEN")
    assert fatigue_case.validate()["status"] == "PASS"


def test_all_invalid_cannot_be_valid_sample(fatigue_case):
    fatigue_case.arrays["valid_mask"][:] = False
    assert fatigue_case.validate()["status"] == "FAIL"


def test_zero_feature_is_not_missing(fatigue_case):
    fatigue_case.arrays["x"][:] = 0
    assert fatigue_case.validate()["status"] == "PASS"


def test_raw_observation_fraction_not_feature_coverage(fatigue_case):
    fatigue_case.arrays["observed_fraction"][:] = 0.99
    assert fatigue_case.validate()["status"] == "PASS"
    fatigue_case.arrays["observed_fraction"][0] = 1.1
    assert fatigue_case.validate()["status"] == "FAIL"


def test_support_union_no_double_count():
    supports = np.array([[0, 4], [2, 6], [8, 10]], dtype=np.float64)
    assert covered_seconds(supports, np.array([True, True, False])) == 6
    assert covered_seconds(supports, np.array([True, True, True])) == 8
    assert covered_seconds(supports, np.array([False, False, False])) == 0


@pytest.mark.parametrize("kind", ["outside", "negative", "empty", "time_outside", "time_unordered"])
def test_bad_time_support(fatigue_case, kind):
    if kind == "outside":
        fatigue_case.arrays["support_s"][-1, 1] = 31
    elif kind == "negative":
        fatigue_case.arrays["support_s"][0, 0] = -1
    elif kind == "empty":
        fatigue_case.arrays["support_s"][0] = [0, 0]
    elif kind == "time_outside":
        fatigue_case.arrays["time_s"][0] = 6
    else:
        fatigue_case.arrays["time_s"][1] = fatigue_case.arrays["time_s"][0]
    assert fatigue_case.validate()["status"] == "FAIL"


def test_failed_row_preserved_without_features(fatigue_case):
    fatigue_case.row.update(valid="false", valid_ratio="0", error="READ_FAILED",
                           feature_path="", mask="", feature_shape="", feature_dtype="")
    report = fatigue_case.validate()
    assert report["status"] == "PASS"
    assert report["row_count"] == 1 and report["checked_feature_count"] == 0


def test_missing_feature_file(fatigue_case):
    fatigue_case.row.update(feature_path="missing.npz", mask="missing.npz::valid_mask")
    assert fatigue_case.validate()["status"] == "FAIL"


def test_missing_array(fatigue_case):
    del fatigue_case.arrays["observed_fraction"]
    assert fatigue_case.validate()["status"] == "FAIL"


def test_corrupt_npz_returns_report(fatigue_case):
    metadata = fatigue_case.write()
    (fatigue_case.root / "synthetic.npz").write_bytes(b"not an npz file")
    report = validate_metadata(metadata, task="fatigue", feature_root=fatigue_case.root)
    assert report["status"] == "FAIL"


def test_empty_and_missing_metadata(fatigue_case):
    assert fatigue_case.validate(rows=[])["status"] == "FAIL"
    assert validate_metadata(fatigue_case.root / "absent.csv", task="fatigue",
                             feature_root=fatigue_case.root)["status"] == "FAIL"


@pytest.mark.parametrize("label_id", range(9))
def test_all_dcpt_classes(distraction_case, label_id):
    distraction_case.row.update(label_id=str(label_id), label_class=DCPT_CLASSES[label_id])
    report = distraction_case.validate()
    assert report["status"] == "PASS", report
    assert report["warnings"]  # Formal fixed subject manifest remains a later acceptance item.


@pytest.mark.parametrize("change", [
    {"label_id": "9"}, {"label_id": "-1"}, {"label_class": "no_task"},
    {"subject_id": "P41"}, {"window_index": "1"},
    {"window_end_ms": "9960", "duration_ms": "9960"},
    {"window_start_ms": "100", "window_end_ms": "10100"},
    {"duration_ms": "30000", "window_end_ms": "30000"},
])
def test_bad_dcpt_rules(distraction_case, change):
    distraction_case.row.update(change)
    assert distraction_case.validate()["status"] == "FAIL"


def test_cli_pass_and_fail(fatigue_case):
    metadata = fatigue_case.write()
    output = fatigue_case.root / "report.json"
    command = [sys.executable, "-m", "driver_state.validation.metadata", "--task", "fatigue",
               "--metadata", str(metadata), "--feature-root", str(fatigue_case.root), "--report", str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    fatigue_case.row["label_id"] = "2"
    fatigue_case.write()
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "FAIL"


def test_report_cannot_overwrite_input(fatigue_case):
    metadata = fatigue_case.write()
    original = metadata.read_bytes()
    command = [sys.executable, "-m", "driver_state.validation.metadata", "--task", "fatigue",
               "--metadata", str(metadata), "--feature-root", str(fatigue_case.root), "--report", str(metadata)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 2
    assert metadata.read_bytes() == original
