"""All fixtures are synthetic and created in pytest's temporary directory."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from driver_state.constants import DCPT_CLASSES
from driver_state.schemas import REQUIRED_FIELDS
from driver_state.validation.metadata import validate_metadata


@dataclass
class SyntheticCase:
    root: Path
    task: str
    row: dict[str, str]
    arrays: dict[str, np.ndarray]

    def write(self, rows=None, omit=()):
        self.root.mkdir(exist_ok=True)
        np.savez(self.root / "synthetic.npz", **self.arrays)
        fields = [name for name in REQUIRED_FIELDS[self.task] if name not in omit]
        metadata = self.root / "synthetic_metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows if rows is not None else [self.row])
        return metadata

    def validate(self, rows=None, omit=()):
        return validate_metadata(self.write(rows, omit), task=self.task, feature_root=self.root)


def make_case(tmp_path, task):
    duration = 30.0 if task == "fatigue" else 10.0
    edges = np.linspace(0, duration, 7, dtype=np.float64)
    arrays = {
        "x": np.arange(24, dtype=np.float32).reshape(6, 4),
        "time_s": (edges[:-1] + edges[1:]) / 2,
        "valid_mask": np.ones(6, dtype=bool),
        "support_s": np.column_stack((edges[:-1], edges[1:])),
        "observed_fraction": np.ones(6, dtype=np.float32),
    }
    row = {name: "" for name in REQUIRED_FIELDS[task]}
    row.update({
        "sample_id": "SYNTHETIC_sample_0001", "modality": "video",
        "subject_id": "D" if task == "fatigue" else "P01", "session_id": "SYNTHETIC_session",
        "split": "train", "source_file": "synthetic/source.bin", "window_index": "0",
        "window_start_ms": "0", "window_end_ms": str(int(duration * 1000)),
        "duration_ms": str(int(duration * 1000)), "label_id": "1", "label_class": "medium",
        "valid": "true", "valid_ratio": "1.0", "mask": "synthetic.npz::valid_mask",
        "feature_path": "synthetic.npz", "feature_shape": "[6,4]", "feature_dtype": "float32",
        "extractor_name": "synthetic_fixture", "extractor_version": "v1", "error": "",
    })
    if task == "fatigue":
        row.update(label_start_ms="0", label_end_ms="240000", kss_score="6.5")
    else:
        row.update(label_id="0", label_class=DCPT_CLASSES[0])
    return SyntheticCase(tmp_path, task, row, arrays)


@pytest.fixture
def fatigue_case(tmp_path):
    return make_case(tmp_path, "fatigue")


@pytest.fixture
def distraction_case(tmp_path):
    return make_case(tmp_path, "distraction")
