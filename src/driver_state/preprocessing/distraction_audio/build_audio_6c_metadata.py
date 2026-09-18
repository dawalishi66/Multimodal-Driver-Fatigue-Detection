"""Build the video-aligned 6-class audio metadata CSV (fusion candidate).

Input: the 9-class raw-audit CSV produced by ``build_metadata``, the audio
6c label scheme, the audio subject splits (mirror of the video module), and an
optional feature index. Rows are filtered to the six tasks that have
upper-body video (01/03/04/05/07/08) so the sample set matches the video module
exactly; labels and splits follow the shared 6c scheme.

Column order mirrors ``distraction_video/metadata/video_windows_10s_v1.csv``:
COMMON_METADATA_FIELDS with an extra ``label_scheme`` column after ``label_id``.
Without a feature index every row is ``valid=false`` + ``error=NO_FEATURES_YET``;
with one, rows whose NPZ exists become valid and reference the feature file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping

import numpy as np

from driver_state.preprocessing.distraction_audio.fusion_labels import (
    AUDIO_LABEL_SCHEME_NAME,
    AUDIO_SPLIT_VERSION,
    load_label_scheme,
    load_subject_splits,
    six_class_label,
)
from driver_state.preprocessing.distraction_audio.naming import parse_clip_filename
from driver_state.schemas import COMMON_METADATA_FIELDS, FEATURE_DTYPES
from driver_state.validation.metadata import covered_seconds

# Mirror of distraction_video/metadata/video_windows_10s_v1.csv column order.
AUDIO_6C_CSV_FIELDS = (
    *COMMON_METADATA_FIELDS[:12],
    "label_scheme",
    *COMMON_METADATA_FIELDS[12:],
)

FEATURE_EXTRACTOR_NAME = "panns_cnn14_16k"
FEATURE_EXTRACTOR_VERSION = "cnn14_16k_mAP0.438_v1"
FEATURE_VERSION = "panns_cnn14_16k_v1"
FEATURE_LAYOUT = "audio_features_v1"  # directory name under the feature root

ERROR_NO_FEATURES = "NO_FEATURES_YET"
ERROR_FEATURE_INDEX_NOT_OK = "FEATURE_INDEX_NOT_OK"
ERROR_FEATURE_FILE_INVALID = "FEATURE_FILE_INVALID"
ERROR_FEATURE_COVERAGE_BELOW_THRESHOLD = "FEATURE_COVERAGE_BELOW_THRESHOLD"
MIN_VALID_RATIO = 0.95
VIDEO_ALIGNED_TASKS = {"01", "03", "04", "05", "07", "08"}


@dataclass(frozen=True)
class FeatureIndexEntry:
    """One validated feature-index record."""

    path: str
    status: str
    sha256: str = ""
    feature_version: str = ""
    time_steps: int | None = None
    feature_dim: int | None = None


def _relative_path(value: str, field: str = "path") -> str:
    """Return a normalized relative POSIX path or reject unsafe input."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative path")
    candidate = value.replace("\\", "/")
    windows_path = PureWindowsPath(candidate)
    if Path(candidate).is_absolute() or windows_path.drive or windows_path.root:
        raise ValueError(f"{field} must be a relative path")
    parts = Path(candidate).parts
    if ".." in parts or not parts:
        raise ValueError(f"{field} must not escape its feature root")
    return Path(candidate).as_posix()


def load_feature_index(path: str | Path | None) -> dict[str, FeatureIndexEntry]:
    """Load a feature index, rejecting duplicate IDs and malformed records."""
    if path is None:
        return {}
    index: dict[str, FeatureIndexEntry] = {}
    for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"feature index line {line_number}: sample_id must be nonempty")
        if sample_id in index:
            raise ValueError(f"feature index line {line_number}: duplicate sample_id {sample_id}")
        feature_path = _relative_path(record.get("path", ""), "feature index path")
        if not feature_path.lower().endswith(".npz"):
            raise ValueError(f"feature index line {line_number}: path must reference an NPZ file")
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"feature index line {line_number}: status must be nonempty")
        time_steps = record.get("T")
        feature_dim = record.get("D")
        if time_steps is not None and (type(time_steps) is not int or time_steps < 1):
            raise ValueError(f"feature index line {line_number}: T must be a positive integer")
        if feature_dim is not None and (type(feature_dim) is not int or feature_dim < 1):
            raise ValueError(f"feature index line {line_number}: D must be a positive integer")
        index[sample_id] = FeatureIndexEntry(
            path=feature_path,
            status=status,
            sha256=str(record.get("sha256", "")),
            feature_version=str(record.get("feature_version", "")),
            time_steps=time_steps,
            feature_dim=feature_dim,
        )
    return index


def _inspect_feature(root: Path, entry: FeatureIndexEntry,
                     sample_id: str) -> tuple[float, list[int], str]:
    """Validate one NPZ and return (coverage, shape, dtype)."""
    path = (root / entry.path).resolve()
    root_resolved = root.resolve()
    if not path.is_relative_to(root_resolved):
        raise ValueError("feature path escapes the feature root")
    if not path.is_file():
        raise ValueError("feature file does not exist")
    if entry.sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != entry.sha256.lower():
            raise ValueError("feature file hash does not match the index")
    if entry.feature_version and entry.feature_version != FEATURE_VERSION:
        raise ValueError("feature_version does not match the configured feature layout")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(FEATURE_DTYPES):
            raise ValueError("NPZ must contain exactly the five documented arrays")
        arrays = {name: archive[name] for name in FEATURE_DTYPES}
    for name, dtype in FEATURE_DTYPES.items():
        if arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"{name} dtype must be {dtype}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains NaN or Inf")
    x = arrays["x"]
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("x must have nonempty shape [T,D]")
    length = x.shape[0]
    expected_shapes = {
        "time_s": (length,),
        "valid_mask": (length,),
        "support_s": (length, 2),
        "observed_fraction": (length,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
    if entry.time_steps is not None and entry.time_steps != length:
        raise ValueError("index T does not match x.shape[0]")
    if entry.feature_dim is not None and entry.feature_dim != x.shape[1]:
        raise ValueError("index D does not match x.shape[1]")
    times = arrays["time_s"]
    supports = arrays["support_s"]
    if np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if np.any(supports[:, 0] < 0) or np.any(supports[:, 1] > 10.0 + 1e-9) \
            or np.any(supports[:, 1] <= supports[:, 0]):
        raise ValueError("support_s must be positive intervals inside the sample")
    if np.any(times < supports[:, 0]) or np.any(times >= supports[:, 1]):
        raise ValueError("time_s must lie inside its own support interval")
    fractions = arrays["observed_fraction"]
    if np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("observed_fraction must be in [0,1]")
    valid_mask = arrays["valid_mask"]
    if not np.any(valid_mask):
        raise ValueError("valid_mask must contain at least one valid token")
    coverage = covered_seconds(supports, valid_mask) / 10.0
    if not np.isfinite(coverage) or coverage < 0 or coverage > 1:
        raise ValueError("feature coverage must be in [0,1]")
    return float(coverage), list(x.shape), str(x.dtype)


def build_6c_rows(
    audit_rows: list[dict[str, str]],
    label_scheme: dict[str, object],
    subject_splits: dict[str, str],
    feature_index: Mapping[str, str | FeatureIndexEntry] | None = None,
    feature_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Turn raw-audit rows into video-aligned 6c metadata rows."""
    features = feature_index or {}
    root = Path(feature_root) if feature_root is not None else None
    rows: list[dict[str, str]] = []
    for audit in audit_rows:
        stem = audit["sample_id"]
        task_code = int(stem[:2])
        task_key = f"{task_code:02d}"
        if task_key not in VIDEO_ALIGNED_TASKS:
            continue
        label_id, label_class = six_class_label(task_code)
        subject = audit["subject_id"]
        split = subject_splits.get(subject, "")
        raw_entry = features.get(stem)
        if isinstance(raw_entry, str):
            entry = FeatureIndexEntry(path=_relative_path(raw_entry, "feature path"),
                                      status="ok")
        else:
            entry = raw_entry

        valid = "false"
        ratio = "0"
        mask = ""
        feature_rel = ""
        shape = ""
        feature_dtype = ""
        error = ERROR_NO_FEATURES
        audit_error = audit.get("error", "")
        if audit_error and audit_error != ERROR_NO_FEATURES:
            error = audit_error
        elif entry is None:
            error = ERROR_NO_FEATURES
        elif entry.status != "ok":
            error = ERROR_FEATURE_INDEX_NOT_OK
        else:
            if root is None:
                raise ValueError("feature_root is required when a feature index is provided")
            try:
                coverage, actual_shape, actual_dtype = _inspect_feature(root, entry, stem)
            except (OSError, ValueError, EOFError):
                error = ERROR_FEATURE_FILE_INVALID
            else:
                if coverage < MIN_VALID_RATIO:
                    error = ERROR_FEATURE_COVERAGE_BELOW_THRESHOLD
                else:
                    valid = "true"
                    ratio = repr(float(coverage))
                    feature_rel = entry.path
                    mask = f"{feature_rel}::valid_mask"
                    shape = json.dumps(actual_shape, separators=(",", ":"))
                    feature_dtype = actual_dtype
                    error = ""
        # session_id must be byte-identical to the distraction-video module so a
        # later fusion join works on {sample_id, subject_id, session_id}. The raw
        # audit's session_id predates the video-aligned format, so re-derive it
        # from the sample_id main identifier rather than trusting a stale column.
        ref = parse_clip_filename(audit["source_file"]) or parse_clip_filename(f"{stem}.wav")
        session_id = ref.session_id if ref is not None else audit["session_id"]
        # source_file mirrors the video format: a single-element JSON array of
        # relative paths (json.dumps without spaces so it round-trips cleanly).
        source_file = json.dumps([audit["source_file"]])
        rows.append({
            "sample_id": stem,
            "modality": "audio",
            "subject_id": subject,
            "session_id": session_id,
            "split": split,
            "source_file": source_file,
            "window_index": "0",
            "window_start_ms": "0",
            "window_end_ms": "10000",
            "duration_ms": "10000",
            "label_class": label_class,
            "label_id": str(label_id),
            "label_scheme": AUDIO_LABEL_SCHEME_NAME,
            "valid": valid,
            "valid_ratio": ratio,
            "mask": mask,
            "feature_path": feature_rel,
            "feature_shape": shape,
            "feature_dtype": feature_dtype,
            "extractor_name": FEATURE_EXTRACTOR_NAME,
            "extractor_version": FEATURE_EXTRACTOR_VERSION,
            "error": error,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build video-aligned 6-class audio metadata CSV (fusion candidate).")
    parser.add_argument("--audit-csv", required=True, help="9-class raw-audit CSV")
    parser.add_argument("--label-scheme", required=True,
                        help="audio_label_scheme_6c_v1.json (mirror of video)")
    parser.add_argument("--subject-splits", required=True,
                        help="audio_subject_splits_6c_v1.json (mirror of video)")
    parser.add_argument("--feature-index", default=None,
                        help="optional audio feature index JSONL to mark rows valid")
    parser.add_argument("--feature-root", default=None,
                        help="root directory containing paths from --feature-index")
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args(argv)
    try:
        if args.feature_index and not args.feature_root:
            raise ValueError("--feature-root is required with --feature-index")
        label_scheme = load_label_scheme(args.label_scheme)
        subject_splits = load_subject_splits(args.subject_splits)
        feature_index = load_feature_index(args.feature_index)
        with Path(args.audit_csv).open(newline="", encoding="utf-8") as stream:
            audit_rows = list(csv.DictReader(stream))
        rows = build_6c_rows(
            audit_rows,
            label_scheme,
            subject_splits,
            feature_index,
            feature_root=args.feature_root,
        )
    except (ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIO_6C_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    n_valid = sum(1 for r in rows if r["valid"] == "true")
    print(f"rows={len(rows)} valid={n_valid} missing_features={len(rows)-n_valid}")
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
