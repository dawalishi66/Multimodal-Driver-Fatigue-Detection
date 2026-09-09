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
import json
import sys
from pathlib import Path

from driver_state.preprocessing.distraction_audio.fusion_labels import (
    AUDIO_LABEL_SCHEME_NAME,
    AUDIO_SPLIT_VERSION,
    load_label_scheme,
    load_subject_splits,
    six_class_label,
)
from driver_state.preprocessing.distraction_audio.naming import parse_clip_filename
from driver_state.schemas import COMMON_METADATA_FIELDS

# Mirror of distraction_video/metadata/video_windows_10s_v1.csv column order.
AUDIO_6C_CSV_FIELDS = (
    *COMMON_METADATA_FIELDS[:12],
    "label_scheme",
    *COMMON_METADATA_FIELDS[12:],
)

FEATURE_EXTRACTOR_NAME = "panns_cnn14_16k"
FEATURE_EXTRACTOR_VERSION = "cnn14_16k_mAP0.438_v1"
FEATURE_LAYOUT = "audio_features_v1"  # directory name under the feature root

ERROR_NO_FEATURES = "NO_FEATURES_YET"
VIDEO_ALIGNED_TASKS = {"01", "03", "04", "05", "07", "08"}


def load_feature_index(path: str | Path | None) -> dict[str, str]:
    """Map sample_id -> relative NPZ path (processed/<layout>/<stem>.npz)."""
    if path is None:
        return {}
    index: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index[record["sample_id"]] = record["path"]
    return index


def build_6c_rows(
    audit_rows: list[dict[str, str]],
    label_scheme: dict[str, object],
    subject_splits: dict[str, str],
    feature_index: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Turn raw-audit rows into video-aligned 6c metadata rows."""
    features = feature_index or {}
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
        feature_rel = features.get(stem, "")
        npz_exists = feature_rel.endswith(".npz")
        if npz_exists:
            valid = "true"
            ratio = "1"
            mask = f"{feature_rel}::valid_mask"
            shape = "[5, 2048]"
            error = ""
        else:
            valid = "false"
            ratio = "0"
            mask = ""
            shape = ""
            error = ERROR_NO_FEATURES
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
            "feature_dtype": "float32",
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
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args(argv)
    try:
        label_scheme = load_label_scheme(args.label_scheme)
        subject_splits = load_subject_splits(args.subject_splits)
        feature_index = load_feature_index(args.feature_index)
        with Path(args.audit_csv).open(newline="", encoding="utf-8") as stream:
            audit_rows = list(csv.DictReader(stream))
        rows = build_6c_rows(audit_rows, label_scheme, subject_splits, feature_index)
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
