"""Check sample/label/split/session alignment with DCPT audio metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _stem(value: str) -> str:
    base = value.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0]


def _source_stems(value: str) -> list[str]:
    entries = json.loads(value) if value.startswith("[") else [value]
    if not isinstance(entries, list):
        return []
    return [_stem(entry) for entry in entries if isinstance(entry, str) and entry]


def check_pairs(
    audio_csv: Path | str,
    video_csv: Path | str,
    report: Path | str | None = None,
) -> dict[str, object]:
    audio = {row["sample_id"]: row for row in _read(Path(audio_csv))}
    video = {row["sample_id"]: row for row in _read(Path(video_csv))}
    audio_ids, video_ids = set(audio), set(video)
    common = audio_ids & video_ids
    label_mismatch = []
    split_mismatch = []
    session_mismatch = []
    source_mismatch = []
    for sample_id in sorted(common):
        left, right = audio[sample_id], video[sample_id]
        if (left["label_id"], left["label_class"]) != (
            right["label_id"],
            right["label_class"],
        ):
            label_mismatch.append(sample_id)
        if (
            left["subject_id"] != right["subject_id"]
            or left["split"] != right["split"]
        ):
            split_mismatch.append(sample_id)
        if left.get("session_id") and right.get("session_id") and left["session_id"] != right["session_id"]:
            session_mismatch.append(sample_id)
        left_names = _source_stems(left.get("source_file", ""))
        right_names = _source_stems(right.get("source_file", ""))
        if left_names and right_names and (
            set(left_names) != set(right_names) or sample_id not in left_names
        ):
            source_mismatch.append(sample_id)

    errors: list[str] = []
    if audio_ids - video_ids:
        errors.append(f"audio-only samples: {len(audio_ids - video_ids)}")
    if video_ids - audio_ids:
        errors.append(f"video-only samples: {len(video_ids - audio_ids)}")
    if label_mismatch:
        errors.append(f"label mismatches: {len(label_mismatch)}")
    if split_mismatch:
        errors.append(f"subject/split mismatches: {len(split_mismatch)}")
    if session_mismatch:
        errors.append(f"session_id mismatches: {len(session_mismatch)}")
    if source_mismatch:
        errors.append(f"source_file stem mismatches: {len(source_mismatch)}")

    result: dict[str, object] = {
        "status": "PASS" if not errors else "FAIL",
        "audio_rows": len(audio),
        "video_rows": len(video),
        "common_samples": len(common),
        "only_audio_count": len(audio_ids - video_ids),
        "only_video_count": len(video_ids - audio_ids),
        "label_mismatch_count": len(label_mismatch),
        "split_mismatch_count": len(split_mismatch),
        "session_mismatch_count": len(session_mismatch),
        "source_mismatch_count": len(source_mismatch),
        "errors": errors,
    }
    if report:
        destination = Path(report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-csv", required=True)
    parser.add_argument("--video-csv", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    try:
        result = check_pairs(args.audio_csv, args.video_csv, args.report)
    except (ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
