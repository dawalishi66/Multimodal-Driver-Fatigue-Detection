"""Build a provisional DCPT upper-body video sample manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<task>\d{2})_P(?P<subject>\d{2})_"
    r"(?P<date>\d{8})_(?P<hour>\d{2})_(?P<minute>\d{2})_"
    r"(?P<second>\d{2})_(?P<takeover>\d+)$"
)


def parse_filename(name: str) -> dict[str, str] | None:
    match = FILENAME_RE.match(name)
    if match is None:
        return None
    return match.groupdict()


def build_manifest(config: dict) -> None:
    extracted_dir = Path(config["extracted_dir"])
    manifest_path = Path(config["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    for video_path in sorted(extracted_dir.rglob("*.mp4")):
        stem = video_path.stem
        parsed = parse_filename(stem)
        if parsed is None:
            errors.append({"path": str(video_path), "reason": "filename_not_parsed"})
            continue
        subject = f"P{parsed['subject']}"
        session = (
            f"{subject}_{parsed['date']}_{parsed['hour']}"
            f"{parsed['minute']}_{parsed['second']}"
        )
        rows.append(
            {
                "schema_version": "0.2.0",
                "manifest_version": "dcpt_upper_body_v0_pre",
                "sample_id": stem,
                "task": "distraction",
                "dataset": "DCPT",
                "subject_id": subject,
                "session_id": session,
                "parent_id": stem,
                "window_index": 0,
                "start_s": 0.0,
                "end_s": 10.0,
                "label_raw": parsed["task"],
                "label_id": None,
                "label_scheme": None,
                "source_refs": {
                    "video_file": str(video_path.relative_to(extracted_dir)),
                },
                "qc_status": "pending",
                "qc_reason_codes": [],
                "provisional": True,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_task = Counter(row["label_raw"] for row in rows)
    by_subject = Counter(row["subject_id"] for row in rows)
    summary = {
        "manifest_path": str(manifest_path),
        "video_count": len(rows),
        "parse_error_count": len(errors),
        "by_task": dict(sorted(by_task.items())),
        "subject_count": len(by_subject),
        "subject_min_max": {
            "min": min(int(row["subject_id"][1:]) for row in rows),
            "max": max(int(row["subject_id"][1:]) for row in rows),
        },
        "errors": errors,
    }
    summary_path = manifest_path.with_name("video_manifest_summary_v0_pre.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distraction_video_feature_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    build_manifest(config)


if __name__ == "__main__":
    main()
