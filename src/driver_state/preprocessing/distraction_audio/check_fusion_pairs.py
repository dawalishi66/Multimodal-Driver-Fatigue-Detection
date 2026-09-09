"""Check that the audio 6c metadata is pairwise-fusable with the video metadata.

Multimodal fusion joins modalities by ``sample_id``. This tool asserts that the
audio CSV and the video CSV contain exactly the same ``sample_id`` set, agree on
label (id + class) per sample, and agree on the subject -> split assignment, so
a later SimpleFusion/MulT run can join without re-deciding the cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _source_basenames(value: str) -> list[str]:
    """Return the basename(s) of a source_file entry (JSON array or bare path)."""
    entries: list[str]
    if value.startswith("["):
        entries = json.loads(value)
        if not isinstance(entries, list):
            return []
    else:
        entries = [value]
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            continue
        base = entry.replace("\\", "/").rsplit("/", 1)[-1]
        names.append(base.rsplit(".", 1)[0])
    return names


def check_fusion_pairs(
    audio_csv: Path | str,
    video_csv: Path | str,
    report: Path | str | None = None,
) -> dict[str, object]:
    audio_rows = _read(Path(audio_csv))
    video_rows = _read(Path(video_csv))
    audio = {r["sample_id"]: r for r in audio_rows}
    video = {r["sample_id"]: r for r in video_rows}
    audio_ids = set(audio)
    video_ids = set(video)

    only_audio = sorted(audio_ids - video_ids)
    only_video = sorted(video_ids - audio_ids)
    common = audio_ids & video_ids
    label_mismatch: list[tuple[str, str, str, str, str]] = []
    split_mismatch: list[tuple[str, str, str, str, str]] = []
    session_mismatch: list[tuple[str, str, str]] = []
    source_mismatch: list[tuple[str, str, str]] = []
    for sample_id in sorted(common):
        a, v = audio[sample_id], video[sample_id]
        if (a["label_id"], a["label_class"]) != (v["label_id"], v["label_class"]):
            label_mismatch.append((sample_id, a["label_id"], a["label_class"],
                                   v["label_id"], v["label_class"]))
        # session_id must be byte-identical across modalities for a fusion join.
        a_session = a.get("session_id", "")
        v_session = v.get("session_id", "")
        if a_session and v_session and a_session != v_session:
            session_mismatch.append((sample_id, a_session, v_session))
        a_split = a["split"]
        # validate against each side's subject list where possible
        if a["subject_id"] != v["subject_id"]:
            split_mismatch.append((sample_id, a["subject_id"], v["subject_id"], a_split, v["split"]))
        elif a_split != v["split"]:
            split_mismatch.append((sample_id, a["subject_id"], v["subject_id"], a_split, v["split"]))
        # source_file entries must share the sample_id main identifier (basename minus extension).
        audio_names = _source_basenames(a.get("source_file", ""))
        video_names = _source_basenames(v.get("source_file", ""))
        if audio_names and video_names:
            if set(audio_names) != set(video_names) or sample_id not in audio_names:
                source_mismatch.append((sample_id, ",".join(audio_names), ",".join(video_names)))

    errors: list[str] = []
    if only_audio:
        errors.append(f"audio-only samples (not in video): {len(only_audio)} e.g. {only_audio[:5]}")
    if only_video:
        errors.append(f"video-only samples (not in audio): {len(only_video)} e.g. {only_video[:5]}")
    if label_mismatch:
        errors.append(f"label mismatches on common samples: {len(label_mismatch)} e.g. {label_mismatch[:5]}")
    if split_mismatch:
        errors.append(f"subject/split mismatches: {len(split_mismatch)} e.g. {split_mismatch[:5]}")
    if session_mismatch:
        errors.append(f"session_id mismatches: {len(session_mismatch)} e.g. {session_mismatch[:5]}")
    if source_mismatch:
        errors.append(f"source_file basename mismatches: {len(source_mismatch)} e.g. {source_mismatch[:5]}")

    result: dict[str, object] = {
        "status": "PASS" if not errors else "FAIL",
        "audio_rows": len(audio_rows),
        "video_rows": len(video_rows),
        "common_samples": len(common),
        "only_audio_count": len(only_audio),
        "only_video_count": len(only_video),
        "label_mismatch_count": len(label_mismatch),
        "split_mismatch_count": len(split_mismatch),
        "session_mismatch_count": len(session_mismatch),
        "source_mismatch_count": len(source_mismatch),
        "errors": errors,
    }
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-csv", required=True)
    parser.add_argument("--video-csv", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    try:
        result = check_fusion_pairs(args.audio_csv, args.video_csv, args.report)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
