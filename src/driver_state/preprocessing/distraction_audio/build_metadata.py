"""DCPT distraction-audio audit: build the standard metadata CSV + QC sidecar.

Stage A/B deliverable owned by 胡煦轩 (distraction audio). It performs a
non-destructive raw audit of ``First_person_view_audio`` clips and exports:

* ``metadata/audio_windows_10s_v1.csv``  (COMMON_METADATA_FIELDS, one row per
  parseable DCPT clip; nominal ``[0, 10000)`` ms; rows are ``valid=false`` until
  PANNs features exist and the DCPT subject split is frozen);
* ``qc/audio_qc_v1.json``                 (per-clip QC records + aggregates);
* ``docs/audio_data_audit_v1.md``         (human-readable audit summary).

No row is ever marked ``valid=true`` here: without extracted features the public
validator requires invalid rows with an explicit error. Until 李坤洋 freezes the
24/8/8 subject manifest, the ``split`` column is left empty (a frozen manifest
can be supplied with ``--subject-splits``).
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from driver_state.constants import SPLITS
from driver_state.preprocessing.distraction_audio.naming import (
    NAMING_SCHEMA_VERSION,
    DcptClipRef,
    parse_clip_filename,
)
from driver_state.preprocessing.distraction_audio.qc import (
    AudioQcResult,
    QcDecodeError,
    analyze_wav_bytes,
)
from driver_state.schemas import COMMON_METADATA_FIELDS

METADATA_FILENAME = "audio_windows_10s_v1.csv"
QC_FILENAME = "audio_qc_v1.json"
AUDIT_FILENAME = "audio_data_audit_v1.md"
QUALITY_VERSION = "qc_v0.1_preflight"
# Identifies the tool that produced the row provenance until PANNs features land.
EXTRACTOR_NAME = "dcpt_audio_qc_audit"
EXTRACTOR_VERSION = "0.1.0"
ERROR_NO_FEATURES = "NO_FEATURES_YET"

@dataclass(frozen=True)
class SourceEntry:
    """One file inside the audio source (zip archive or directory)."""

    rel: str  # archive- / root-relative posix path (machine independent)
    base: str
    reader: Callable[[], bytes]


class _ZipSource:
    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> Iterator[SourceEntry]:
        with zipfile.ZipFile(self.path) as archive:
            names = sorted(n for n in archive.namelist() if not n.endswith("/"))
            for name in names:
                rel = name.replace("\\", "/")

                def read(name: str = name, archive: zipfile.ZipFile = archive) -> bytes:
                    return archive.read(name)

                yield SourceEntry(rel=rel, base=rel.rsplit("/", 1)[-1], reader=read)


class _DirSource:
    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> Iterator[SourceEntry]:
        for file in sorted(self.path.rglob("*")):
            if not file.is_file():
                continue
            rel = file.relative_to(self.path).as_posix()

            def read(file: Path = file) -> bytes:
                return file.read_bytes()

            yield SourceEntry(rel=rel, base=rel.rsplit("/", 1)[-1], reader=read)


def _open_source(audio_source: str | Path) -> _ZipSource | _DirSource:
    path = Path(audio_source)
    if path.is_dir():
        return _DirSource(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        return _ZipSource(path)
    raise ValueError("audio-source must be a directory or a .zip archive")


def _load_subject_splits(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("subject-splits must be a JSON object mapping subject_id to split")
    splits: dict[str, str] = {}
    for subject, split in data.items():
        if not isinstance(subject, str) or not isinstance(split, str) or split not in SPLITS:
            raise ValueError("subject-splits must map subject_id -> one of train/val/test")
        splits[subject] = split
    return splits


def _row_for(ref: DcptClipRef, rel: str, qc: AudioQcResult | None, error: str,
             split: str) -> dict[str, str]:
    return {
        "sample_id": ref.stem,
        "modality": "audio",
        "subject_id": ref.subject_id,
        "session_id": ref.session_id,
        "split": split,
        "source_file": rel,
        "window_index": "0",
        "window_start_ms": "0",
        "window_end_ms": "10000",
        "duration_ms": "10000",
        "label_class": ref.label_class,
        "label_id": str(ref.label_id),
        "valid": "false",
        "valid_ratio": "0",
        "mask": "",
        "feature_path": "",
        "feature_shape": "",
        "feature_dtype": "",
        "extractor_name": EXTRACTOR_NAME,
        "extractor_version": EXTRACTOR_VERSION,
        "error": error or ERROR_NO_FEATURES,
    }


def _qc_record(ref: DcptClipRef, rel: str, qc: AudioQcResult | None,
               error: str) -> dict[str, object]:
    base: dict[str, object] = {
        "sample_id": ref.stem,
        "subject_id": ref.subject_id,
        "session_id": ref.session_id,
        "task_code": ref.task_code,
        "label_id": ref.label_id,
        "label_class": ref.label_class,
        "source_rel": rel,
        "qc_status": qc.qc_status if qc else "pending",
        "flags": qc.flags if qc else [],
        "reason_codes": qc.reason_codes if qc else [error],
        "metadata_error": error,
    }
    if qc is not None:
        base.update({
            "measured_duration_s": qc.measured_duration_s,
            "samplerate": qc.samplerate,
            "channels": qc.channels,
            "sample_width_bytes": qc.sample_width_bytes,
            "nframes": qc.nframes,
            "tail_shortfall_s": qc.tail_shortfall_s,
            "raw_observed_fraction": qc.raw_observed_fraction,
        })
    return base


def build_audio_metadata(audio_source: str | Path, output_dir: str | Path,
                         subject_splits: dict[str, str] | None = None,
                         source_name: str | None = None) -> dict[str, object]:
    """Audit every clip and write metadata CSV + QC sidecar + audit markdown."""
    source = _open_source(audio_source)
    splits = dict(subject_splits or {})
    output = Path(output_dir)
    metadata_dir = output / "metadata"
    qc_dir = output / "qc"
    docs_dir = output / "docs"
    for directory in (metadata_dir, qc_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    unparsed: list[str] = []
    task_counter: collections.Counter = collections.Counter()
    subject_counter: collections.Counter = collections.Counter()
    status_counter: collections.Counter = collections.Counter()
    flag_counter: collections.Counter = collections.Counter()
    reason_counter: collections.Counter = collections.Counter()
    durations: list[float] = []
    shortfalls: list[float] = []

    files_total = 0
    for entry in source.entries():
        files_total += 1
        ref = parse_clip_filename(entry.rel)
        if ref is None:
            unparsed.append(entry.rel)
            continue
        try:
            data = entry.reader()
            qc = analyze_wav_bytes(data, ref.stem)
        except QcDecodeError as exc:
            qc = None
            error = exc.code
        except (OSError, zipfile.BadZipFile, EOFError):
            qc = None
            error = "FILE_UNREADABLE"
        else:
            error = qc.primary_error
        split = splits.get(ref.subject_id, "")
        rows.append(_row_for(ref, entry.rel, qc, error, split))
        records.append(_qc_record(ref, entry.rel, qc, error))
        task_counter[ref.label_class] += 1
        subject_counter[ref.subject_id] += 1
        status = qc.qc_status if qc else "pending"
        status_counter[status] += 1
        if qc is not None:
            durations.append(qc.measured_duration_s)
            shortfalls.append(qc.tail_shortfall_s)
            for flag in qc.flags:
                flag_counter[flag] += 1
            for reason in qc.reason_codes:
                reason_counter[reason] += 1
        else:
            reason_counter[error] += 1

    metadata_path = metadata_dir / METADATA_FILENAME
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMMON_METADATA_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    duration_stats = {
        "count": len(durations),
        "min_s": round(min(durations), 6) if durations else None,
        "max_s": round(max(durations), 6) if durations else None,
        "mean_s": round(sum(durations) / len(durations), 6) if durations else None,
    }
    shortfall_stats = {
        "count": len(shortfalls),
        "min_s": round(min(shortfalls), 6) if shortfalls else None,
        "max_s": round(max(shortfalls), 6) if shortfalls else None,
    }
    aggregates = {
        "files_total": files_total,
        "rows_total": len(rows),
        "unparsed_total": len(unparsed),
        "rows_by_class": dict(sorted(task_counter.items())),
        "rows_by_subject": dict(sorted(subject_counter.items())),
        "rows_by_qc_status": dict(sorted(status_counter.items())),
        "flag_counts": dict(sorted(flag_counter.items())),
        "reason_counts": dict(sorted(reason_counter.items())),
        "measured_duration": duration_stats,
        "tail_shortfall": shortfall_stats,
        "subject_splits_provided": bool(splits),
        "unparsed_files": unparsed,
    }

    sidecar = {
        "schema_version": "0.2.0",
        "task": "distraction",
        "modality": "audio",
        "dataset": "DCPT",
        "quality_version": QUALITY_VERSION,
        "naming_schema": NAMING_SCHEMA_VERSION,
        "extractor_name": EXTRACTOR_NAME,
        "extractor_version": EXTRACTOR_VERSION,
        "source": str(Path(audio_source).resolve()),
        "source_label": source_name or Path(audio_source).name,
        "nominal_duration_s": 10.0,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "aggregates": aggregates,
        "clips": records,
    }
    qc_path = qc_dir / QC_FILENAME
    qc_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit_path = docs_dir / AUDIT_FILENAME
    audit_path.write_text(_render_audit_markdown(sidecar), encoding="utf-8")

    return {
        "metadata_path": str(metadata_path),
        "qc_path": str(qc_path),
        "audit_path": str(audit_path),
        "aggregates": aggregates,
    }


def _render_audit_markdown(sidecar: dict[str, object]) -> str:
    agg = sidecar["aggregates"]
    lines = [
        "# DCPT Distraction Audio Audit v1",
        "",
        f"- Task: {sidecar['task']} / {sidecar['modality']} / {sidecar['dataset']}",
        f"- Schema version: {sidecar['schema_version']}",
        f"- Quality version: {sidecar['quality_version']}",
        f"- Naming schema: {sidecar['naming_schema']} (provisional until P04 cross-modal check)",
        f"- Source label: {sidecar['source_label']}",
        f"- Generated: {sidecar['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Files enumerated: {agg['files_total']}",
        f"- Metadata rows: {agg['rows_total']} (unparsed: {agg['unparsed_total']})",
        f"- Rows by class: {json.dumps(agg['rows_by_class'], ensure_ascii=False)}",
        f"- Rows by QC status: {json.dumps(agg['rows_by_qc_status'], ensure_ascii=False)}",
        f"- Flags: {json.dumps(agg['flag_counts'], ensure_ascii=False)}",
        f"- Pending reasons: {json.dumps(agg['reason_counts'], ensure_ascii=False)}",
        f"- Measured duration s: {json.dumps(agg['measured_duration'], ensure_ascii=False)}",
        f"- Tail shortfall s: {json.dumps(agg['tail_shortfall'], ensure_ascii=False)}",
        f"- Subjects (rows): {len(agg['rows_by_subject'])}",
        f"- Subject split frozen: {'yes' if agg['subject_splits_provided'] else 'no (pending 李坤洋)'}",
        "",
        "## Meaning of the metadata rows",
        "",
        "Every row is nominal `[0, 10000)` ms with `valid=false` until PANNs",
        "features are extracted and the DCPT 24/8/8 subject manifest is frozen.",
        "Measured format facts live in `qc/audio_qc_v1.json`; nothing here is",
        "fabricated observation.",
        "",
        "## Open items (owner-gated, not decided by code)",
        "",
        "1. DCPT fixed subject split (24/8/8, seed 2026) - freeze by 李坤洋.",
        "2. Pre-feature valid-row policy (all `NO_FEATURES_YET` vs an exception).",
        "3. P04: cross-modal main-id / common start verification with 陈星宇.",
        "4. P07: PANNs feature extraction and feature-version freeze (next milestone).",
        "",
    ]
    if agg["unparsed_files"]:
        lines += ["## Unparsed files", ""]
        lines += [f"- `{name}`" for name in agg["unparsed_files"]]
        lines += [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DCPT distraction-audio raw audit -> metadata CSV + QC sidecar.")
    parser.add_argument("--audio-source", required=True,
                        help="Directory or .zip archive containing DCPT audio clips")
    parser.add_argument("--output-dir", required=True,
                        help="Directory that receives metadata/, qc/ and docs/")
    parser.add_argument("--subject-splits", default=None,
                        help="Optional JSON object {subject_id: train|val|test} (frozen manifest)")
    parser.add_argument("--source-label", default=None,
                        help="Human-readable label recorded in the QC sidecar")
    args = parser.parse_args(argv)
    try:
        splits = _load_subject_splits(args.subject_splits)
        result = build_audio_metadata(
            audio_source=args.audio_source,
            output_dir=args.output_dir,
            subject_splits=splits,
            source_name=args.source_label,
        )
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    agg = result["aggregates"]
    print(f"files_total={agg['files_total']} rows={agg['rows_total']} "
          f"unparsed={agg['unparsed_total']}")
    print(f"metadata: {result['metadata_path']}")
    print(f"qc:       {result['qc_path']}")
    print(f"audit:    {result['audit_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())