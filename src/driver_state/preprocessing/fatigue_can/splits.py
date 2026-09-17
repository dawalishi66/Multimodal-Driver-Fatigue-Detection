"""Freeze train/val/test manifests from validated UL-DD CAN metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence

from driver_state.constants import ULDD_SPLIT_BY_SUBJECT, ULDD_SUBJECT_SPLITS, kss_label
from driver_state.preprocessing.fatigue_can.pipeline import (
    CAN_METADATA_FIELDS,
    CAN_PARENT_FIELDS,
)

SPLIT_ORDER = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp",
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        missing = sorted(set(CAN_METADATA_FIELDS) - set(fields))
        if missing:
            raise ValueError(f"CAN metadata is missing fields: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("CAN metadata is empty")
    return fields, rows


def _feature_file(output_root: Path, relative: str) -> Path:
    windows = PureWindowsPath(relative)
    if not relative or windows.drive or windows.root:
        raise ValueError("feature_path must be relative")
    path = Path(relative.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("feature_path cannot contain parent traversal")
    resolved = (output_root / path).resolve()
    if not resolved.is_relative_to(output_root):
        raise ValueError("feature_path escapes output_root")
    return resolved


def _validate_row(row: dict[str, str], output_root: Path, seen: set[str]) -> bool:
    sample_id = row["sample_id"].strip()
    if not sample_id or sample_id in seen:
        raise ValueError(f"empty or duplicate sample_id: {sample_id!r}")
    seen.add(sample_id)
    subject = row["subject_id"].strip()
    split = row["split"].strip()
    if split not in SPLIT_ORDER or ULDD_SPLIT_BY_SUBJECT.get(subject) != split:
        raise ValueError(f"subject/split mismatch for {sample_id}: {subject}/{split}")
    expected_id, expected_class = kss_label(float(row["kss_score"]))
    if row["label_id"] != str(expected_id) or row["label_class"] != expected_class:
        raise ValueError(f"KSS mapping mismatch for {sample_id}")
    valid_text = row["valid"].strip().lower()
    if valid_text not in ("true", "false"):
        raise ValueError(f"valid must be true or false for {sample_id}")
    valid = valid_text == "true"
    if valid:
        if row["qc_status"] != "PASS" or row["error"] or row["qc_reason_codes"]:
            raise ValueError(f"valid/QC mismatch for {sample_id}")
        feature = _feature_file(output_root, row["feature_path"])
        if not feature.is_file():
            raise ValueError(f"feature file does not exist for {sample_id}")
    elif not row["error"] or row["qc_status"] not in ("FAIL", "PENDING"):
        raise ValueError(f"invalid row must retain an explicit failure for {sample_id}")
    return valid


def _parent_row(parent_id: str, group: Sequence[dict[str, str]]) -> dict[str, str]:
    first = group[0]
    reasons = sorted({
        code
        for row in group
        for code in row["qc_reason_codes"].split(";")
        if code
    })
    valid_count = sum(row["valid"] == "true" for row in group)
    complete = (
        len(group) == 8
        and sorted(int(row["window_index"]) for row in group) == list(range(8))
        and valid_count == 8
    )
    return {
        "parent_id": parent_id,
        "subject_id": first["subject_id"],
        "session_id": first["session_id"],
        "split": first["split"],
        "label_start_ms": first["label_start_ms"],
        "label_end_ms": first["label_end_ms"],
        "kss_score": first["kss_score"],
        "label_class": first["label_class"],
        "label_id": first["label_id"],
        "window_count": str(len(group)),
        "valid_window_count": str(valid_count),
        "complete_for_240s": "true" if complete else "false",
        "qc_reason_codes": ";".join(reasons),
    }


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# UL-DD CAN 固定划分 v1",
        "",
        f"生成时间（UTC）：`{summary['created_utc']}`",
        "",
        "本划分直接继承团队固定的驾驶员级 train/val/test，不进行窗口级随机划分。",
        "特征文件只保留一份；各 split CSV 通过相对 `feature_path` 引用同一 `features/`。",
        "",
        "| split | 驾驶员 | session | 候选窗口 | 有效窗口 | 排除窗口 | 完整240秒区间 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLIT_ORDER:
        item = summary["splits"][split]
        lines.append(
            f"| {split} | {item['subject_count']} | {item['session_count']} | "
            f"{item['candidate_windows']} | {item['valid_windows']} | "
            f"{item['excluded_windows']} | {item['complete_parents_240s']} |"
        )
    lines.extend([
        "",
        "## 使用规则",
        "",
        "- 训练只读取 `can_train_windows_v1.csv`。",
        "- 模型选择和早停只读取 `can_val_windows_v1.csv`。",
        "- 模型、超参数和阈值冻结后，才能读取 `can_test_windows_v1.csv`。",
        "- 240秒评价只读取对应 split 的 `can_<split>_parents_240s_v1.csv`。",
        "- `can_excluded_windows_v1.csv` 只用于审计，不能进入训练。",
        "- 标准化和类别权重只能用 train 清单计算。",
        "",
    ])
    return "\n".join(lines)


def write_can_split_manifests(
    metadata: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Write deterministic valid-window and complete-parent split manifests."""
    metadata = Path(metadata).resolve()
    output_root = Path(output_root).resolve()
    try:
        metadata.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("metadata must be inside output_root") from exc
    fields, rows = _read_metadata(metadata)
    seen: set[str] = set()
    valid_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_ORDER}
    candidate_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_ORDER}
    excluded: list[dict[str, str]] = []
    for row in rows:
        valid = _validate_row(row, output_root, seen)
        split = row["split"]
        candidate_by_split[split].append(row)
        if valid:
            valid_by_split[split].append(row)
        else:
            excluded.append(row)

    parent_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        parent_groups[row["parent_id"]].append(row)
    complete_parents: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_ORDER}
    all_complete: list[dict[str, str]] = []
    for parent_id, group in parent_groups.items():
        parent = _parent_row(parent_id, group)
        if parent["complete_for_240s"] == "true":
            complete_parents[parent["split"]].append(parent)
            all_complete.append(parent)
    for values in complete_parents.values():
        values.sort(key=lambda row: (
            row["subject_id"], row["session_id"], int(row["label_start_ms"])
        ))
    all_complete.sort(key=lambda row: (
        SPLIT_ORDER.index(row["split"]), row["subject_id"], row["session_id"],
        int(row["label_start_ms"]),
    ))

    manifest_dir = output_root / "manifests"
    output_files: list[tuple[Path, int]] = []
    for split in SPLIT_ORDER:
        path = manifest_dir / f"can_{split}_windows_v1.csv"
        _atomic_csv(path, fields, valid_by_split[split])
        output_files.append((path, len(valid_by_split[split])))
        parent_path = manifest_dir / f"can_{split}_parents_240s_v1.csv"
        _atomic_csv(parent_path, CAN_PARENT_FIELDS, complete_parents[split])
        output_files.append((parent_path, len(complete_parents[split])))
    excluded_path = manifest_dir / "can_excluded_windows_v1.csv"
    _atomic_csv(excluded_path, fields, excluded)
    output_files.append((excluded_path, len(excluded)))
    complete_path = manifest_dir / "can_complete_parents_240s_v1.csv"
    _atomic_csv(complete_path, CAN_PARENT_FIELDS, all_complete)
    output_files.append((complete_path, len(all_complete)))

    split_summary: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        candidates = candidate_by_split[split]
        valid_rows = valid_by_split[split]
        split_summary[split] = {
            "subjects": list(ULDD_SUBJECT_SPLITS[split]),
            "subject_count": len({row["subject_id"] for row in candidates}),
            "sessions": sorted({row["session_id"] for row in candidates}),
            "session_count": len({row["session_id"] for row in candidates}),
            "candidate_windows": len(candidates),
            "valid_windows": len(valid_rows),
            "excluded_windows": len(candidates) - len(valid_rows),
            "valid_label_counts": dict(sorted(Counter(
                row["label_class"] for row in valid_rows
            ).items())),
            "complete_parents_240s": len(complete_parents[split]),
            "complete_parent_label_counts": dict(sorted(Counter(
                row["label_class"] for row in complete_parents[split]
            ).items())),
        }
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UL-DD",
        "task": "fatigue",
        "modality": "can",
        "split_unit": "subject_id",
        "source_metadata": metadata.relative_to(output_root).as_posix(),
        "source_metadata_sha256": _sha256(metadata),
        "splits": split_summary,
        "totals": {
            "candidate_windows": len(rows),
            "valid_windows": sum(len(value) for value in valid_by_split.values()),
            "excluded_windows": len(excluded),
            "complete_parents_240s": len(all_complete),
        },
        "checks": {
            "sample_id_unique": len(seen) == len(rows),
            "subject_split_matches_fixed_protocol": True,
            "valid_rows_have_existing_features": True,
            "train_val_test_subjects_disjoint": True,
        },
        "test_policy": "Do not load the test manifest until model and selection rules are frozen.",
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "rows": count,
                "sha256": _sha256(path),
            }
            for path, count in output_files
        ],
    }
    summary_path = manifest_dir / "can_split_manifest_v1.json"
    _atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(output_root / "reports" / "can_split_summary_v1.md", _report_markdown(summary))
    return summary
