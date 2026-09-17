import csv

from driver_state.preprocessing.fatigue_can.pipeline import CAN_METADATA_FIELDS
from driver_state.preprocessing.fatigue_can.splits import write_can_split_manifests


def make_row(root, *, subject, split, index, valid=True):
    feature = root / "features" / f"{subject}_{index}.npz"
    feature.parent.mkdir(exist_ok=True)
    feature.write_bytes(b"synthetic feature placeholder")
    row = {field: "" for field in CAN_METADATA_FIELDS}
    row.update({
        "sample_id": f"SYNTHETIC_{subject}_{index}",
        "subject_id": subject,
        "session_id": f"{subject}_A",
        "split": split,
        "kss_score": "3",
        "label_id": "0",
        "label_class": "low",
        "window_index": str(index),
        "label_start_ms": "0",
        "label_end_ms": "240000",
        "parent_id": f"SYNTHETIC_{subject}_P0",
        "valid": "true" if valid else "false",
        "qc_status": "PASS" if valid else "PENDING",
        "feature_path": f"features/{subject}_{index}.npz" if valid else "",
        "error": "" if valid else "SYNTHETIC_PENDING",
        "qc_reason_codes": "" if valid else "SYNTHETIC_PENDING",
    })
    return row


def test_split_manifests_use_fixed_subject_partition_and_valid_rows(tmp_path):
    rows = [make_row(tmp_path, subject="D", split="train", index=i) for i in range(8)]
    excluded = make_row(tmp_path, subject="D", split="train", index=8, valid=False)
    excluded.update({
        "parent_id": "SYNTHETIC_D_P1",
        "window_index": "0",
        "label_start_ms": "240000",
        "label_end_ms": "480000",
    })
    rows.append(excluded)
    rows.append(make_row(tmp_path, subject="A", split="val", index=0))
    rows.append(make_row(tmp_path, subject="C", split="test", index=0))
    metadata = tmp_path / "metadata" / "can_windows_30s_v1.csv"
    metadata.parent.mkdir()
    with metadata.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CAN_METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = write_can_split_manifests(metadata, tmp_path)

    assert summary["totals"] == {
        "candidate_windows": 11,
        "valid_windows": 10,
        "excluded_windows": 1,
        "complete_parents_240s": 1,
    }
    assert summary["splits"]["train"]["valid_windows"] == 8
    assert summary["splits"]["val"]["valid_windows"] == 1
    assert summary["splits"]["test"]["valid_windows"] == 1
    with (tmp_path / "manifests" / "can_excluded_windows_v1.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 1
