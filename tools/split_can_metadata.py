"""Create frozen train/val/test manifests from validated CAN metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driver_state.preprocessing.fatigue_can.splits import write_can_split_manifests
from driver_state.validation.can_metadata import validate_can_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    validation = validate_can_metadata(
        args.metadata, feature_root=args.output_root, min_valid_ratio=0.95
    )
    if validation["status"] != "PASS":
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1
    summary = write_can_split_manifests(args.metadata, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
