"""Validate UL-DD CAN metadata and NPZ features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driver_state.validation.can_metadata import validate_can_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--feature-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    args = parser.parse_args()
    report = validate_can_metadata(
        args.metadata, feature_root=args.feature_root,
        min_valid_ratio=args.min_valid_ratio,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
