"""Command-line entry point for UL-DD CAN preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driver_state.preprocessing.fatigue_can.pipeline import CanConfig, build_can_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = CanConfig.from_json(args.config)
    audit = build_can_dataset(args.dataset_root, args.output_root, config=config)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
