"""Thin CLI: run the public metadata validator forced to task=distraction.

Example:
    python scripts/validate_distraction_audio_metadata.py \
        --metadata <metadata.csv> --feature-root <feature_root> [--report out.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.validation.metadata import main as validate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(validate_main(["--task", "distraction", *sys.argv[1:]]))