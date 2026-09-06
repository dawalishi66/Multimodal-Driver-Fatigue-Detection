"""Thin CLI: DCPT distraction-audio audit -> metadata CSV + QC sidecar.

Runs the same entry point as
``python -m driver_state.preprocessing.distraction_audio.build_metadata`` without
requiring an editable install (adds ``src/`` to ``sys.path``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.preprocessing.distraction_audio.build_metadata import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())