"""Thin CLI wrapper (adds src/ to sys.path; no editable install required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.preprocessing.distraction_audio.build_audio_evaluation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
