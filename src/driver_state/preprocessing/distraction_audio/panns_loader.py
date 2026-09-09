"""Load the frozen PANNs Cnn14_16k model for feature extraction.

numba/llvmlite is broken in this Python 3.14 environment (llvmlite.dll cannot
load, NTSTATUS 0xc0e90002). PANNs' torchlibrosa modules only touch librosa at
construction time, so we replace the few entry points it needs with equivalent
numpy/scipy implementations and never import numba.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch


def _make_librosa_stubs() -> None:
    import scipy.signal

    filters = types.ModuleType("librosa.filters")
    filters.get_window = lambda window, win_length, fftbins=True: scipy.signal.get_window(
        window, win_length, fftbins=fftbins)

    def _mel(sr, n_fft, n_mels, fmin, fmax, **kwargs):
        # Values are replaced by the official checkpoint weights (melW) on load;
        # only the shape must match: (n_mels, 1 + n_fft // 2).
        return np.zeros((n_mels, 1 + n_fft // 2), dtype="float64")

    filters.mel = _mel
    filters.window_sumsquare = lambda *a, **k: np.zeros(1 + 512 // 2, dtype="float64")
    sys.modules["librosa.filters"] = filters

    util = types.ModuleType("librosa.util")
    def _pad_center(data, size):
        data = np.asarray(data)
        length = data.shape[-1]
        if length == size:
            return data
        if length < size:
            pad = size - length
            pad_left = pad // 2
            pad_right = pad - pad_left
            pad_width = [(0, 0)] * data.ndim
            pad_width[-1] = (pad_left, pad_right)
            return np.pad(data, pad_width, mode="constant")
        trim = length - size
        trim_left = trim // 2
        slices = [slice(None)] * data.ndim
        slices[-1] = slice(trim_left, trim_left + size)
        return data[tuple(slices)]

    util.pad_center = _pad_center
    util.normalize = lambda x, norm=np.inf, axis=-1: x
    sys.modules["librosa.util"] = util


def load_cnn14_16k(weights_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Return frozen Cnn14_16k with official weights on ``device`` (eval mode)."""
    _make_librosa_stubs()
    vendor = Path(__file__).resolve().parent / "vendor" / "audioset_tagging_cnn" / "pytorch"
    sys.path.insert(0, str(vendor))
    import torchlibrosa  # noqa: F401  (must import after stubs)
    from models import Cnn14_16k  # type: ignore

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160,
                      mel_bins=64, fmin=50, fmax=8000, classes_num=527)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model