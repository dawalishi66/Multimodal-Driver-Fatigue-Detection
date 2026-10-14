"""Synthetic WAV builders for distraction-audio tests (no real data)."""

from __future__ import annotations

import io
import wave

import numpy as np


def make_wav_bytes(
    *,
    samplerate: int = 44100,
    channels: int = 2,
    sampwidth: int = 2,
    duration_s: float = 10.0,
    amplitude: int = 8000,
    freq: float = 440.0,
    style: str = "tone",
    seed: int = 0,
) -> bytes:
    """Return a small valid WAV (as bytes) with the requested properties."""
    n = int(round(duration_s * samplerate))
    if style == "tone":
        time = np.arange(n) / samplerate
        samples = (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.int16)
    elif style == "zero":
        samples = np.zeros(n, dtype=np.int16)
    elif style == "clip":
        time = np.arange(n) / samplerate
        period = 1.0 / freq
        samples = np.where((time % period) < (period / 2), 32767, -32767).astype(np.int16)
    elif style == "noise":
        rng = np.random.default_rng(seed)
        samples = rng.integers(-30000, 30000, size=n, dtype=np.int16)
    else:  # pragma: no cover - test misuse
        raise ValueError(f"unknown style: {style}")

    if channels == 1:
        frames = samples
    else:
        frames = np.repeat(samples[:, None], channels, axis=1)

    if sampwidth == 2:
        payload = frames.astype("<i2").tobytes()
    elif sampwidth == 1:
        unsigned = np.clip((frames.astype(np.int32) >> 8) + 128, 0, 255).astype(np.uint8)
        payload = unsigned.tobytes()
    else:  # pragma: no cover - test misuse
        raise ValueError("sampwidth must be 1 or 2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(samplerate)
        writer.writeframes(payload)
    return buffer.getvalue()