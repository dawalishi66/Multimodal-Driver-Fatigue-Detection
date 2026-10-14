"""Lightweight DCPT audio QC using only the standard library ``wave`` + numpy.

The audit is deliberately conservative and non-destructive:

* it never trims, re-samples, re-channels, deletes silence, de-noises,
  time-stretches or interpolates the waveform;
* it only records measured format facts and a few markers (all-zero, possible
  clipping) that the team must review before anything is excluded;
* a short tail is allowed up to ``MAX_TAIL_PADDING_S`` (0.10 s) and is logged as
  traceable ``pass_with_flags``; anything beyond that is ``pending``.

Quality vocabulary follows the v0.2 spec: ``pass`` / ``pass_with_flags`` /
``pending`` / ``exclude``. Nothing is automatically ``exclude``d here.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field

import numpy as np

NOMINAL_DURATION_S = 10.0
MAX_TAIL_PADDING_S = 0.10
EXPECTED_SAMPLERATE = 44_100
EXPECTED_CHANNELS = 2
EXPECTED_SAMPLE_WIDTH = 2  # 16-bit PCM
# Int16 peaks; a small fraction of extreme samples is treated only as a marker.
CLIPPING_PEAK = 32_767
CLIPPING_RATIO_THRESHOLD = 1e-4
# Slack before "longer than nominal" is reported (float duration rounding).
DURATION_TOLERANCE_S = 0.05

# Stable marker/reason codes (also reused as metadata ``error`` values).
CODE_EMPTY_AUDIO = "EMPTY_AUDIO"
CODE_ZERO_AUDIO = "ZERO_AUDIO"
CODE_POSSIBLE_CLIPPING = "POSSIBLE_CLIPPING"
CODE_SAMPLE_RATE_DEVIATION = "SAMPLE_RATE_DEVIATION"
CODE_CHANNELS_NOT_STEREO = "CHANNELS_NOT_STEREO"
CODE_SAMPLE_WIDTH_NOT_16BIT = "SAMPLE_WIDTH_NOT_16BIT"
CODE_LONGER_THAN_NOMINAL = "LONGER_THAN_NOMINAL"
CODE_TAIL_SHORTFALL_EXCEEDS = "TAIL_SHORTFALL_EXCEEDS"
CODE_MINOR_TAIL_SHORTFALL = "MINOR_TAIL_SHORTFALL"

# Codes that block a clip from the formal set until a human reviews the file.
PENDING_CODES = {
    CODE_EMPTY_AUDIO,
    CODE_ZERO_AUDIO,
    CODE_SAMPLE_RATE_DEVIATION,
    CODE_CHANNELS_NOT_STEREO,
    CODE_SAMPLE_WIDTH_NOT_16BIT,
    CODE_LONGER_THAN_NOMINAL,
    CODE_TAIL_SHORTFALL_EXCEEDS,
}


class QcDecodeError(Exception):
    """Raised when a file cannot be decoded as a WAV container."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AudioQcResult:
    """Measured facts and quality state for one WAV clip."""

    sample_id: str
    measured_duration_s: float
    samplerate: int
    channels: int
    sample_width_bytes: int
    nframes: int
    tail_shortfall_s: float
    raw_observed_fraction: float
    qc_status: str
    flags: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    @property
    def primary_error(self) -> str:
        """Most specific reason a row is not usable yet (None if only pending on features)."""
        if self.reason_codes:
            return self.reason_codes[0]
        if self.qc_status == "pending":
            return "PENDING_REVIEW"
        return ""


def analyze_wav_bytes(data: bytes, sample_id: str) -> AudioQcResult:
    """Analyze raw WAV bytes and return measured format + quality markers.

    Raises :class:`QcDecodeError` for containers that ``wave`` cannot decode so
    the caller can record a stable error instead of fabricating a duration.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            samplerate = reader.getframerate()
            nframes = reader.getnframes()
            raw = reader.readframes(nframes)
    except (wave.Error, EOFError) as exc:  # pragma: no cover - container guards
        raise QcDecodeError("UNSUPPORTED_CONTAINER", f"cannot decode WAV: {type(exc).__name__}") from exc

    measured = nframes / samplerate if samplerate > 0 else 0.0
    tail_shortfall = max(0.0, NOMINAL_DURATION_S - measured) if measured > 0 else NOMINAL_DURATION_S
    raw_observed = min(measured, NOMINAL_DURATION_S) / NOMINAL_DURATION_S if measured > 0 else 0.0

    flags: list[str] = []
    reasons: list[str] = []

    if samplerate != EXPECTED_SAMPLERATE:
        flags.append(CODE_SAMPLE_RATE_DEVIATION)
        reasons.append(CODE_SAMPLE_RATE_DEVIATION)
    if channels != EXPECTED_CHANNELS:
        flags.append(CODE_CHANNELS_NOT_STEREO)
        reasons.append(CODE_CHANNELS_NOT_STEREO)
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        flags.append(CODE_SAMPLE_WIDTH_NOT_16BIT)
        reasons.append(CODE_SAMPLE_WIDTH_NOT_16BIT)

    if nframes == 0:
        flags.append(CODE_EMPTY_AUDIO)
        reasons.append(CODE_EMPTY_AUDIO)
    elif sample_width == EXPECTED_SAMPLE_WIDTH:
        samples = np.frombuffer(raw, dtype=np.int16)
        if samples.size == 0:
            flags.append(CODE_EMPTY_AUDIO)
            reasons.append(CODE_EMPTY_AUDIO)
        else:
            if np.count_nonzero(samples) == 0:
                flags.append(CODE_ZERO_AUDIO)
                reasons.append(CODE_ZERO_AUDIO)
            abs_samples = np.abs(samples.astype(np.int32))
            if np.mean(abs_samples >= CLIPPING_PEAK) > CLIPPING_RATIO_THRESHOLD:
                flags.append(CODE_POSSIBLE_CLIPPING)

    if measured > NOMINAL_DURATION_S + DURATION_TOLERANCE_S:
        flags.append(CODE_LONGER_THAN_NOMINAL)
        reasons.append(CODE_LONGER_THAN_NOMINAL)
    elif measured < NOMINAL_DURATION_S:
        if tail_shortfall <= MAX_TAIL_PADDING_S:
            flags.append(CODE_MINOR_TAIL_SHORTFALL)
        else:
            flags.append(CODE_TAIL_SHORTFALL_EXCEEDS)
            reasons.append(CODE_TAIL_SHORTFALL_EXCEEDS)

    status = "pending" if reasons else ("pass_with_flags" if flags else "pass")
    return AudioQcResult(
        sample_id=sample_id,
        measured_duration_s=round(measured, 6),
        samplerate=samplerate,
        channels=channels,
        sample_width_bytes=sample_width,
        nframes=nframes,
        tail_shortfall_s=round(tail_shortfall, 6),
        raw_observed_fraction=round(raw_observed, 6),
        qc_status=status,
        flags=flags,
        reason_codes=reasons,
    )