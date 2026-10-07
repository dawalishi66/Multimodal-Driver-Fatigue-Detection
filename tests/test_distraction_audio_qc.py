"""Tests for the DCPT audio QC audit (numpy + stdlib only)."""

from __future__ import annotations

import pytest

from driver_state.preprocessing.distraction_audio.qc import (
    CODE_CHANNELS_NOT_STEREO,
    CODE_LONGER_THAN_NOMINAL,
    CODE_MINOR_TAIL_SHORTFALL,
    CODE_POSSIBLE_CLIPPING,
    CODE_SAMPLE_RATE_DEVIATION,
    CODE_SAMPLE_WIDTH_NOT_16BIT,
    CODE_TAIL_SHORTFALL_EXCEEDS,
    CODE_ZERO_AUDIO,
    QcDecodeError,
    analyze_wav_bytes,
)

from wav_utils import make_wav_bytes

SID = "01_P01_20231111_09_31_43_12"


def test_normal_10s_stereo_16k_passes() -> None:
    data = make_wav_bytes(duration_s=10.0)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pass"
    assert qc.samplerate == 44100
    assert qc.channels == 2
    assert qc.sample_width_bytes == 2
    assert qc.measured_duration_s == pytest.approx(10.0, abs=1e-6)
    assert qc.tail_shortfall_s == pytest.approx(0.0, abs=1e-6)
    assert qc.raw_observed_fraction == pytest.approx(1.0)
    assert qc.flags == []
    assert qc.reason_codes == []


def test_minor_tail_shortfall_is_pass_with_flags() -> None:
    data = make_wav_bytes(duration_s=9.95)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pass_with_flags"
    assert CODE_MINOR_TAIL_SHORTFALL in qc.flags
    assert qc.reason_codes == []
    assert qc.tail_shortfall_s == pytest.approx(0.05, abs=1e-6)
    assert qc.raw_observed_fraction == pytest.approx(0.995, abs=1e-6)


def test_tail_shortfall_above_limit_is_pending() -> None:
    data = make_wav_bytes(duration_s=9.8)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pending"
    assert CODE_TAIL_SHORTFALL_EXCEEDS in qc.reason_codes
    assert qc.primary_error == CODE_TAIL_SHORTFALL_EXCEEDS


def test_zero_audio_is_pending_marker() -> None:
    data = make_wav_bytes(duration_s=10.0, style="zero")
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pending"
    assert CODE_ZERO_AUDIO in qc.reason_codes


def test_clipping_is_only_a_flag() -> None:
    data = make_wav_bytes(duration_s=10.0, style="clip")
    qc = analyze_wav_bytes(data, SID)
    assert CODE_POSSIBLE_CLIPPING in qc.flags
    assert qc.reason_codes == []
    assert qc.qc_status == "pass_with_flags"


def test_mono_is_pending() -> None:
    data = make_wav_bytes(duration_s=10.0, channels=1)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pending"
    assert CODE_CHANNELS_NOT_STEREO in qc.reason_codes


def test_8bit_is_pending() -> None:
    data = make_wav_bytes(duration_s=10.0, sampwidth=1)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pending"
    assert CODE_SAMPLE_WIDTH_NOT_16BIT in qc.reason_codes


def test_wrong_samplerate_is_pending() -> None:
    data = make_wav_bytes(duration_s=10.0, samplerate=22050)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pending"
    assert CODE_SAMPLE_RATE_DEVIATION in qc.reason_codes


def test_longer_than_nominal_is_pending() -> None:
    data = make_wav_bytes(duration_s=10.2)
    qc = analyze_wav_bytes(data, SID)
    assert qc.qc_status == "pending"
    assert CODE_LONGER_THAN_NOMINAL in qc.reason_codes


def test_undecodable_bytes_raise_stable_error() -> None:
    with pytest.raises(QcDecodeError) as excinfo:
        analyze_wav_bytes(b"this is not a wav file at all", SID)
    assert excinfo.value.code == "UNSUPPORTED_CONTAINER"