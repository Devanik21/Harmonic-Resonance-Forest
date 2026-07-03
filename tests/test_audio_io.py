"""
tests/test_audio_io.py

Unit tests for the multi-format audio I/O module. The WAV path is exercised
fully with stdlib-only fixtures; FLAC/OGG round trips run only when the
optional soundfile backend is installed.

Run from repo root: pytest tests/test_audio_io.py -v
"""

import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_io import (  # noqa: E402
    AudioIOError,
    UnsupportedFormatError,
    available_formats,
    convert_format,
    detect_format,
    load_audio,
    resample_audio,
    save_audio,
    to_mono,
)

SR = 22050


def sine(freq=440.0, duration=0.25, sr=SR, channels=1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave_ = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 1:
        return wave_
    return np.stack([wave_] * channels, axis=1)


def write_wav(path, samples, sr=SR, sampwidth=2):
    """Write PCM WAV at an arbitrary bit depth for decoder tests."""
    samples = np.atleast_2d(samples.T).T
    n_channels = samples.shape[1]
    scale = float(1 << (8 * sampwidth - 1))
    ints = (np.clip(samples, -1, 1) * (scale - 1)).astype(np.int64)
    if sampwidth == 3:
        frames = b"".join(
            struct.pack("<i", v)[:3] for v in ints.flatten()
        )
    else:
        dt = {1: np.uint8, 2: "<i2", 4: "<i4"}[sampwidth]
        if sampwidth == 1:
            ints = ints + 128  # 8-bit WAV is unsigned
        frames = ints.astype(dt).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(frames)


class TestWavRoundTrip:
    def test_mono_roundtrip(self, tmp_path):
        original = sine()
        p = tmp_path / "mono.wav"
        save_audio(str(p), original, SR)
        loaded, sr = load_audio(str(p))
        assert sr == SR
        assert loaded.shape == original.shape
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, original, atol=1e-3)

    def test_stereo_roundtrip(self, tmp_path):
        original = sine(channels=2)
        p = tmp_path / "stereo.wav"
        save_audio(str(p), original, SR)
        loaded, sr = load_audio(str(p))
        assert loaded.shape == original.shape
        np.testing.assert_allclose(loaded, original, atol=1e-3)

    @pytest.mark.parametrize("sampwidth", [1, 2, 3, 4])
    def test_bit_depths(self, tmp_path, sampwidth):
        original = sine()
        p = tmp_path / f"pcm{sampwidth * 8}.wav"
        write_wav(p, original, sampwidth=sampwidth)
        loaded, sr = load_audio(str(p))
        assert sr == SR
        tol = {1: 2e-2, 2: 1e-3, 3: 1e-5, 4: 1e-7}[sampwidth]
        np.testing.assert_allclose(loaded, original, atol=tol)

    def test_normalized_range(self, tmp_path):
        p = tmp_path / "loud.wav"
        save_audio(str(p), sine() * 4.0, SR)  # clipped on save
        loaded, _ = load_audio(str(p))
        assert np.abs(loaded).max() <= 1.0


class TestDetectFormat:
    def test_wav_magic(self, tmp_path):
        p = tmp_path / "renamed.bin"
        save_audio(str(tmp_path / "a.wav"), sine(), SR)
        (tmp_path / "a.wav").rename(p)
        assert detect_format(str(p)) == "wav"

    def test_flac_magic(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"fLaC" + b"\x00" * 32)
        assert detect_format(str(p)) == "flac"

    def test_ogg_magic(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"OggS" + b"\x00" * 32)
        assert detect_format(str(p)) == "ogg"

    def test_mp3_id3_magic(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"ID3" + b"\x00" * 32)
        assert detect_format(str(p)) == "mp3"

    def test_mp3_frame_sync(self, tmp_path):
        p = tmp_path / "x.dat"
        p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 32)
        assert detect_format(str(p)) == "mp3"

    def test_unknown_raises(self, tmp_path):
        p = tmp_path / "mystery.xyz"
        p.write_bytes(b"NOPE" + b"\x00" * 32)
        with pytest.raises(UnsupportedFormatError):
            detect_format(str(p))


class TestTransforms:
    def test_resample_halves_length(self):
        original = sine(sr=SR)
        out = resample_audio(original, SR, SR // 2)
        assert abs(len(out) - len(original) // 2) <= 1
        assert out.dtype == original.dtype

    def test_resample_identity(self):
        original = sine()
        assert resample_audio(original, SR, SR) is original

    def test_resample_preserves_pitch(self):
        # A 440 Hz tone must still peak at 440 Hz after resampling
        original = sine(freq=440.0, duration=1.0)
        out = resample_audio(original, SR, 16000)
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(len(out), 1 / 16000)[np.argmax(spectrum)]
        assert abs(peak_hz - 440.0) < 5.0

    def test_resample_rejects_bad_rates(self):
        with pytest.raises(ValueError):
            resample_audio(sine(), 0, SR)

    def test_to_mono(self):
        stereo = sine(channels=2)
        mono = to_mono(stereo)
        assert mono.ndim == 1
        np.testing.assert_allclose(mono, stereo[:, 0], atol=1e-6)

    def test_to_mono_passthrough(self):
        mono = sine()
        assert to_mono(mono) is mono

    def test_load_with_target_sr(self, tmp_path):
        p = tmp_path / "a.wav"
        save_audio(str(p), sine(), SR)
        loaded, sr = load_audio(str(p), sample_rate=16000)
        assert sr == 16000
        assert abs(len(loaded) - int(0.25 * 16000)) <= 1

    def test_load_mono_flag(self, tmp_path):
        p = tmp_path / "st.wav"
        save_audio(str(p), sine(channels=2), SR)
        loaded, _ = load_audio(str(p), mono=True)
        assert loaded.ndim == 1


class TestConvert:
    def test_wav_to_wav_with_resample(self, tmp_path):
        src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
        save_audio(str(src), sine(), SR)
        samples, sr = convert_format(str(src), str(dst), sample_rate=8000, mono=True)
        assert sr == 8000
        loaded, sr2 = load_audio(str(dst))
        assert sr2 == 8000
        assert loaded.ndim == 1

    def test_corrupt_wav_raises(self, tmp_path):
        p = tmp_path / "bad.wav"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
        with pytest.raises(AudioIOError):
            load_audio(str(p))


class TestOptionalBackends:
    def test_wav_always_available(self):
        assert available_formats()["wav"] == "stdlib"

    def test_missing_backend_message_is_actionable(self, tmp_path):
        pytest.importorskip
        p = tmp_path / "x.flac"
        p.write_bytes(b"fLaC" + b"\x00" * 64)
        try:
            import soundfile  # noqa: F401, PLC0415
            pytest.skip("soundfile installed; missing-backend path not reachable")
        except ImportError:
            pass
        with pytest.raises(UnsupportedFormatError, match="pip install soundfile"):
            load_audio(str(p))

    @pytest.mark.skipif(
        not pytest.importorskip("importlib.util").find_spec("soundfile"),
        reason="soundfile not installed",
    )
    def test_flac_roundtrip(self, tmp_path):
        original = sine()
        p = tmp_path / "a.flac"
        save_audio(str(p), original, SR)
        loaded, sr = load_audio(str(p))
        assert sr == SR
        np.testing.assert_allclose(loaded, original, atol=1e-3)
