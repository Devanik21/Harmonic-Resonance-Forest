"""
audio_io.py

Multi-format audio loading, saving, conversion, and resampling for the
Harmonic Resonance Forest (HRF) pipeline.

HRF treats classification as a resonance problem, and audio is one of its
primary harmonic time-series domains (speech, EEG sonification, conference
recordings). Until now only uncompressed WAV files could be fed into the
pipeline. This module adds a single entry point, ``load_audio``, that accepts
WAV out of the box (stdlib only) and FLAC / OGG / MP3 through optional
backends, always returning the same normalized representation:

    samples : np.ndarray, float32 in [-1.0, 1.0], shape (n,) mono
              or (n, channels)
    sample_rate : int

Backend strategy
----------------
* WAV      : Python stdlib ``wave`` + numpy. Zero extra dependencies.
             Supports 8/16/24/32-bit PCM and 32-bit float.
* FLAC/OGG : ``soundfile`` (libsndfile) when installed.
* MP3      : ``soundfile`` >= 0.12 (libsndfile >= 1.1) when available,
             otherwise ``librosa`` (audioread/ffmpeg) as a fallback.

Optional backends are probed lazily; a missing backend raises
``UnsupportedFormatError`` with the exact pip command needed, instead of an
opaque ImportError deep inside a training run.

Example
-------
>>> from audio_io import load_audio, save_audio, convert_format
>>> samples, sr = load_audio("recording.flac", sample_rate=16000, mono=True)
>>> save_audio("recording_16k.wav", samples, sr)
>>> convert_format("recording.flac", "recording.wav")
"""

from __future__ import annotations

import os
import struct
import wave

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "AudioIOError",
    "UnsupportedFormatError",
    "SUPPORTED_FORMATS",
    "available_formats",
    "detect_format",
    "load_audio",
    "save_audio",
    "convert_format",
    "resample_audio",
    "to_mono",
]


class AudioIOError(Exception):
    """Base error for audio reading and writing failures."""


class UnsupportedFormatError(AudioIOError):
    """Raised when a file's format has no available backend."""


#: Formats this module knows about and the backend that serves each one.
SUPPORTED_FORMATS = {
    "wav": "stdlib",
    "flac": "soundfile",
    "ogg": "soundfile",
    "mp3": "soundfile|librosa",
}

# Magic byte signatures for content-based format detection. Extension alone
# is unreliable for files exported by third-party tools.
_MAGIC_SIGNATURES = (
    (b"RIFF", "wav"),
    (b"fLaC", "flac"),
    (b"OggS", "ogg"),
    (b"ID3", "mp3"),
)


def _try_import_soundfile():
    try:
        import soundfile  # noqa: PLC0415

        return soundfile
    except (ImportError, OSError):
        return None


def _try_import_librosa():
    try:
        import librosa  # noqa: PLC0415

        return librosa
    except (ImportError, OSError):
        return None


def available_formats():
    """Return the subset of SUPPORTED_FORMATS usable in this environment.

    Returns
    -------
    dict
        Mapping of format name to the backend that will actually serve it.
    """
    formats = {"wav": "stdlib"}
    sf = _try_import_soundfile()
    if sf is not None:
        formats["flac"] = "soundfile"
        formats["ogg"] = "soundfile"
        if "MP3" in getattr(sf, "available_formats", lambda: {})():
            formats["mp3"] = "soundfile"
    if "mp3" not in formats and _try_import_librosa() is not None:
        formats["mp3"] = "librosa"
    return formats


def detect_format(path):
    """Identify an audio file's format from its magic bytes.

    Falls back to the file extension when the header is inconclusive
    (raw MP3 frames without an ID3 tag, for example).

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    str
        One of the SUPPORTED_FORMATS keys.

    Raises
    ------
    UnsupportedFormatError
        When neither the header nor the extension identify a known format.
    """
    with open(path, "rb") as fh:
        header = fh.read(12)

    for signature, fmt in _MAGIC_SIGNATURES:
        if header.startswith(signature):
            # RIFF is shared by several container types; confirm WAVE.
            if fmt == "wav" and header[8:12] != b"WAVE":
                continue
            return fmt

    # MP3 files without an ID3 tag start with an 0xFFEx/0xFFFx frame sync.
    if len(header) >= 2:
        sync = struct.unpack(">H", header[:2])[0]
        if (sync & 0xFFE0) == 0xFFE0:
            return "mp3"

    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext in SUPPORTED_FORMATS:
        return ext

    raise UnsupportedFormatError(
        f"Could not identify the format of '{path}'. "
        f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
    )


def load_audio(path, sample_rate=None, mono=False, dtype=np.float32):
    """Load an audio file of any supported format.

    Parameters
    ----------
    path : str
        Path to a WAV, FLAC, OGG, or MP3 file.
    sample_rate : int, optional
        Target sample rate. When given and different from the file's native
        rate, the signal is resampled with a polyphase filter.
    mono : bool
        When True, multi-channel audio is averaged down to one channel.
    dtype : numpy dtype
        Floating point dtype of the returned array.

    Returns
    -------
    (np.ndarray, int)
        Normalized samples in [-1.0, 1.0] and the (possibly resampled)
        sample rate. Shape is (n,) for mono and (n, channels) otherwise.

    Raises
    ------
    UnsupportedFormatError
        When the format needs a backend that is not installed.
    AudioIOError
        When the file exists but cannot be decoded.
    """
    fmt = detect_format(path)

    if fmt == "wav":
        samples, native_sr = _load_wav(path)
    else:
        samples, native_sr = _load_with_backend(path, fmt)

    samples = samples.astype(dtype, copy=False)

    if mono:
        samples = to_mono(samples)

    if sample_rate is not None and sample_rate != native_sr:
        samples = resample_audio(samples, native_sr, sample_rate)
        native_sr = sample_rate

    return samples, native_sr


def save_audio(path, samples, sample_rate, subtype="PCM_16"):
    """Write samples to disk; the format follows the file extension.

    WAV output uses the stdlib writer and needs no optional dependencies.
    FLAC and OGG require ``soundfile``.

    Parameters
    ----------
    path : str
        Destination path ending in .wav, .flac, or .ogg.
    samples : np.ndarray
        Float samples in [-1.0, 1.0], shape (n,) or (n, channels).
    sample_rate : int
        Sample rate in Hz.
    subtype : str
        Encoding subtype for soundfile-backed formats (ignored for WAV,
        which always writes 16-bit PCM).
    """
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    samples = np.asarray(samples)

    if ext == "wav":
        _save_wav(path, samples, sample_rate)
        return

    sf = _try_import_soundfile()
    if sf is None:
        raise UnsupportedFormatError(
            f"Writing '.{ext}' requires the soundfile backend. "
            "Install it with: pip install soundfile"
        )
    try:
        sf.write(path, samples, sample_rate, subtype=subtype)
    except Exception as exc:  # noqa: BLE001
        raise AudioIOError(f"Failed to write '{path}': {exc}") from exc


def convert_format(src_path, dst_path, sample_rate=None, mono=False):
    """Convert an audio file between any two supported formats.

    Parameters
    ----------
    src_path : str
        Input file in any supported format.
    dst_path : str
        Output path; the extension selects the output format.
    sample_rate : int, optional
        Resample during conversion when provided.
    mono : bool
        Downmix to mono during conversion when True.

    Returns
    -------
    (np.ndarray, int)
        The converted samples and sample rate, for immediate reuse.
    """
    samples, sr = load_audio(src_path, sample_rate=sample_rate, mono=mono)
    save_audio(dst_path, samples, sr)
    return samples, sr


def resample_audio(samples, orig_sr, target_sr):
    """Resample with scipy's polyphase filter (already a core HRF dependency).

    Polyphase resampling preserves the harmonic structure HRF's resonance
    kernels depend on far better than naive interpolation.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n,) or (n, channels).
    orig_sr, target_sr : int
        Source and destination sample rates in Hz.

    Returns
    -------
    np.ndarray
        Resampled signal with the same dtype and channel layout.
    """
    if orig_sr == target_sr:
        return samples
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers.")
    gcd = np.gcd(int(orig_sr), int(target_sr))
    up, down = int(target_sr) // gcd, int(orig_sr) // gcd
    out = resample_poly(samples, up, down, axis=0)
    return out.astype(samples.dtype, copy=False)


def to_mono(samples):
    """Average a (n, channels) signal down to a (n,) mono signal."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1).astype(samples.dtype, copy=False)


# ---------------------------------------------------------------------------
# WAV backend (stdlib)
# ---------------------------------------------------------------------------

def _load_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioIOError(f"Failed to decode WAV file '{path}': {exc}") from exc

    if sampwidth == 1:
        # 8-bit WAV is unsigned
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        data = _decode_pcm24(raw).astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        # 32-bit WAV is either int PCM or IEEE float; floats are already
        # normalized, so distinguish by value range after both decodings.
        as_float = np.frombuffer(raw, dtype="<f4")
        if np.all(np.isfinite(as_float)) and as_float.size and np.abs(as_float).max() <= 64.0:
            data = as_float.astype(np.float32)
        else:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioIOError(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data, sample_rate


def _decode_pcm24(raw):
    """Decode little-endian signed 24-bit PCM into int32."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    out = (
        b[:, 0].astype(np.int32)
        | (b[:, 1].astype(np.int32) << 8)
        | (b[:, 2].astype(np.int32) << 16)
    )
    # Sign-extend from bit 23
    out[out >= 1 << 23] -= 1 << 24
    return out


def _save_wav(path, samples, sample_rate):
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    n_channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Optional backends (soundfile / librosa)
# ---------------------------------------------------------------------------

def _load_with_backend(path, fmt):
    sf = _try_import_soundfile()
    if sf is not None:
        try:
            samples, sr = sf.read(path, dtype="float32", always_2d=False)
            return samples, sr
        except Exception as exc:  # noqa: BLE001
            if fmt != "mp3":
                raise AudioIOError(f"Failed to decode '{path}': {exc}") from exc
            # Older libsndfile builds lack MP3; fall through to librosa.

    if fmt == "mp3":
        librosa = _try_import_librosa()
        if librosa is not None:
            try:
                samples, sr = librosa.load(path, sr=None, mono=False)
                if samples.ndim == 2:  # librosa returns (channels, n)
                    samples = samples.T
                return samples.astype(np.float32), int(sr)
            except Exception as exc:  # noqa: BLE001
                raise AudioIOError(f"Failed to decode '{path}': {exc}") from exc

    raise UnsupportedFormatError(
        f"Loading '.{fmt}' files requires an optional backend. "
        "Install one with: pip install soundfile"
        + ("  (or: pip install librosa)" if fmt == "mp3" else "")
    )
