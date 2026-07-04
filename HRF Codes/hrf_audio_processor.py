# -*- coding: utf-8 -*-
"""
HRF Audio Processor: Multi-Format Audio Support for Harmonic Resonance Fields

Extends HRF algorithm to audio signal processing with support for multiple formats and codecs.
Provides unified interface for loading, processing, and analyzing audio data using HRF.

Supported Formats:
- WAV (PCM, float32, int16)
- MP3 (via librosa or pydub)
- FLAC (via librosa)
- OGG (via librosa)
- M4A/AAC (via librosa or pydub)

Author: GSSoC 2026 Contributors
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Union, List
from pathlib import Path

# Attempt to import audio libraries with graceful fallback
AUDIO_BACKENDS = {}

def _load_librosa():
    try:
        import librosa
        AUDIO_BACKENDS['librosa'] = librosa
        return True
    except ImportError:
        return False

def _load_soundfile():
    try:
        import soundfile as sf
        AUDIO_BACKENDS['soundfile'] = sf
        return True
    except ImportError:
        return False

def _load_pydub():
    try:
        from pydub import AudioSegment
        AUDIO_BACKENDS['pydub'] = AudioSegment
        return True
    except ImportError:
        return False

def _load_scipy():
    try:
        from scipy import signal as scipy_signal
        from scipy.io import wavfile
        AUDIO_BACKENDS['scipy'] = {'signal': scipy_signal, 'wavfile': wavfile}
        return True
    except ImportError:
        return False

# Initialize available backends
_load_librosa()
_load_soundfile()
_load_pydub()
_load_scipy()

if not AUDIO_BACKENDS:
    raise ImportError(
        "No audio libraries found. Install at least one: "
        "librosa, soundfile, pydub, or scipy"
    )


class AudioFormatHandler:
    """Handles loading and conversion of various audio formats."""

    SUPPORTED_FORMATS = {
        '.wav': 'WAV (PCM)',
        '.mp3': 'MP3',
        '.flac': 'FLAC',
        '.ogg': 'OGG Vorbis',
        '.m4a': 'M4A/AAC',
        '.aac': 'AAC',
    }

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Return list of supported audio formats."""
        return list(AudioFormatHandler.SUPPORTED_FORMATS.keys())

    @staticmethod
    def is_format_supported(file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in AudioFormatHandler.SUPPORTED_FORMATS

    @staticmethod
    def load_audio(
        file_path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file in any supported format.

        Parameters:
        -----------
        file_path : str or Path
            Path to audio file
        sr : int, optional
            Target sample rate. If None, uses native rate
        mono : bool
            Convert to mono if True (default: True)
        offset : float
            Start reading after this time (in seconds)
        duration : float, optional
            Only load this much audio (in seconds)

        Returns:
        --------
        y : np.ndarray
            Audio time series (mono: 1D, stereo: 2D)
        sr : int
            Sample rate
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if not AudioFormatHandler.is_format_supported(file_path):
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {AudioFormatHandler.get_supported_formats()}"
            )

        # Try librosa first (most reliable for all formats)
        if 'librosa' in AUDIO_BACKENDS:
            return AudioFormatHandler._load_with_librosa(
                file_path, sr, mono, offset, duration
            )

        # Format-specific loaders
        if ext == '.wav':
            if 'soundfile' in AUDIO_BACKENDS:
                return AudioFormatHandler._load_wav_soundfile(file_path, sr, mono)
            elif 'scipy' in AUDIO_BACKENDS:
                return AudioFormatHandler._load_wav_scipy(file_path, sr, mono)

        if ext in ['.mp3', '.m4a', '.aac']:
            if 'pydub' in AUDIO_BACKENDS:
                return AudioFormatHandler._load_with_pydub(file_path, sr, mono)

        raise RuntimeError(
            f"Cannot load {ext} format. "
            f"Please install librosa or format-specific library."
        )

    @staticmethod
    def _load_with_librosa(
        file_path: Path,
        sr: Optional[int],
        mono: bool,
        offset: float,
        duration: Optional[float],
    ) -> Tuple[np.ndarray, int]:
        """Load audio using librosa (supports all formats)."""
        librosa = AUDIO_BACKENDS['librosa']
        y, sr_loaded = librosa.load(
            str(file_path),
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration,
        )
        return y, sr_loaded

    @staticmethod
    def _load_wav_soundfile(
        file_path: Path,
        sr: Optional[int],
        mono: bool,
    ) -> Tuple[np.ndarray, int]:
        """Load WAV using soundfile."""
        sf = AUDIO_BACKENDS['soundfile']
        y, sr_loaded = sf.read(str(file_path), dtype=np.float32)

        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)

        if sr is not None and sr != sr_loaded:
            # Resample if needed
            if 'librosa' in AUDIO_BACKENDS:
                librosa = AUDIO_BACKENDS['librosa']
                y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr)
                sr_loaded = sr
            else:
                warnings.warn(
                    f"Target sample rate {sr} != file rate {sr_loaded}. "
                    f"Install librosa for resampling."
                )

        return y, sr_loaded

    @staticmethod
    def _load_wav_scipy(
        file_path: Path,
        sr: Optional[int],
        mono: bool,
    ) -> Tuple[np.ndarray, int]:
        """Load WAV using scipy."""
        scipy_backends = AUDIO_BACKENDS['scipy']
        sr_loaded, y = scipy_backends['wavfile'].read(str(file_path))

        # Normalize to [-1, 1] range
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0

        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)

        if sr is not None and sr != sr_loaded:
            warnings.warn(
                f"Target sample rate {sr} != file rate {sr_loaded}. "
                f"Install librosa for resampling."
            )

        return y, sr_loaded

    @staticmethod
    def _load_with_pydub(
        file_path: Path,
        sr: Optional[int],
        mono: bool,
    ) -> Tuple[np.ndarray, int]:
        """Load audio using pydub (MP3, M4A, AAC)."""
        AudioSegment = AUDIO_BACKENDS['pydub']

        # Determine format
        ext = file_path.suffix.lower()[1:]  # Remove leading dot
        if ext == 'm4a':
            ext = 'mp4'

        audio = AudioSegment.from_file(str(file_path), format=ext)
        sr_loaded = audio.frame_rate

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize based on sample width
        if audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0

        # Handle stereo/mono conversion
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            if mono:
                samples = np.mean(samples, axis=1)
        elif audio.channels != 1:
            warnings.warn(f"Unexpected channel count: {audio.channels}")

        # Resample if needed
        if sr is not None and sr != sr_loaded:
            if 'librosa' in AUDIO_BACKENDS:
                librosa = AUDIO_BACKENDS['librosa']
                samples = librosa.resample(
                    samples, orig_sr=sr_loaded, target_sr=sr
                )
                sr_loaded = sr
            else:
                warnings.warn(
                    f"Target sample rate {sr} != file rate {sr_loaded}. "
                    f"Install librosa for resampling."
                )

        return samples, sr_loaded

    @staticmethod
    def convert_format(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_sr: Optional[int] = None,
    ) -> bool:
        """
        Convert audio between formats.

        Parameters:
        -----------
        input_path : str or Path
            Input audio file
        output_path : str or Path
            Output audio file (format determined by extension)
        target_sr : int, optional
            Target sample rate

        Returns:
        --------
        success : bool
            True if conversion succeeded
        """
        try:
            output_path = Path(output_path)
            output_ext = output_path.suffix.lower()

            if not AudioFormatHandler.is_format_supported(output_ext):
                print(f"❌ Unsupported output format: {output_ext}")
                return False

            # Load audio
            y, sr = AudioFormatHandler.load_audio(input_path, sr=target_sr)

            # Save in target format
            if output_ext == '.wav':
                if 'soundfile' in AUDIO_BACKENDS:
                    AUDIO_BACKENDS['soundfile'].write(
                        str(output_path), y, sr, subtype='FLOAT'
                    )
                elif 'scipy' in AUDIO_BACKENDS:
                    # Normalize before saving
                    y_int16 = np.int16(y * 32767.0)
                    AUDIO_BACKENDS['scipy']['wavfile'].write(
                        str(output_path), sr, y_int16
                    )
                else:
                    print("❌ No WAV encoder available")
                    return False

            elif output_ext in ['.mp3', '.m4a']:
                if 'librosa' in AUDIO_BACKENDS:
                    # Use soundfile for export
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                        if 'soundfile' in AUDIO_BACKENDS:
                            AUDIO_BACKENDS['soundfile'].write(tmp.name, y, sr)
                        else:
                            print(f"❌ Cannot export to {output_ext}")
                            return False

                        # Convert WAV to target format
                        if 'pydub' in AUDIO_BACKENDS:
                            AudioSegment = AUDIO_BACKENDS['pydub']
                            audio = AudioSegment.from_wav(tmp.name)
                            fmt = 'mp4' if output_ext == '.m4a' else output_ext[1:]
                            audio.export(str(output_path), format=fmt)
                        else:
                            print(f"❌ Cannot export to {output_ext}")
                            return False
                else:
                    print(f"❌ No encoder for {output_ext}")
                    return False

            else:
                print(f"❌ Conversion to {output_ext} not yet implemented")
                return False

            print(f"✅ Successfully converted to {output_path}")
            return True

        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False


class HRFAudioProcessor:
    """Apply HRF algorithm to audio signals."""

    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        """
        Initialize audio processor.

        Parameters:
        -----------
        sr : int
            Target sample rate (default: 22050 Hz)
        n_mfcc : int
            Number of MFCC features to extract
        """
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_features(
        self, y: np.ndarray, sr: int
    ) -> np.ndarray:
        """
        Extract audio features suitable for HRF processing.

        Parameters:
        -----------
        y : np.ndarray
            Audio time series
        sr : int
            Sample rate

        Returns:
        --------
        features : np.ndarray
            Feature matrix for HRF
        """
        if 'librosa' not in AUDIO_BACKENDS:
            raise RuntimeError(
                "librosa required for feature extraction. "
                "Install it with: pip install librosa"
            )

        librosa = AUDIO_BACKENDS['librosa']

        # Extract MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        # Stack features
        features = np.vstack([
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
        ])

        return features.T  # Transpose to (n_samples, n_features)

    def process_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load and process audio file for HRF.

        Parameters:
        -----------
        file_path : str or Path
            Path to audio file

        Returns:
        --------
        features : np.ndarray
            Processed features for HRF
        sr : int
            Sample rate
        """
        # Load audio in supported format
        y, sr = AudioFormatHandler.load_audio(
            file_path, sr=self.sr, mono=True
        )

        # Extract features
        features = self.extract_features(y, sr)

        return features, sr


# Example usage and testing
if __name__ == "__main__":
    print("🎵 HRF Audio Processor - Format Support Test")
    print(f"Available backends: {list(AUDIO_BACKENDS.keys())}")
    print(f"Supported formats: {AudioFormatHandler.get_supported_formats()}")
    print("\n✅ Audio format support module loaded successfully")
