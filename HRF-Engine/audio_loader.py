import os
import numpy as np
import librosa

SUPPORTED_FORMATS = [
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aac"
]

def load_audio(file_path):
    """
    Load audio files in multiple formats.

    Parameters
    ----------
    file_path : str
        Path to audio file

    Returns
    -------
    audio : np.ndarray
    sample_rate : int
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {extension}"
        )

    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    return audio, sample_rate