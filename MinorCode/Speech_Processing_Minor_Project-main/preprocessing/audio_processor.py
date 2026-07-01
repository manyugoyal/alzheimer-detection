"""
audio_processor.py
==================
Handles all audio-level preprocessing for the Speech Processing Branch:

  1. MP3 → WAV conversion using pydub
  2. Amplitude normalisation
  3. Resampling to 16kHz mono using librosa
  4. Log-Mel spectrogram computation (librosa)
  5. Long-audio chunking strategy for HuBERT inference
  6. eGeMAPS feature extraction via opensmile

This module does NOT run the HuBERT forward pass — that lives in
models/acoustic_encoder.py. This module is solely responsible for
producing the correctly formatted waveform tensor that the encoder expects.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional-import guard for heavy dependencies
# ---------------------------------------------------------------------------
try:
    import librosa
    import librosa.feature
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed. Audio processing will be unavailable.")

try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False
    logger.warning("pydub not installed. MP3 conversion will be unavailable.")

try:
    import opensmile
    _OPENSMILE_AVAILABLE = True
except ImportError:
    _OPENSMILE_AVAILABLE = False
    logger.warning("opensmile not installed. eGeMAPS extraction will be unavailable.")


# ---------------------------------------------------------------------------
# MP3 → WAV conversion
# ---------------------------------------------------------------------------

def convert_mp3_to_wav(mp3_path: str, wav_path: Optional[str] = None) -> str:
    """
    Convert an MP3 file to a WAV file using pydub.

    If ``wav_path`` is not specified, a temporary WAV file is created in the
    system's temp directory and its path is returned.

    Parameters
    ----------
    mp3_path : str
        Path to the input .mp3 file.
    wav_path : str, optional
        Path where the output .wav file should be written.
        If None, a temporary file is used.

    Returns
    -------
    str
        Path to the resulting .wav file.

    Raises
    ------
    ImportError
        If pydub is not installed.
    FileNotFoundError
        If the mp3 file does not exist.
    """
    if not _PYDUB_AVAILABLE:
        raise ImportError(
            "pydub is required for MP3-to-WAV conversion. "
            "Install it with: pip install pydub"
        )

    mp3_path = Path(mp3_path)
    if not mp3_path.exists():
        raise FileNotFoundError(f"Audio file not found: {mp3_path}")

    if wav_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

    logger.debug("Converting %s → %s", mp3_path, wav_path)
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio.export(wav_path, format="wav")
    return wav_path


# ---------------------------------------------------------------------------
# Waveform loading and normalisation
# ---------------------------------------------------------------------------

def load_waveform(
    audio_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file (WAV or MP3) as a float32 NumPy array resampled to
    ``target_sr`` Hz, with optional amplitude normalisation.

    For MP3 inputs, pydub is used to convert to a temporary WAV file first,
    then librosa loads it.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav or .mp3).
    target_sr : int
        Target sampling rate in Hz. Default is 16000 (required by HuBERT).
    mono : bool
        If True, convert stereo/multichannel audio to mono.
    normalize : bool
        If True, normalise the waveform amplitude to the range [-1, 1].

    Returns
    -------
    waveform : np.ndarray of shape (num_samples,)
        Float32 waveform array.
    sr : int
        Actual sampling rate after resampling (= ``target_sr``).

    Raises
    ------
    ImportError
        If librosa is not installed.
    FileNotFoundError
        If the audio file does not exist.
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for waveform loading. "
            "Install it with: pip install librosa"
        )

    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Auto-convert MP3 to temp WAV if needed
    tmp_wav_path: Optional[str] = None
    if audio_path.lower().endswith(".mp3"):
        tmp_wav_path = convert_mp3_to_wav(audio_path)
        load_path = tmp_wav_path
    else:
        load_path = audio_path

    try:
        # librosa.load returns (waveform, sr) with waveform in float32
        waveform, sr = librosa.load(
            load_path,
            sr=target_sr,     # Resample on load
            mono=mono,
            dtype=np.float32,
        )
    finally:
        # Clean up temp WAV if we created one
        if tmp_wav_path is not None and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    # Normalise amplitude to [-1, 1]
    if normalize:
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

    logger.debug(
        "Loaded audio: %s | samples=%d | sr=%d | duration=%.2fs",
        os.path.basename(audio_path),
        len(waveform),
        sr,
        len(waveform) / sr,
    )
    return waveform, sr


# ---------------------------------------------------------------------------
# Log-Mel Spectrogram
# ---------------------------------------------------------------------------

def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Compute a log-scale Mel spectrogram from a 1-D float32 waveform.

    Steps
    -----
    1. Compute Mel spectrogram using ``librosa.feature.melspectrogram``.
    2. Convert power spectrogram to decibels with ``librosa.power_to_db``.

    Parameters
    ----------
    waveform : np.ndarray, shape (T,)
        Input waveform in float32.
    sample_rate : int
        Sample rate of the waveform in Hz.
    n_mels : int
        Number of Mel filter banks.
    n_fft : int
        FFT window size.
    hop_length : int
        Number of samples between STFT frames.
    fmax : float, optional
        Maximum frequency for the Mel filter bank. Defaults to ``sr / 2``.

    Returns
    -------
    np.ndarray, shape (n_mels, time_frames)
        Log-Mel spectrogram in dB scale.
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for spectrogram computation.")

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    logger.debug(
        "Log-Mel spectrogram shape: %s (n_mels=%d, frames=%d)",
        log_mel_spec.shape,
        log_mel_spec.shape[0],
        log_mel_spec.shape[1],
    )
    return log_mel_spec


# ---------------------------------------------------------------------------
# Long-audio chunking for HuBERT
# ---------------------------------------------------------------------------

def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    chunk_size_seconds: int = 30,
) -> List[np.ndarray]:
    """
    Split a long waveform into fixed-size chunks for memory-efficient
    HuBERT inference.

    HuBERT can be slow or OOM on very long recordings (>3 min) when run on CPU.
    This function slices the waveform into non-overlapping chunks of
    ``chunk_size_seconds`` seconds. The caller is responsible for averaging
    the resulting embeddings.

    Parameters
    ----------
    waveform : np.ndarray, shape (T,)
        Input waveform in float32.
    sample_rate : int
        Sample rate in Hz.
    chunk_size_seconds : int
        Duration of each chunk in seconds.

    Returns
    -------
    List[np.ndarray]
        List of waveform chunks, each of length ≤ chunk_size_seconds * sample_rate.
        The last chunk may be shorter if the audio length is not evenly divisible.
    """
    chunk_size_samples = chunk_size_seconds * sample_rate
    total_samples = len(waveform)
    chunks: List[np.ndarray] = []

    start = 0
    while start < total_samples:
        end = min(start + chunk_size_samples, total_samples)
        chunk = waveform[start:end]
        chunks.append(chunk)
        start += chunk_size_samples

    logger.debug(
        "Chunked audio: total=%d samples → %d chunk(s) of ~%ds each",
        total_samples,
        len(chunks),
        chunk_size_seconds,
    )
    return chunks


# ---------------------------------------------------------------------------
# eGeMAPS feature extraction
# ---------------------------------------------------------------------------

def extract_egemaps_features(
    audio_path: str,
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
) -> np.ndarray:
    """
    Extract eGeMAPS acoustic features from an audio file using opensmile.

    The eGeMAPS v02 feature set contains 88 functional features covering
    frequency, energy, temporal, and spectral descriptors relevant to
    voice quality and speaking style — validated for clinical speech analysis.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.wav or .mp3).
    feature_set : str
        opensmile feature set identifier. Default: 'eGeMAPSv02'.
    feature_level : str
        Feature level: 'Functionals' or 'LowLevelDescriptors'.
        'Functionals' returns one vector per file (88-dim); recommended.

    Returns
    -------
    np.ndarray, shape (88,)
        eGeMAPS feature vector as float32.

    Raises
    ------
    ImportError
        If opensmile is not installed.
    """
    if not _OPENSMILE_AVAILABLE:
        raise ImportError(
            "opensmile is required for eGeMAPS extraction. "
            "Install it with: pip install opensmile"
        )

    # Resolve opensmile enum values
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel[feature_level],
        )
    except KeyError:
        # Fallback: try direct string passing (older opensmile versions)
        smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=feature_level,
        )

    # For MP3, convert first
    tmp_wav_path: Optional[str] = None
    if str(audio_path).lower().endswith(".mp3"):
        tmp_wav_path = convert_mp3_to_wav(audio_path)
        process_path = tmp_wav_path
    else:
        process_path = audio_path

    try:
        features_df = smile.process_file(process_path)
    finally:
        if tmp_wav_path is not None and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    features_array = features_df.values.flatten().astype(np.float32)

    if features_array.shape[0] == 0:
        logger.warning("opensmile returned empty features for: %s", audio_path)
        features_array = np.zeros(88, dtype=np.float32)

    logger.debug(
        "eGeMAPS features extracted: shape=%s from %s",
        features_array.shape,
        os.path.basename(audio_path),
    )
    return features_array


# ---------------------------------------------------------------------------
# End-to-end waveform preparation (used by the Dataset)
# ---------------------------------------------------------------------------

def prepare_waveform_for_hubert(
    audio_path: str,
    target_sr: int = 16000,
    chunk_size_seconds: int = 30,
) -> Tuple[List[np.ndarray], int]:
    """
    Load, normalise, and chunk an audio file into 30-second waveform segments
    ready for HuBERT inference.

    This is the main entry point called by the Dataset class and inference.py.

    Parameters
    ----------
    audio_path : str
        Path to the .wav or .mp3 audio file.
    target_sr : int
        Target sampling rate (must be 16000 for HuBERT).
    chunk_size_seconds : int
        Duration of each chunk in seconds for long-audio processing.

    Returns
    -------
    chunks : List[np.ndarray]
        List of float32 waveform chunks.
    sr : int
        Sample rate (= target_sr after resampling).
    """
    waveform, sr = load_waveform(
        audio_path,
        target_sr=target_sr,
        mono=True,
        normalize=True,
    )
    chunks = chunk_audio(waveform, sample_rate=sr, chunk_size_seconds=chunk_size_seconds)
    return chunks, sr
