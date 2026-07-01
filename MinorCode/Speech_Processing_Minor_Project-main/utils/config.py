"""
config.py
=========
Central configuration for the Speech Processing Branch of the AD detection pipeline.
All hyperparameters, model names, and file paths are consolidated here as a dataclass
to ensure reproducibility and easy modification.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class AudioConfig:
    """Configuration for audio processing pipeline."""

    # Sampling rate expected by HuBERT (16kHz)
    sample_rate: int = 16000

    # Log-Mel Spectrogram parameters
    n_mels: int = 128
    hop_length: int = 512
    n_fft: int = 1024
    fmax: Optional[float] = None  # None uses sample_rate / 2

    # Chunking for long audio (in seconds)
    chunk_size_seconds: int = 30

    # HuBERT model identifier from HuggingFace Hub
    hubert_model_name: str = "facebook/hubert-base-ls960"

    # Dimensionality of HuBERT hidden states
    hubert_hidden_dim: int = 768

    # eGeMAPS feature set name for opensmile
    opensmile_feature_set: str = "eGeMAPSv02"
    opensmile_feature_level: str = "Functionals"

    # Number of eGeMAPS features
    egemaps_dim: int = 88


@dataclass
class LinguisticConfig:
    """Configuration for linguistic processing pipeline."""

    # BERT model identifier from HuggingFace Hub
    bert_model_name: str = "bert-base-uncased"

    # Tokenizer settings
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True

    # Dimensionality of BERT [CLS] embedding
    bert_hidden_dim: int = 768

    # Filler words to count in linguistic features
    filler_words: List[str] = field(
        default_factory=lambda: ["uh", "um", "er", "hmm", "hm", "ah"]
    )


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Common projection dimension for cross-modal attention
    d_model: int = 256

    # Number of attention heads in MultiheadAttention
    num_heads: int = 4

    # Fused embedding dimension after concatenation of two attended vectors
    fused_dim: int = 512  # 256 + 256

    # SDS Head intermediate dimensions
    sds_hidden_1: int = 128
    sds_hidden_2: int = 32
    sds_dropout: float = 0.3

    # Whether to freeze HuBERT encoder weights
    freeze_hubert: bool = True

    # Whether to freeze BERT encoder weights
    freeze_bert: bool = True


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    # Scheduler: CosineAnnealingLR
    scheduler_t_max: int = 30  # Max epochs for cosine cycle

    # Training loop
    num_epochs: int = 30
    batch_size: int = 16
    early_stopping_patience: int = 5

    # Data split ratios (must sum to 1.0)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Number of DataLoader worker processes
    num_workers: int = 0  # Set to 0 on Windows to avoid multiprocessing issues

    # Metric used to select best checkpoint
    best_metric: str = "auc"  # Options: 'auc', 'f1', 'accuracy'

    # Gradient clipping max norm (None = disabled)
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class PathConfig:
    """File system paths for data loading and checkpointing."""

    # Root directory for raw dataset
    data_root: str = "data/DementiaBank"

    # Subdirectory for .wav audio files
    audio_dir: str = "data/DementiaBank/audio"

    # Subdirectory for .cha transcript files
    transcript_dir: str = "data/DementiaBank/transcripts"

    # Directory where model checkpoints are saved
    checkpoint_dir: str = "checkpoints"

    # Filename template for the best model checkpoint
    best_checkpoint_name: str = "best_speech_branch.pt"

    # Log file path
    log_file: str = "logs/speech_branch_training.log"

    # Path to the pre-computed feature cache (for faster re-training)
    feature_cache_dir: str = "cache/features"

    def get_best_checkpoint_path(self) -> str:
        """Return full path to the best checkpoint file."""
        return os.path.join(self.checkpoint_dir, self.best_checkpoint_name)


@dataclass
class ReproducibilityConfig:
    """Seeds and determinism settings."""

    seed: int = 42
    # Set True to enable cuDNN deterministic mode (slower but fully reproducible)
    deterministic: bool = True
    benchmark: bool = False  # torch.backends.cudnn.benchmark


@dataclass
class Config:
    """
    Master configuration object that aggregates all sub-configs.

    Usage
    -----
    from utils.config import Config
    cfg = Config()
    print(cfg.audio.sample_rate)     # 16000
    print(cfg.model.d_model)         # 256
    print(cfg.training.batch_size)   # 16
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    linguistic: LinguisticConfig = field(default_factory=LinguisticConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    repro: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    # Device string — will be overridden by helpers.get_device()
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Create necessary directories after initialisation."""
        os.makedirs(self.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.paths.log_file), exist_ok=True)
        os.makedirs(self.paths.feature_cache_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Default singleton — importable directly for convenience
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = Config()
