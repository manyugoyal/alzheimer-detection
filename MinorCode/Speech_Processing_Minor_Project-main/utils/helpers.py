"""
helpers.py
==========
Utility functions for the Speech Processing Branch:
  - Seed setting for full reproducibility
  - Device selection (CUDA / MPS / CPU)
  - Python logging setup
  - Checkpoint saving / loading
  - Training-curve summary printer
"""

import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set all random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The random seed value to apply everywhere.
    deterministic : bool
        If True, enables cuDNN deterministic mode (slightly slower).
    benchmark : bool
        If True, enables cuDNN benchmark mode (faster on fixed input sizes,
        but non-deterministic). Should be False when ``deterministic=True``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.benchmark = benchmark

    # Also set the environment variable used by some CUDA operations
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Automatically select the best available compute device.

    Priority order: CUDA GPU → Apple MPS → CPU.

    Returns
    -------
    torch.device
        The selected device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logging.getLogger(__name__).info("Using CUDA GPU: %s", gpu_name)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.getLogger(__name__).info("Using Apple MPS (Metal) device.")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).info("CUDA/MPS not available — using CPU.")
    return device


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure the root logger to write to both stdout and an optional file.

    Parameters
    ----------
    log_file : str, optional
        Path to a log file. If None, logs only to stdout.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    fmt : str
        Log message format string.
    datefmt : str
        Date format string for log timestamps.

    Returns
    -------
    logging.Logger
        The configured root logger instance.
    """
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplication on re-initialisation
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: str,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Save a model checkpoint to disk.

    The checkpoint dictionary stores:
      - model state dict
      - optimiser state dict
      - scheduler state dict (if provided)
      - current epoch
      - best validation metrics

    Parameters
    ----------
    model : nn.Module
        The PyTorch model whose weights are to be saved.
    optimizer : torch.optim.Optimizer
        The optimiser state (contains momentum buffers, etc.).
    epoch : int
        The epoch number at which this checkpoint was saved.
    metrics : dict
        Dictionary of evaluation metrics (e.g., {'auc': 0.91, 'f1': 0.88}).
    checkpoint_path : str
        Full path where the .pt file will be written.
    scheduler : optional
        Learning-rate scheduler. Its state dict is saved if provided.
    """
    logger = logging.getLogger(__name__)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    logger.info(
        "Checkpoint saved → %s  (epoch=%d, %s)",
        checkpoint_path,
        epoch,
        ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
    )


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk and restore model (and optionally optimiser/scheduler) state.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved .pt checkpoint file.
    model : nn.Module
        The model instance into which weights will be loaded.
    optimizer : optional
        If provided, the optimiser state will also be restored.
    scheduler : optional
        If provided, the scheduler state will also be restored.
    device : torch.device, optional
        The device to map tensors onto. Defaults to CPU.

    Returns
    -------
    dict
        The full checkpoint dictionary (contains 'epoch', 'metrics', etc.).
    """
    logger = logging.getLogger(__name__)

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Loaded model weights from %s (epoch=%d)",
        checkpoint_path,
        checkpoint.get("epoch", -1),
    )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimiser state.")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored LR scheduler state.")

    return checkpoint


# ---------------------------------------------------------------------------
# Training-curve summary
# ---------------------------------------------------------------------------

def print_training_summary(
    train_losses: list,
    val_aucs: list,
    val_losses: list,
) -> None:
    """
    Print a formatted table summarising the training loss curve and
    validation AUC trend for each epoch.

    Parameters
    ----------
    train_losses : list of float
        Per-epoch training loss values.
    val_aucs : list of float
        Per-epoch validation AUC-ROC values.
    val_losses : list of float
        Per-epoch validation loss values.
    """
    logger = logging.getLogger(__name__)

    header = f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val AUC':>10}"
    separator = "-" * len(header)

    summary_lines = ["\n=== TRAINING SUMMARY ===", separator, header, separator]
    for epoch_idx, (tr_loss, vl_loss, vl_auc) in enumerate(
        zip(train_losses, val_losses, val_aucs), start=1
    ):
        line = f"{epoch_idx:>6}  {tr_loss:>12.6f}  {vl_loss:>10.6f}  {vl_auc:>10.4f}"
        summary_lines.append(line)

    summary_lines.append(separator)

    # Highlight the best epoch for AUC
    best_epoch = int(np.argmax(val_aucs)) + 1
    best_auc = max(val_aucs)
    summary_lines.append(f"Best Val AUC: {best_auc:.4f}  @ Epoch {best_epoch}")
    summary_lines.append("=========================\n")

    summary_str = "\n".join(summary_lines)
    print(summary_str)
    logger.info(summary_str)


# ---------------------------------------------------------------------------
# Count trainable parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.

    Returns
    -------
    int
        Total number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """
    Log the number of trainable and total parameters for a model.

    Parameters
    ----------
    model : nn.Module
        The model to summarise.
    model_name : str
        Human-readable name used in the log message.
    """
    logger = logging.getLogger(__name__)
    total = sum(p.numel() for p in model.parameters())
    trainable = count_parameters(model)
    frozen = total - trainable
    logger.info(
        "%s — Total params: %s | Trainable: %s | Frozen: %s",
        model_name,
        f"{total:,}",
        f"{trainable:,}",
        f"{frozen:,}",
    )
