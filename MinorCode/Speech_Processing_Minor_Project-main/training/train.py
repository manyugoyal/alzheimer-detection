"""
train.py
========
Full training loop for the Speech Processing Branch.

Features
--------
  - Stratified 80/10/10 train/val/test split
  - WeightedRandomSampler to handle class imbalance
  - AdamW optimiser with CosineAnnealingLR scheduler
  - BCELoss on sigmoid SDS output
  - Early stopping on validation AUC (patience=5)
  - Checkpoint saved when best validation AUC is achieved
  - Gradient clipping
  - Per-epoch logging with tqdm
  - Training summary table printed at end

Usage
-----
  python -m training.train                     # use default Config
  python -m training.train --epochs 50         # override epochs
  python -m training.train --help              # show all arguments

"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertTokenizer

# --- Project imports ---
# Add speech_branch root to path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DementiaBankDataset, collate_fn, _discover_samples
from models.sds_head import SpeechBranchModel
from training.evaluate import evaluate_model
from utils.config import Config
from utils.helpers import (
    get_device,
    load_checkpoint,
    log_model_summary,
    print_training_summary,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data split helper
# ---------------------------------------------------------------------------

def split_samples(
    samples: List[dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Perform a stratified 80/10/10 split on the full sample list.

    Stratification is by label (0/1) to preserve class balance across splits.

    Parameters
    ----------
    samples : list of dict
        All samples from ``_discover_samples``.
    train_ratio : float
        Proportion for training. Default 0.80.
    val_ratio : float
        Proportion for validation. Default 0.10.
        The test proportion is 1 - train_ratio - val_ratio.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    train_samples, val_samples, test_samples : list of dict
        Three non-overlapping lists of sample dicts.
    """
    labels = [s["label"] for s in samples]
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - relative_val),
        stratify=temp_labels,
        random_state=seed,
    )

    logger.info(
        "Data split → train: %d | val: %d | test: %d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )
    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# WeightedRandomSampler builder
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: DementiaBankDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so that each class is sampled with
    equal expected frequency, mitigating class imbalance.

    Parameters
    ----------
    dataset : DementiaBankDataset

    Returns
    -------
    WeightedRandomSampler
    """
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)

    # Weight for each class = 1 / class_count
    class_weights = 1.0 / class_counts.astype(float)

    # Assign per-sample weight
    sample_weights = [class_weights[lbl] for lbl in labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(labels),
        replacement=True,
    )
    logger.info(
        "WeightedRandomSampler: class counts=%s, class weights=%s",
        class_counts.tolist(),
        [f"{w:.4f}" for w in class_weights.tolist()],
    )
    return sampler


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: Config,
    tokenizer: BertTokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Discover samples, split, and build DataLoaders with appropriate samplers.

    Parameters
    ----------
    cfg : Config
        Master configuration.
    tokenizer : BertTokenizer
        Pre-loaded tokenizer.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    all_samples = _discover_samples(
        audio_dir=cfg.paths.audio_dir,
        transcript_dir=cfg.paths.transcript_dir,
    )

    train_s, val_s, test_s = split_samples(
        all_samples,
        train_ratio=cfg.training.train_ratio,
        val_ratio=cfg.training.val_ratio,
        seed=cfg.repro.seed,
    )

    train_dataset = DementiaBankDataset(
        audio_dir=cfg.paths.audio_dir,
        transcript_dir=cfg.paths.transcript_dir,
        cfg=cfg,
        tokenizer=tokenizer,
        samples=train_s,
        extract_egemaps=True,
    )
    val_dataset = DementiaBankDataset(
        audio_dir=cfg.paths.audio_dir,
        transcript_dir=cfg.paths.transcript_dir,
        cfg=cfg,
        tokenizer=tokenizer,
        samples=val_s,
        extract_egemaps=True,
    )
    test_dataset = DementiaBankDataset(
        audio_dir=cfg.paths.audio_dir,
        transcript_dir=cfg.paths.transcript_dir,
        cfg=cfg,
        tokenizer=tokenizer,
        samples=test_s,
        extract_egemaps=True,
    )

    # Use WeightedRandomSampler only for training
    train_sampler = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,  # Overrides shuffle
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if str(cfg.device) != "cpu" else False,
        drop_last=True,   # Avoid batch-size-1 issues with BatchNorm
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if str(cfg.device) != "cpu" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if str(cfg.device) != "cpu" else False,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip_norm: float = 1.0,
) -> float:
    """
    Run one training epoch and return the mean training loss.

    Parameters
    ----------
    model : nn.Module
        The SpeechBranchModel.
    dataloader : DataLoader
        Training DataLoader.
    optimizer : torch.optim.Optimizer
        AdamW optimiser.
    criterion : nn.Module
        BCELoss.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch index (1-based, for logging).
    grad_clip_norm : float
        Max gradient norm for clipping. Set 0 to disable.

    Returns
    -------
    float
        Mean training loss over all batches in this epoch.
    """
    model.train()
    # Keep frozen encoders in eval mode to prevent BatchNorm/Dropout updates
    if hasattr(model, "acoustic_encoder"):
        model.acoustic_encoder.hubert.eval()
    if hasattr(model, "linguistic_encoder"):
        model.linguistic_encoder.bert.eval()

    total_loss = 0.0
    num_batches = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [train]",
        leave=True,
    )

    for batch in progress:
        # Move inputs to device
        waveforms = batch["waveform"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["label"].to(device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            waveforms=waveforms,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_embeddings=False,
        )
        sds = outputs["sds"].squeeze(-1)   # (B,)

        # Compute loss
        loss = criterion(sds, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip_norm and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    mean_loss = total_loss / max(num_batches, 1)
    logger.info("Epoch %02d — train_loss=%.6f", epoch, mean_loss)
    return mean_loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    """
    Full training loop with validation, early stopping, and checkpointing.

    Parameters
    ----------
    cfg : Config
        Master configuration object. All hyperparameters are read from here.
    """
    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    setup_logging(log_file=cfg.paths.log_file)
    set_seed(
        cfg.repro.seed,
        deterministic=cfg.repro.deterministic,
        benchmark=cfg.repro.benchmark,
    )
    device = get_device()
    cfg.device = str(device)

    logger.info("=== Starting Speech Branch Training ===")
    logger.info("Config: %s", cfg)

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    logger.info("Loading BERT tokenizer: %s", cfg.linguistic.bert_model_name)
    tokenizer = BertTokenizer.from_pretrained(cfg.linguistic.bert_model_name)

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    train_loader, val_loader, test_loader = build_dataloaders(cfg, tokenizer)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = SpeechBranchModel(cfg=cfg, device=device)
    model = model.to(device)
    log_model_summary(model, "SpeechBranchModel")

    # ------------------------------------------------------------------ #
    # Loss, Optimiser, Scheduler
    # ------------------------------------------------------------------ #
    criterion = nn.BCELoss()

    # Only optimise trainable parameters (CrossModalAttention + SDSHead)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Optimising %d parameter tensors (frozen encoders excluded).",
        len(trainable_params),
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.scheduler_t_max,
        eta_min=1e-6,
    )

    # ------------------------------------------------------------------ #
    # Training history
    # ------------------------------------------------------------------ #
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_aucs: List[float] = []

    best_val_auc = 0.0
    epochs_without_improvement = 0
    best_checkpoint_path = cfg.paths.get_best_checkpoint_path()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    for epoch in range(1, cfg.training.num_epochs + 1):
        # Training epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            grad_clip_norm=cfg.training.grad_clip_norm or 1.0,
        )
        train_losses.append(train_loss)

        # Validation
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            desc=f"Epoch {epoch:02d} [val]",
        )
        val_losses.append(val_metrics["loss"])
        val_aucs.append(val_metrics["auc"])

        # LR scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info("Epoch %02d — LR=%.2e", epoch, current_lr)

        # Early stopping and checkpointing
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_path=best_checkpoint_path,
                scheduler=scheduler,
            )
            logger.info(
                "✓ New best val AUC: %.4f — checkpoint saved.", best_val_auc
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement for %d / %d epochs.",
                epochs_without_improvement,
                cfg.training.early_stopping_patience,
            )

        if epochs_without_improvement >= cfg.training.early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d).",
                epoch,
                cfg.training.early_stopping_patience,
            )
            break

    # ------------------------------------------------------------------ #
    # Training summary
    # ------------------------------------------------------------------ #
    print_training_summary(train_losses, val_aucs, val_losses)

    # ------------------------------------------------------------------ #
    # Final test evaluation with best checkpoint
    # ------------------------------------------------------------------ #
    logger.info("Loading best checkpoint for test evaluation...")
    load_checkpoint(best_checkpoint_path, model, device=device)

    from training.evaluate import run_test_evaluation
    test_metrics = run_test_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    logger.info("=== Training Complete ===")
    logger.info("Best Val AUC: %.4f", best_val_auc)
    logger.info("Test Metrics: %s", test_metrics)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for overriding config values."""
    parser = argparse.ArgumentParser(
        description="Train Speech Processing Branch for Alzheimer's Detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs (default from config)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--audio_dir", type=str, default=None,
        help="Override audio data directory",
    )
    parser.add_argument(
        "--transcript_dir", type=str, default=None,
        help="Override transcript directory",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Override checkpoint save directory",
    )
    parser.add_argument(
        "--no_egemaps", action="store_true",
        help="Disable eGeMAPS extraction during data loading (faster)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.audio_dir is not None:
        cfg.paths.audio_dir = args.audio_dir
    if args.transcript_dir is not None:
        cfg.paths.transcript_dir = args.transcript_dir
    if args.checkpoint_dir is not None:
        cfg.paths.checkpoint_dir = args.checkpoint_dir

    train(cfg)
