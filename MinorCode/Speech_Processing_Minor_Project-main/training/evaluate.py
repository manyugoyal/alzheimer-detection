"""
evaluate.py
===========
Evaluation utilities for the Speech Processing Branch.

Computes and reports:
  - Accuracy
  - F1-Score (binary, positive class = AD)
  - AUC-ROC
  - Sensitivity (Recall for AD class)
  - Specificity (Recall for CN class)
  - Confusion matrix

Can be called during training (validation loop) or as a standalone
post-training evaluation script.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """
    Run a full evaluation loop over a DataLoader and compute all metrics.

    Parameters
    ----------
    model : nn.Module
        The SpeechBranchModel (or any model with the same interface).
    dataloader : DataLoader
        Validation or test DataLoader.
    criterion : nn.Module
        Loss function (nn.BCELoss).
    device : torch.device
        Device the model is on.
    threshold : float
        Decision threshold for converting SDS probabilities to binary labels.
        Default: 0.5.
    desc : str
        tqdm progress bar description.

    Returns
    -------
    dict
        Keys: 'loss', 'accuracy', 'f1', 'auc', 'sensitivity', 'specificity'.
        All values are Python floats.
    """
    model.eval()

    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc=desc, leave=False)
        for batch in progress:
            # Move tensors to device
            waveforms = batch["waveform"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device).float()

            # Forward pass
            outputs = model(
                waveforms=waveforms,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_embeddings=False,
            )

            sds = outputs["sds"].squeeze(-1)  # (B,)

            # Compute loss
            loss = criterion(sds, labels)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            probs = sds.cpu().numpy().tolist()
            preds = (sds >= threshold).long().cpu().numpy().tolist()
            lbls = labels.long().cpu().numpy().tolist()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbls)

    # ---------------------------------------------------------------------- #
    # Compute metrics
    # ---------------------------------------------------------------------- #
    avg_loss = total_loss / max(num_batches, 1)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC-ROC requires probabilities, not binary predictions
    if len(set(all_labels)) < 2:
        logger.warning(
            "Only one class present in labels — AUC-ROC is not defined. "
            "Setting AUC = 0.5."
        )
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    # Confusion matrix for sensitivity / specificity
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / max(tp + fn, 1)  # True Positive Rate (Recall for AD)
        specificity = tn / max(tn + fp, 1)  # True Negative Rate (Recall for CN)
    else:
        # Edge case: only one class in the batch
        sensitivity = 0.0
        specificity = 0.0

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }

    logger.info(
        "%s → loss=%.4f | acc=%.4f | F1=%.4f | AUC=%.4f | "
        "sens=%.4f | spec=%.4f",
        desc,
        metrics["loss"],
        metrics["accuracy"],
        metrics["f1"],
        metrics["auc"],
        metrics["sensitivity"],
        metrics["specificity"],
    )

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix printer
# ---------------------------------------------------------------------------

def print_confusion_matrix(
    all_labels: List[int],
    all_preds: List[int],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Print a formatted confusion matrix with class labels.

    Parameters
    ----------
    all_labels : list of int
        Ground-truth binary labels.
    all_preds : list of int
        Predicted binary labels.
    class_names : list of str, optional
        Names for classes [0, 1]. Default: ['Control (CN)', 'Dementia (AD)'].
    """
    if class_names is None:
        class_names = ["Control (CN)", "Dementia (AD)"]

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    header = f"{'':>20} | {'Pred CN':>10} | {'Pred AD':>10}"
    separator = "-" * len(header)
    row_cn = f"{'True CN (Control)':>20} | {cm[0, 0]:>10} | {cm[0, 1]:>10}"
    row_ad = f"{'True AD (Dementia)':>20} | {cm[1, 0]:>10} | {cm[1, 1]:>10}"

    output = "\n".join(["", "=== Confusion Matrix ===", separator, header, separator, row_cn, row_ad, separator, ""])
    print(output)
    logger.info(output)


# ---------------------------------------------------------------------------
# Full test-set evaluation (standalone use)
# ---------------------------------------------------------------------------

def run_test_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run evaluation on the held-out test set and print a detailed report.

    Optionally loads a checkpoint before evaluating.

    Parameters
    ----------
    model : nn.Module
        The SpeechBranchModel.
    test_loader : DataLoader
        DataLoader for the test split.
    device : torch.device
        Compute device.
    checkpoint_path : str, optional
        Path to a saved .pt checkpoint. If provided, loads weights first.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    if checkpoint_path is not None:
        from utils.helpers import load_checkpoint
        load_checkpoint(checkpoint_path, model, device=device)
        logger.info("Loaded checkpoint from: %s", checkpoint_path)

    criterion = nn.BCELoss()

    # Collect all labels and probabilities for confusion matrix
    model.eval()
    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            waveforms = batch["waveform"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device).float()

            outputs = model(
                waveforms=waveforms,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_embeddings=False,
            )
            sds = outputs["sds"].squeeze(-1)

            loss = criterion(sds, labels)
            total_loss += loss.item()
            num_batches += 1

            all_probs.extend(sds.cpu().numpy().tolist())
            all_preds.extend((sds >= threshold).long().cpu().numpy().tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
    else:
        sensitivity = specificity = 0.0

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }

    # Detailed report
    report_lines = [
        "",
        "=" * 50,
        "  TEST SET EVALUATION REPORT",
        "=" * 50,
        f"  Loss          : {metrics['loss']:.6f}",
        f"  Accuracy      : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)",
        f"  F1-Score      : {metrics['f1']:.4f}",
        f"  AUC-ROC       : {metrics['auc']:.4f}",
        f"  Sensitivity   : {metrics['sensitivity']:.4f}  (true positive rate)",
        f"  Specificity   : {metrics['specificity']:.4f}  (true negative rate)",
        "=" * 50,
    ]
    report = "\n".join(report_lines)
    print(report)
    logger.info(report)

    print_confusion_matrix(all_labels, all_preds)

    return metrics
