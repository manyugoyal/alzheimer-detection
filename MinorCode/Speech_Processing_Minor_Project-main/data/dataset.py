"""
dataset.py
==========
PyTorch Dataset for the DementiaBank Pitt Corpus.

Expected directory layout
--------------------------
data/DementiaBank/
├── audio/
│   ├── Dementia/       # AD-positive recordings (.wav or .mp3)
│   │   └── 001-1c.wav
│   └── Control/        # Control recordings
│       └── 002-1c.wav
└── transcripts/
    ├── Dementia/
    │   └── 001-1c.cha
    └── Control/
        └── 002-1c.cha

Label assignment
----------------
Files inside the 'Dementia' subdirectory receive label 1 (AD).
Files inside the 'Control'  subdirectory receive label 0 (CN).

Note: Subdirectory names are matched case-insensitively for robustness
(e.g., 'dementia', 'Dementia', 'DEMENTIA' are all accepted).

This Dataset returns pre-tokenised BERT inputs and waveform chunks.
It does NOT run HuBERT or BERT forward passes — those happen inside
the model at training time, keeping the dataset lightweight and cacheable.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from preprocessing.audio_processor import load_waveform, extract_egemaps_features
from preprocessing.chat_parser import parse_chat_file, extract_linguistic_features
from preprocessing.linguistic_encoder import tokenize_text
from utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: scan dataset directory for (audio_path, transcript_path, label) triples
# ---------------------------------------------------------------------------

def _discover_samples(
    audio_dir: str,
    transcript_dir: str,
    audio_extensions: Tuple[str, ...] = (".wav", ".mp3"),
) -> List[Dict]:
    """
    Walk the dataset directories and build a flat list of sample dictionaries.

    Each sample dict has keys: 'audio_path', 'transcript_path', 'label', 'participant_id'.

    Files are matched by participant ID (basename without extension).
    If a matching transcript is not found, the sample is skipped with a warning.

    Parameters
    ----------
    audio_dir : str
        Root directory containing 'Dementia/' and 'Control/' subdirectories
        with audio files.
    transcript_dir : str
        Root directory containing 'Dementia/' and 'Control/' subdirectories
        with .cha transcript files.
    audio_extensions : tuple of str
        File extensions to accept as audio files.

    Returns
    -------
    List[dict]
        List of sample dictionaries, each containing:
          - 'audio_path'      : str — path to audio file
          - 'transcript_path' : str — path to .cha transcript file
          - 'label'           : int — 1 for AD, 0 for Control
          - 'participant_id'  : str — stem of the audio filename
    """
    audio_root = Path(audio_dir)
    trans_root = Path(transcript_dir)

    if not audio_root.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not trans_root.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    # Map subfolder name → label
    label_map = {
        "dementia": 1,
        "control": 0,
    }

    samples: List[Dict] = []

    for label_folder in audio_root.iterdir():
        if not label_folder.is_dir():
            continue

        folder_name_lower = label_folder.name.lower()
        label: Optional[int] = None
        for key, val in label_map.items():
            if key in folder_name_lower:
                label = val
                break

        if label is None:
            logger.warning(
                "Unknown subfolder '%s' in audio directory — skipping. "
                "Expected 'Dementia' or 'Control'.",
                label_folder.name,
            )
            continue

        # Corresponding transcript subfolder (same name)
        trans_folder = trans_root / label_folder.name
        if not trans_folder.exists():
            # Try to find it case-insensitively
            matched = [
                d for d in trans_root.iterdir()
                if d.is_dir() and d.name.lower() == folder_name_lower
            ]
            trans_folder = matched[0] if matched else None

        if trans_folder is None:
            logger.warning(
                "No matching transcript folder for '%s' — samples in this "
                "class will be skipped.",
                label_folder.name,
            )
            continue

        for audio_file in sorted(label_folder.iterdir()):
            if audio_file.suffix.lower() not in audio_extensions:
                continue

            participant_id = audio_file.stem

            # Find transcript with same stem
            trans_file = trans_folder / f"{participant_id}.cha"
            if not trans_file.exists():
                # Try .CHA variant
                trans_file_upper = trans_folder / f"{participant_id}.CHA"
                if trans_file_upper.exists():
                    trans_file = trans_file_upper
                else:
                    logger.warning(
                        "No transcript found for participant '%s' — skipping.",
                        participant_id,
                    )
                    continue

            samples.append({
                "audio_path": str(audio_file),
                "transcript_path": str(trans_file),
                "label": label,
                "participant_id": participant_id,
            })

    logger.info(
        "Discovered %d samples (%d AD, %d CN)",
        len(samples),
        sum(s["label"] == 1 for s in samples),
        sum(s["label"] == 0 for s in samples),
    )
    return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DementiaBankDataset(Dataset):
    """
    PyTorch Dataset for the DementiaBank Pitt Corpus.

    Each ``__getitem__`` call returns a dictionary with:
      - 'input_ids'         : LongTensor (max_length,)   — BERT token IDs
      - 'attention_mask'    : LongTensor (max_length,)   — BERT attention mask
      - 'token_type_ids'    : LongTensor (max_length,)   — BERT token type IDs
      - 'waveform'          : FloatTensor (T,)           — raw 16kHz waveform
      - 'label'             : LongTensor scalar          — 0 (CN) or 1 (AD)
      - 'participant_id'    : str                        — participant identifier
      - 'linguistic_feats'  : FloatTensor (11,)          — interpretable ling. features
      - 'egemaps_feats'     : FloatTensor (88,)          — eGeMAPS acoustic features

    HuBERT and BERT forward passes are intentionally NOT run here to keep
    the Dataset fast and memory-efficient. The model's forward method handles
    feature extraction at training time.

    Parameters
    ----------
    audio_dir : str
        Root directory with Dementia/ and Control/ audio subdirectories.
    transcript_dir : str
        Root directory with matching .cha transcript subdirectories.
    cfg : Config
        Master configuration object.
    tokenizer : BertTokenizer
        Pre-loaded BERT tokenizer.
    samples : list of dict, optional
        If provided, this list is used directly instead of scanning directories.
        Useful for passing pre-split train/val/test subsets.
    max_audio_duration : float
        Maximum audio duration in seconds. Longer files are truncated.
        Default: 300 seconds (5 minutes).
    extract_egemaps : bool
        Whether to extract eGeMAPS features during __getitem__.
        Set False to speed up data loading when eGeMAPS is not needed.
    """

    def __init__(
        self,
        audio_dir: str,
        transcript_dir: str,
        cfg: Config,
        tokenizer: BertTokenizer,
        samples: Optional[List[Dict]] = None,
        max_audio_duration: float = 300.0,
        extract_egemaps: bool = True,
    ) -> None:
        super().__init__()
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_audio_duration = max_audio_duration
        self.extract_egemaps = extract_egemaps

        if samples is not None:
            self.samples = samples
        else:
            self.samples = _discover_samples(audio_dir, transcript_dir)

        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples found. Check that audio_dir and transcript_dir "
                "contain 'Dementia/' and 'Control/' subdirectories."
            )

        logger.info(
            "DementiaBankDataset initialised with %d samples.", len(self.samples)
        )

    def __len__(self) -> int:
        """Return the total number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and preprocess a single sample.

        Parameters
        ----------
        idx : int
            Index into the sample list.

        Returns
        -------
        dict
            Sample dictionary as described in the class docstring.
        """
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        transcript_path = sample["transcript_path"]
        label = sample["label"]
        participant_id = sample["participant_id"]

        # ------------------------------------------------------------------ #
        # 1. Load and preprocess waveform
        # ------------------------------------------------------------------ #
        try:
            waveform, sr = load_waveform(
                audio_path,
                target_sr=self.cfg.audio.sample_rate,
                mono=True,
                normalize=True,
            )
            # Truncate if too long
            max_samples = int(self.max_audio_duration * self.cfg.audio.sample_rate)
            if len(waveform) > max_samples:
                logger.debug(
                    "Truncating audio '%s' from %d to %d samples.",
                    participant_id,
                    len(waveform),
                    max_samples,
                )
                waveform = waveform[:max_samples]
        except Exception as exc:
            logger.error(
                "Failed to load audio for '%s': %s. Using zero waveform.",
                participant_id,
                exc,
            )
            waveform = np.zeros(self.cfg.audio.sample_rate, dtype=np.float32)

        waveform_tensor = torch.from_numpy(waveform).float()

        # ------------------------------------------------------------------ #
        # 2. Parse CHAT transcript
        # ------------------------------------------------------------------ #
        try:
            patient_text = parse_chat_file(transcript_path)
        except Exception as exc:
            logger.error(
                "Failed to parse transcript for '%s': %s. Using empty string.",
                participant_id,
                exc,
            )
            patient_text = ""

        # ------------------------------------------------------------------ #
        # 3. Tokenise text for BERT
        # ------------------------------------------------------------------ #
        encoding = tokenize_text(
            text=patient_text,
            tokenizer=self.tokenizer,
            max_length=self.cfg.linguistic.max_length,
            padding=self.cfg.linguistic.padding,
            truncation=self.cfg.linguistic.truncation,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)        # (max_length,)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        # ------------------------------------------------------------------ #
        # 4. Extract interpretable linguistic features
        # ------------------------------------------------------------------ #
        try:
            ling_feature_dict = extract_linguistic_features(
                text=patient_text,
                filler_words=self.cfg.linguistic.filler_words,
            )
            ling_feature_values = list(ling_feature_dict.values())
            linguistic_feats = torch.tensor(ling_feature_values, dtype=torch.float32)
        except Exception as exc:
            logger.error(
                "Failed to extract linguistic features for '%s': %s",
                participant_id,
                exc,
            )
            linguistic_feats = torch.zeros(11, dtype=torch.float32)

        # ------------------------------------------------------------------ #
        # 5. Extract eGeMAPS if requested
        # ------------------------------------------------------------------ #
        if self.extract_egemaps:
            try:
                egemaps = extract_egemaps_features(
                    audio_path,
                    feature_set=self.cfg.audio.opensmile_feature_set,
                    feature_level=self.cfg.audio.opensmile_feature_level,
                )
                egemaps_tensor = torch.from_numpy(egemaps).float()
            except Exception as exc:
                logger.warning(
                    "eGeMAPS extraction failed for '%s': %s. Using zeros.",
                    participant_id,
                    exc,
                )
                egemaps_tensor = torch.zeros(
                    self.cfg.audio.egemaps_dim, dtype=torch.float32
                )
        else:
            egemaps_tensor = torch.zeros(
                self.cfg.audio.egemaps_dim, dtype=torch.float32
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "waveform": waveform_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "participant_id": participant_id,
            "linguistic_feats": linguistic_feats,
            "egemaps_feats": egemaps_tensor,
        }

    def get_labels(self) -> List[int]:
        """
        Return a list of integer labels for all samples.

        Used by ``WeightedRandomSampler`` to compute per-class sampling weights.

        Returns
        -------
        List[int]
            List of 0/1 labels in dataset order.
        """
        return [s["label"] for s in self.samples]


# ---------------------------------------------------------------------------
# Collate function: handles variable-length waveforms
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function that zero-pads variable-length waveforms
    to the length of the longest waveform in the batch.

    BERT inputs are already padded to max_length by the tokenizer, so they
    require no special handling here.

    Parameters
    ----------
    batch : list of dict
        List of sample dictionaries from ``DementiaBankDataset.__getitem__``.

    Returns
    -------
    dict
        Batched tensors with the same keys as the individual sample dicts.
        'participant_id' is returned as a list of strings.
    """
    # Find the maximum waveform length in this batch
    max_len = max(sample["waveform"].shape[0] for sample in batch)

    waveforms_padded = []
    for sample in batch:
        wf = sample["waveform"]
        pad_len = max_len - wf.shape[0]
        if pad_len > 0:
            wf = torch.nn.functional.pad(wf, (0, pad_len), mode="constant", value=0.0)
        waveforms_padded.append(wf)

    return {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "token_type_ids": torch.stack([s["token_type_ids"] for s in batch]),
        "waveform": torch.stack(waveforms_padded),
        "label": torch.stack([s["label"] for s in batch]),
        "participant_id": [s["participant_id"] for s in batch],
        "linguistic_feats": torch.stack([s["linguistic_feats"] for s in batch]),
        "egemaps_feats": torch.stack([s["egemaps_feats"] for s in batch]),
    }
