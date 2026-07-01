"""
inference.py
============
Single-patient inference for the Speech Processing Branch.

Exposes the ``run_speech_branch`` function that is the primary interface
for downstream modules (Cross-Modal Analysis, LLM Fusion, XAI).

Usage (standalone)
------------------
  python inference.py \\
      --audio_path  data/DementiaBank/audio/Dementia/001-1c.wav \\
      --transcript  data/DementiaBank/transcripts/Dementia/001-1c.cha \\
      --checkpoint  checkpoints/best_speech_branch.pt

Usage (from Python)
-------------------
  from inference import run_speech_branch
  result = run_speech_branch(
      audio_path="path/to/audio.wav",
      transcript_path="path/to/transcript.cha",
      checkpoint_path="checkpoints/best_speech_branch.pt",
  )
  print(result["SDS"])        # Float in [0, 1]
  print(result["F_fused"].shape)  # (512,)
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
from transformers import BertTokenizer

# Add speech_branch root to Python path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sds_head import SpeechBranchModel
from preprocessing.audio_processor import (
    extract_egemaps_features,
    load_waveform,
)
from preprocessing.chat_parser import extract_linguistic_features, parse_chat_file
from preprocessing.linguistic_encoder import tokenize_text
from utils.config import Config
from utils.helpers import get_device, load_checkpoint, set_seed, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cached model (avoid re-loading on repeated calls)
# ---------------------------------------------------------------------------
_cached_model: Optional[SpeechBranchModel] = None
_cached_tokenizer: Optional[BertTokenizer] = None
_cached_cfg: Optional[Config] = None


def _get_or_load_model(
    cfg: Config,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> SpeechBranchModel:
    """
    Return the cached SpeechBranchModel if already loaded, otherwise create
    and (optionally) load weights from checkpoint.

    This avoids re-downloading HuBERT/BERT on every inference call.

    Parameters
    ----------
    cfg : Config
    device : torch.device
    checkpoint_path : str, optional

    Returns
    -------
    SpeechBranchModel
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    logger.info("Initialising SpeechBranchModel for inference...")
    model = SpeechBranchModel(cfg=cfg, device=device)
    model = model.to(device)
    model.eval()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        if checkpoint_path is not None:
            logger.warning(
                "Checkpoint not found at '%s'. Running with random weights.",
                checkpoint_path,
            )

    _cached_model = model
    return model


def _get_or_load_tokenizer(bert_model_name: str) -> BertTokenizer:
    """Return cached BertTokenizer or load fresh."""
    global _cached_tokenizer
    if _cached_tokenizer is None:
        logger.info("Loading BertTokenizer: %s", bert_model_name)
        _cached_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    return _cached_tokenizer


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_speech_branch(
    audio_path: str,
    transcript_path: str,
    checkpoint_path: Optional[str] = None,
    cfg: Optional[Config] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run the full Speech Processing Branch on a single patient recording and
    return all outputs required by downstream modules.

    Parameters
    ----------
    audio_path : str
        Path to the patient's audio file (.wav or .mp3). Expected 16kHz mono,
        but resampling is applied automatically if the sample rate differs.
    transcript_path : str
        Path to the patient's CHAT transcript file (.cha). Used for linguistic
        feature extraction and BERT encoding.
    checkpoint_path : str, optional
        Path to a saved SpeechBranchModel checkpoint (.pt file). If None, the
        model runs with random (untrained) weights — useful for testing.
        Defaults to ``cfg.paths.get_best_checkpoint_path()``.
    cfg : Config, optional
        Master configuration. If None, uses the default singleton Config().
    device : torch.device, optional
        Compute device. Auto-detected if None.

    Returns
    -------
    dict with the following keys:
      'SDS'                      : float ∈ [0, 1]
          Speech Diagnostic Score. Higher value → greater AD probability.

      'F_fused'                  : np.ndarray, shape (512,)
          Fused acoustic-linguistic embedding from CrossModalAttention.
          Primary input to downstream Cross-Modal Analysis module.

      'F_ac'                     : np.ndarray, shape (768,)
          Raw mean-pooled HuBERT acoustic embedding (F_ac_pooled).

      'F_ling'                   : np.ndarray, shape (768,)
          [CLS] BERT linguistic embedding.

      'linguistic_features'      : dict
          Interpretable linguistic feature dictionary:
            type_token_ratio, mean_utterance_length, total_utterances,
            total_words, filler_count, filler_rate, lexical_density,
            unique_noun_count, unique_verb_count, brunet_w_index,
            honore_r_statistic.

      'acoustic_features_egemaps': np.ndarray, shape (88,)
          eGeMAPS v02 acoustic feature vector (88-dimensional).

      'patient_text'             : str
          Clean patient speech extracted from the CHAT file.

    Raises
    ------
    FileNotFoundError
        If audio_path or transcript_path does not exist.
    """
    # ------------------------------------------------------------------ #
    # Defaults
    # ------------------------------------------------------------------ #
    if cfg is None:
        cfg = Config()

    if device is None:
        device = get_device()

    if checkpoint_path is None:
        checkpoint_path = cfg.paths.get_best_checkpoint_path()

    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    logger.info(
        "Running Speech Branch inference on: %s", os.path.basename(audio_path)
    )

    # ------------------------------------------------------------------ #
    # Step 1: Parse CHAT transcript → clean patient text
    # ------------------------------------------------------------------ #
    patient_text = parse_chat_file(transcript_path)
    logger.debug("Patient text length: %d chars", len(patient_text))

    # ------------------------------------------------------------------ #
    # Step 2: Extract interpretable linguistic features
    # ------------------------------------------------------------------ #
    linguistic_features = extract_linguistic_features(
        text=patient_text,
        filler_words=cfg.linguistic.filler_words,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Load and preprocess waveform
    # ------------------------------------------------------------------ #
    waveform, sr = load_waveform(
        audio_path,
        target_sr=cfg.audio.sample_rate,
        mono=True,
        normalize=True,
    )
    # Convert to (1, T) batch tensor for model
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, T)

    # ------------------------------------------------------------------ #
    # Step 4: Extract eGeMAPS features (supplementary)
    # ------------------------------------------------------------------ #
    try:
        egemaps_features = extract_egemaps_features(
            audio_path,
            feature_set=cfg.audio.opensmile_feature_set,
            feature_level=cfg.audio.opensmile_feature_level,
        )
    except Exception as exc:
        logger.warning(
            "eGeMAPS extraction failed: %s. Returning zero vector.", exc
        )
        egemaps_features = np.zeros(cfg.audio.egemaps_dim, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Step 5: Tokenise text for BERT
    # ------------------------------------------------------------------ #
    tokenizer = _get_or_load_tokenizer(cfg.linguistic.bert_model_name)
    encoding = tokenize_text(
        text=patient_text,
        tokenizer=tokenizer,
        max_length=cfg.linguistic.max_length,
        padding=cfg.linguistic.padding,
        truncation=cfg.linguistic.truncation,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]          # (1, max_length)
    attention_mask = encoding["attention_mask"] # (1, max_length)
    token_type_ids = encoding.get("token_type_ids")  # (1, max_length) or None

    # ------------------------------------------------------------------ #
    # Step 6: Load model (cached)
    # ------------------------------------------------------------------ #
    model = _get_or_load_model(cfg, device, checkpoint_path)

    # ------------------------------------------------------------------ #
    # Step 7: Model forward pass (with all embeddings)
    # ------------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        outputs = model.predict(
            waveforms=waveform_tensor.to(device),
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device) if token_type_ids is not None else None,
        )

    sds_scalar = float(outputs["sds"].squeeze().cpu().item())
    f_fused_np = outputs["f_fused"].squeeze(0).cpu().numpy()   # (512,)
    f_ac_np = outputs["f_ac"].squeeze(0).cpu().numpy()          # (768,)
    f_ling_np = outputs["f_ling"].squeeze(0).cpu().numpy()      # (768,)

    # ------------------------------------------------------------------ #
    # Build result dictionary
    # ------------------------------------------------------------------ #
    result = {
        "SDS": sds_scalar,
        "F_fused": f_fused_np,
        "F_ac": f_ac_np,
        "F_ling": f_ling_np,
        "linguistic_features": linguistic_features,
        "acoustic_features_egemaps": egemaps_features,
        "patient_text": patient_text,
    }

    logger.info(
        "Inference complete. SDS=%.4f (%s)",
        sds_scalar,
        "AD" if sds_scalar >= 0.5 else "CN",
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech Branch Inference: Alzheimer's Disease Detection"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file (.wav or .mp3)",
    )
    parser.add_argument(
        "--transcript_path",
        type=str,
        required=True,
        help="Path to the CHAT transcript file (.cha)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Defaults to config best checkpoint.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If provided, saves the scalar outputs (SDS, linguistic features) "
             "to this JSON file. Numpy arrays are not JSON-serialised.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = _parse_args()

    result = run_speech_branch(
        audio_path=args.audio_path,
        transcript_path=args.transcript_path,
        checkpoint_path=args.checkpoint,
    )

    # Print human-readable summary
    print("\n" + "=" * 55)
    print("  SPEECH BRANCH INFERENCE RESULTS")
    print("=" * 55)
    print(f"  Audio          : {os.path.basename(args.audio_path)}")
    print(f"  Transcript     : {os.path.basename(args.transcript_path)}")
    print(f"  SDS            : {result['SDS']:.6f}")
    print(f"  Prediction     : {'DEMENTIA (AD)' if result['SDS'] >= 0.5 else 'CONTROL (CN)'}")
    print()
    print("  Linguistic Features:")
    for feat_name, feat_val in result["linguistic_features"].items():
        print(f"    {feat_name:<30}: {feat_val:.4f}")
    print()
    print(f"  F_fused shape  : {result['F_fused'].shape}")
    print(f"  F_ac shape     : {result['F_ac'].shape}")
    print(f"  F_ling shape   : {result['F_ling'].shape}")
    print(f"  eGeMAPS shape  : {result['acoustic_features_egemaps'].shape}")
    print("=" * 55 + "\n")

    # Optionally save JSON output (scalar values only)
    if args.output_json is not None:
        json_output = {
            "SDS": result["SDS"],
            "linguistic_features": result["linguistic_features"],
            "egemaps_dim": int(result["acoustic_features_egemaps"].shape[0]),
            "f_fused_dim": int(result["F_fused"].shape[0]),
            "f_ac_dim": int(result["F_ac"].shape[0]),
            "f_ling_dim": int(result["F_ling"].shape[0]),
            "patient_text_preview": result["patient_text"][:300] + "..."
            if len(result["patient_text"]) > 300
            else result["patient_text"],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON output saved to: {args.output_json}")
