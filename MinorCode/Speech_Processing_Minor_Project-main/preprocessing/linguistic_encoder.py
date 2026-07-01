"""
linguistic_encoder.py  (preprocessing/)
========================================
Preprocessing pipeline for the linguistic branch:

  1. Tokenises clean patient text using BertTokenizer
  2. Returns PyTorch tensors (input_ids, attention_mask, token_type_ids)
     ready for BertModel forward pass

Note: The actual BERT forward pass lives in models/linguistic_encoder.py.
This module is purely the tokenisation / preprocessing step.
"""

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def tokenize_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Tokenise a plain-text string using a pre-loaded BertTokenizer.

    Parameters
    ----------
    text : str
        Clean patient speech string (output of chat_parser.parse_chat_file).
    tokenizer : BertTokenizer
        A ``transformers.BertTokenizer`` instance (already loaded).
    max_length : int
        Maximum sequence length. Sequences longer than this are truncated;
        shorter ones are padded.
    padding : str
        Padding strategy. 'max_length' pads all sequences to ``max_length``.
    truncation : bool
        Whether to truncate sequences exceeding ``max_length``.
    return_tensors : str
        Format of returned tensors: 'pt' for PyTorch tensors.

    Returns
    -------
    dict
        Dictionary with keys:
          - 'input_ids'      : LongTensor of shape (1, max_length)
          - 'attention_mask' : LongTensor of shape (1, max_length)
          - 'token_type_ids' : LongTensor of shape (1, max_length)
    """
    if not text or not text.strip():
        logger.warning(
            "Empty text passed to tokenizer. Returning zero-filled tensors."
        )
        text = "[PAD]"  # Fallback to a known token

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
        return_token_type_ids=True,
    )

    logger.debug(
        "Tokenised text: length=%d chars → %d tokens",
        len(text),
        (encoding["attention_mask"] == 1).sum().item(),
    )
    return encoding
