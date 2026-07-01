"""
linguistic_encoder.py  (models/)
==================================
BERT-based linguistic encoder for the Speech Processing Branch.

Architecture
------------
  Input  : Tokenised BERT inputs (input_ids, attention_mask, token_type_ids)
  Step 1 : BertModel (frozen) → last hidden state, shape (B, seq_len, 768)
  Step 2 : Extract [CLS] token at position 0 → F_ling ∈ R^{B × 768}

The [CLS] embedding encodes a sentence-level semantic representation of the
patient's speech, which is then fused with acoustic features in CrossModalAttention.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from transformers import BertModel

logger = logging.getLogger(__name__)


class LinguisticEncoder(nn.Module):
    """
    Frozen BERT encoder that extracts the [CLS] token embedding from the
    last hidden layer as F_ling ∈ R^{768}.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub identifier.
        Default: 'bert-base-uncased'
    freeze : bool
        If True (default), all BERT parameters are frozen.
    device : torch.device, optional
        Device to move the model to. Defaults to CPU.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        if device is None:
            device = torch.device("cpu")
        self.device = device

        logger.info("Loading BERT model: %s", model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.bert = self.bert.to(device)

        if freeze:
            self._freeze_bert()
            logger.info("BERT weights frozen — will not be updated during training.")

        # Dimensionality of the [CLS] output vector
        self.output_dim = self.bert.config.hidden_size  # 768

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _freeze_bert(self) -> None:
        """Disable gradient computation for all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract F_ling from BERT's [CLS] token.

        Parameters
        ----------
        input_ids : torch.Tensor, shape (B, seq_len)
            Token IDs from BertTokenizer (LongTensor).
        attention_mask : torch.Tensor, shape (B, seq_len)
            Binary mask — 1 for real tokens, 0 for padding (LongTensor).
        token_type_ids : torch.Tensor, shape (B, seq_len), optional
            Segment IDs. For single-sentence inputs all zeros (LongTensor).
            If None, BERT handles it internally.

        Returns
        -------
        torch.Tensor, shape (B, 768)
            [CLS] token embeddings from the last hidden layer.
            F_ling ∈ R^{B × 768}.
        """
        # Move tensors to the model's device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=False,   # We only need the last layer
                return_dict=True,
            )

        # outputs.last_hidden_state: (B, seq_len, 768)
        # [CLS] token is at position 0
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        return cls_embedding
