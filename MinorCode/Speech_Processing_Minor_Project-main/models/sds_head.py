"""
sds_head.py
===========
Speech Diagnostic Score (SDS) computation head and the full SpeechBranchModel.

SDS Formula (from paper)
-------------------------
  SDS = σ( W_f · concat(F_ac, F_ling) + b_f )

The full pipeline (SpeechBranchModel) integrates:
  1. AcousticEncoder  → F_ac   ∈ R^{768}
  2. LinguisticEncoder → F_ling ∈ R^{768}
  3. CrossModalAttention → F_fused ∈ R^{512}
  4. SDSHead → SDS ∈ [0, 1]

SDSHead Architecture
--------------------
  Input:  F_fused ∈ R^{512}
  FC1:    512 → 128, ReLU
  Dropout(0.3)
  FC2:    128 → 32,  ReLU
  FC3:    32  → 1,   Sigmoid
  Output: SDS ∈ [0, 1]

SDS → 1 indicates high probability of Alzheimer's Disease.
SDS → 0 indicates healthy control.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.acoustic_encoder import AcousticEncoder
from models.linguistic_encoder import LinguisticEncoder
from models.cross_attention import CrossModalAttention
from utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDS MLP Head
# ---------------------------------------------------------------------------

class SDSHead(nn.Module):
    """
    Multi-Layer Perceptron that maps the fused embedding F_fused to a scalar
    Speech Diagnostic Score (SDS) ∈ [0, 1] via sigmoid activation.

    Implements:
        SDS = σ(W_f · F_fused + b_f)

    expanded as:
        x  = ReLU(FC1(F_fused))   — 512 → 128
        x  = Dropout(x, p=0.3)
        x  = ReLU(FC2(x))         — 128 → 32
        SDS = σ(FC3(x))           — 32  → 1

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input fused embedding. Default: 512.
    hidden_1 : int
        Size of the first hidden layer. Default: 128.
    hidden_2 : int
        Size of the second hidden layer. Default: 32.
    dropout : float
        Dropout probability between FC1 and FC2. Default: 0.3.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_1: int = 128,
        hidden_2: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # FC1: 512 → 128
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # FC2: 128 → 32
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(inplace=True),
            # FC3: 32 → 1
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )

        # Initialise all linear layers with Xavier uniform
        self._init_weights()

        logger.debug(
            "SDSHead: input_dim=%d → hidden=%d → hidden=%d → 1 (sigmoid)",
            input_dim,
            hidden_1,
            hidden_2,
        )

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, f_fused: torch.Tensor) -> torch.Tensor:
        """
        Compute the Speech Diagnostic Score.

        Parameters
        ----------
        f_fused : torch.Tensor, shape (B, 512)
            Fused acoustic-linguistic embedding from CrossModalAttention.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            SDS values in [0, 1]. Values near 1 indicate AD; near 0 indicate CN.
        """
        return self.mlp(f_fused)  # (B, 1)


# ---------------------------------------------------------------------------
# Full Speech Branch Model
# ---------------------------------------------------------------------------

class SpeechBranchModel(nn.Module):
    """
    End-to-end Speech Processing Branch for Alzheimer's Disease detection.

    Pipeline
    --------
    waveform     → AcousticEncoder  → F_ac   ∈ R^{B × 768}
    text tokens  → LinguisticEncoder → F_ling ∈ R^{B × 768}
    (F_ac, F_ling) → CrossModalAttention → F_fused ∈ R^{B × 512}
    F_fused      → SDSHead          → SDS   ∈ R^{B × 1}

    Only CrossModalAttention and SDSHead are trainable by default
    (HuBERT and BERT are frozen).

    Parameters
    ----------
    cfg : Config
        Master configuration object with audio, linguistic, and model sub-configs.
    device : torch.device
        Target compute device.
    """

    def __init__(
        self,
        cfg: Config,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.device = device

        # ------------------------------------------------------------------ #
        # 1. Acoustic Encoder (HuBERT — frozen)
        # ------------------------------------------------------------------ #
        self.acoustic_encoder = AcousticEncoder(
            model_name=cfg.audio.hubert_model_name,
            sample_rate=cfg.audio.sample_rate,
            freeze=cfg.model.freeze_hubert,
            chunk_size_seconds=cfg.audio.chunk_size_seconds,
            device=device,
        )

        # ------------------------------------------------------------------ #
        # 2. Linguistic Encoder (BERT — frozen)
        # ------------------------------------------------------------------ #
        self.linguistic_encoder = LinguisticEncoder(
            model_name=cfg.linguistic.bert_model_name,
            freeze=cfg.model.freeze_bert,
            device=device,
        )

        # ------------------------------------------------------------------ #
        # 3. Cross-Modal Attention Fusion (trainable)
        # ------------------------------------------------------------------ #
        self.cross_attention = CrossModalAttention(
            acoustic_dim=cfg.audio.hubert_hidden_dim,      # 768
            linguistic_dim=cfg.linguistic.bert_hidden_dim, # 768
            d_model=cfg.model.d_model,                     # 256
            num_heads=cfg.model.num_heads,                 # 4
            dropout=0.1,
        ).to(device)

        # ------------------------------------------------------------------ #
        # 4. SDS Head (trainable)
        # ------------------------------------------------------------------ #
        self.sds_head = SDSHead(
            input_dim=cfg.model.fused_dim,          # 512
            hidden_1=cfg.model.sds_hidden_1,        # 128
            hidden_2=cfg.model.sds_hidden_2,        # 32
            dropout=cfg.model.sds_dropout,          # 0.3
        ).to(device)

        logger.info(
            "SpeechBranchModel initialised on device=%s. "
            "Trainable modules: CrossModalAttention + SDSHead.",
            device,
        )

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        waveforms: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the Speech Processing Branch.

        Parameters
        ----------
        waveforms : torch.Tensor, shape (B, T)
            Zero-padded batch of 16kHz waveforms.
        input_ids : torch.Tensor, shape (B, seq_len)
            BERT input token IDs.
        attention_mask : torch.Tensor, shape (B, seq_len)
            BERT attention mask.
        token_type_ids : torch.Tensor, shape (B, seq_len), optional
            BERT segment IDs.
        return_embeddings : bool
            If True, the returned dict also contains F_ac, F_ling, and F_fused.
            Default False (used during training for efficiency).

        Returns
        -------
        dict with keys:
          - 'sds'     : FloatTensor (B, 1) — Speech Diagnostic Score
          - 'f_fused' : FloatTensor (B, 512) — fused embedding (always returned
                         regardless of return_embeddings for downstream pipeline)
          - 'f_ac'    : FloatTensor (B, 768) — if return_embeddings=True
          - 'f_ling'  : FloatTensor (B, 768) — if return_embeddings=True
        """
        # -- Step 1: Acoustic encoding --
        # HuBERT encoder processes each waveform; returns (B, 768)
        f_ac = self.acoustic_encoder(waveforms)    # (B, 768) on self.device

        # -- Step 2: Linguistic encoding --
        f_ling = self.linguistic_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # (B, 768) on self.device

        # -- Step 3: Cross-modal attention fusion --
        # Both tensors must be on the same device as cross_attention
        f_fused = self.cross_attention(
            f_ac=f_ac.to(self.device),
            f_ling=f_ling.to(self.device),
        )  # (B, 512)

        # -- Step 4: SDS computation --
        sds = self.sds_head(f_fused)  # (B, 1)

        result = {
            "sds": sds,
            "f_fused": f_fused,
        }

        if return_embeddings:
            result["f_ac"] = f_ac
            result["f_ling"] = f_ling

        return result

    # ---------------------------------------------------------------------- #
    # Convenience method for inference
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self,
        waveforms: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-mode forward pass (no gradient computation).

        Returns all embeddings (f_ac, f_ling, f_fused) in addition to the SDS.

        Parameters
        ----------
        waveforms, input_ids, attention_mask, token_type_ids :
            Same as ``forward``.

        Returns
        -------
        dict
            {'sds', 'f_fused', 'f_ac', 'f_ling'}
        """
        self.eval()
        return self.forward(
            waveforms=waveforms,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_embeddings=True,
        )
