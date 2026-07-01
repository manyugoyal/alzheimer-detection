"""
cross_attention.py
==================
Cross-Modal Attention Fusion module for the Speech Processing Branch.

Architecture
------------
Inputs:
  F_ac   ∈ R^{B × 768}  — pooled acoustic embedding (from HuBERT)
  F_ling ∈ R^{B × 768}  — [CLS] linguistic embedding (from BERT)

Steps:
  1. Project F_ac   → Q_ac,  K_ac,  V_ac   ∈ R^{B × 1 × d_model}
     Project F_ling → Q_ling, K_ling, V_ling ∈ R^{B × 1 × d_model}

  2. Acoustic-to-Linguistic attention:
       attended_ac = MultiheadAttention(Q=Q_ac, K=K_ling, V=V_ling)
       → F_ac_attended ∈ R^{B × d_model}
     Linguistic-to-Acoustic attention:
       attended_ling = MultiheadAttention(Q=Q_ling, K=K_ac, V=V_ac)
       → F_ling_attended ∈ R^{B × d_model}

  3. Concatenate both attended outputs:
       F_fused = concat(F_ac_attended, F_ling_attended) ∈ R^{B × 512}

The fused embedding F_fused ∈ R^{512} is then passed to SDSHead.

Note on attention semantics
---------------------------
Acoustic-to-Linguistic:
  "Which parts of the linguistic summary are most relevant
   to the acoustic pattern I observed?"

Linguistic-to-Acoustic:
  "Which acoustic properties best explain the semantic content
   of what was said?"

Both directions are concatenated to preserve complementary information.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention fusion between acoustic and linguistic
    embeddings, producing a fused vector F_fused ∈ R^{B × 512}.

    Parameters
    ----------
    acoustic_dim : int
        Dimensionality of the acoustic input embedding. Default: 768.
    linguistic_dim : int
        Dimensionality of the linguistic input embedding. Default: 768.
    d_model : int
        Common projection dimension for attention queries/keys/values.
        Default: 256.
    num_heads : int
        Number of parallel attention heads in MultiheadAttention. Default: 4.
    dropout : float
        Dropout probability applied inside MultiheadAttention. Default: 0.1.
    """

    def __init__(
        self,
        acoustic_dim: int = 768,
        linguistic_dim: int = 768,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.output_dim = 2 * d_model  # 512 after concatenation

        # ------------------------------------------------------------------
        # Linear projections: map input embeddings to common d_model space
        # ------------------------------------------------------------------
        # Acoustic projections
        self.proj_ac_q = nn.Linear(acoustic_dim, d_model)    # Q for ac→ling attention
        self.proj_ac_k = nn.Linear(acoustic_dim, d_model)    # K for ling→ac attention
        self.proj_ac_v = nn.Linear(acoustic_dim, d_model)    # V for ling→ac attention

        # Linguistic projections
        self.proj_ling_q = nn.Linear(linguistic_dim, d_model)  # Q for ling→ac attention
        self.proj_ling_k = nn.Linear(linguistic_dim, d_model)  # K for ac→ling attention
        self.proj_ling_v = nn.Linear(linguistic_dim, d_model)  # V for ac→ling attention

        # ------------------------------------------------------------------
        # Multi-head attention modules
        # batch_first=True so shapes are (B, seq_len, d_model)
        # ------------------------------------------------------------------
        self.attn_ac_to_ling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ling_to_ac = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # Layer norms (post-attention)
        # ------------------------------------------------------------------
        self.norm_ac = nn.LayerNorm(d_model)
        self.norm_ling = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # Optional output projection after fusion (identity-like by default)
        # ------------------------------------------------------------------
        self.output_proj = nn.Linear(2 * d_model, 2 * d_model)

        # Initialise weights
        self._init_weights()

        logger.debug(
            "CrossModalAttention: "
            "acoustic_dim=%d, linguistic_dim=%d, d_model=%d, "
            "num_heads=%d, output_dim=%d",
            acoustic_dim,
            linguistic_dim,
            d_model,
            num_heads,
            self.output_dim,
        )

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for all linear projection layers."""
        for module in [
            self.proj_ac_q, self.proj_ac_k, self.proj_ac_v,
            self.proj_ling_q, self.proj_ling_k, self.proj_ling_v,
            self.output_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        f_ac: torch.Tensor,
        f_ling: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor:
        """
        Fuse acoustic and linguistic embeddings via bidirectional cross-attention.

        Parameters
        ----------
        f_ac : torch.Tensor, shape (B, 768)
            Pooled acoustic embedding from HuBERT (F_ac_pooled).
        f_ling : torch.Tensor, shape (B, 768)
            [CLS] linguistic embedding from BERT (F_ling).
        return_attention_weights : bool
            If True, also returns a tuple of attention weight tensors for
            interpretability / XAI use. Default: False.

        Returns
        -------
        f_fused : torch.Tensor, shape (B, 512)
            Fused multi-modal embedding.
        attn_weights : tuple of (torch.Tensor, torch.Tensor), optional
            Returned only when ``return_attention_weights=True``.
            (ac_to_ling_weights, ling_to_ac_weights), each (B, 1, 1).
        """
        # Reshape from (B, dim) → (B, 1, d_model) for MultiheadAttention
        # (sequence length = 1 since we have a single pooled vector per modality)

        # -- Acoustic projections --
        q_ac = self.proj_ac_q(f_ac).unsqueeze(1)    # (B, 1, d_model)
        k_ac = self.proj_ac_k(f_ac).unsqueeze(1)    # (B, 1, d_model)
        v_ac = self.proj_ac_v(f_ac).unsqueeze(1)    # (B, 1, d_model)

        # -- Linguistic projections --
        q_ling = self.proj_ling_q(f_ling).unsqueeze(1)  # (B, 1, d_model)
        k_ling = self.proj_ling_k(f_ling).unsqueeze(1)  # (B, 1, d_model)
        v_ling = self.proj_ling_v(f_ling).unsqueeze(1)  # (B, 1, d_model)

        # -- Acoustic-to-Linguistic attention --
        # Q = acoustic, K/V = linguistic
        # "Which linguistic features are most relevant given my acoustic input?"
        ac_attended, attn_w_ac = self.attn_ac_to_ling(
            query=q_ac,
            key=k_ling,
            value=v_ling,
            need_weights=True,
        )
        # ac_attended: (B, 1, d_model) → squeeze → (B, d_model)
        ac_attended = ac_attended.squeeze(1)
        ac_attended = self.norm_ac(ac_attended)  # LayerNorm

        # -- Linguistic-to-Acoustic attention --
        # Q = linguistic, K/V = acoustic
        # "Which acoustic properties best match my linguistic content?"
        ling_attended, attn_w_ling = self.attn_ling_to_ac(
            query=q_ling,
            key=k_ac,
            value=v_ac,
            need_weights=True,
        )
        # ling_attended: (B, 1, d_model) → squeeze → (B, d_model)
        ling_attended = ling_attended.squeeze(1)
        ling_attended = self.norm_ling(ling_attended)  # LayerNorm

        # -- Concatenate both attended representations --
        f_fused = torch.cat([ac_attended, ling_attended], dim=-1)  # (B, 512)

        # -- Optional output projection --
        f_fused = self.output_proj(f_fused)                        # (B, 512)
        f_fused = F.relu(f_fused)                                  # Non-linearity

        if return_attention_weights:
            return f_fused, (attn_w_ac, attn_w_ling)
        return f_fused
