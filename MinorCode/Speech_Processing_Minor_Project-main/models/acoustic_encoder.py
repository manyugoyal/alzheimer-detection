"""
acoustic_encoder.py
====================
HuBERT-based acoustic encoder for the Speech Processing Branch.

Architecture
------------
  Input  : Raw 16kHz float32 waveform batches (variable length)
  Step 1 : AutoFeatureExtractor normalises and pads to HuBERT input spec
  Step 2 : HubertModel (frozen) produces hidden states of shape (B, T, 768)
  Step 3 : Mean pooling over T → (B, 768)  = F_ac_pooled

For batches with long audio, each sample's waveform is processed independently
via chunk-based averaging to avoid OOM on CPU.

Chunking strategy
-----------------
If a waveform is longer than `chunk_size_seconds`, it is split into 30-second
chunks. HuBERT runs on each chunk separately; the resulting (T_chunk, 768)
outputs are mean-pooled per chunk, then averaged across chunks to yield the
final (768,) embedding.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, HubertModel

logger = logging.getLogger(__name__)


class AcousticEncoder(nn.Module):
    """
    Frozen HuBERT feature extractor that converts raw 16kHz waveforms into
    pooled acoustic embeddings F_ac_pooled ∈ R^{768}.

    This class wraps HubertModel with:
      - Automatic feature extraction via AutoFeatureExtractor
      - Mean pooling over the time dimension
      - Optional chunk-based processing for long audio

    Parameters
    ----------
    model_name : str
        HuggingFace model hub identifier.
        Default: 'facebook/hubert-base-ls960'
    sample_rate : int
        Expected waveform sample rate. Must match HuBERT's expectation (16000).
    freeze : bool
        If True (default), all HuBERT parameters are frozen.
        Set False to enable fine-tuning (not recommended for data-scarce settings).
    chunk_size_seconds : int
        Maximum seconds of audio processed per forward chunk.
        Audio longer than this is split and averaged.
    device : torch.device or str
        Device to move the model to. Default CPU.
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        sample_rate: int = 16000,
        freeze: bool = True,
        chunk_size_seconds: int = 30,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.sample_rate = sample_rate
        self.chunk_size_seconds = chunk_size_seconds

        if device is None:
            device = torch.device("cpu")
        self.device = device

        logger.info("Loading HuBERT feature extractor: %s", model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name,
            return_attention_mask=True,
            sampling_rate=sample_rate,
        )

        logger.info("Loading HuBERT model: %s", model_name)
        self.hubert = HubertModel.from_pretrained(model_name)
        self.hubert = self.hubert.to(device)

        # Freeze HuBERT weights if requested
        if freeze:
            self._freeze_hubert()
            logger.info("HuBERT weights frozen — will not be updated during training.")

        # Number of output features (matches HuBERT-base hidden size)
        self.output_dim = self.hubert.config.hidden_size  # 768

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _freeze_hubert(self) -> None:
        """Disable gradient computation for all HuBERT parameters."""
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.hubert.eval()

    def _encode_single_waveform(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Run HuBERT on a single 1-D float32 waveform and return the mean-pooled
        last hidden state.

        Parameters
        ----------
        waveform : np.ndarray, shape (T,)
            Single channel, 16kHz float32 audio signal.

        Returns
        -------
        torch.Tensor, shape (768,)
            Mean-pooled acoustic embedding for this waveform.
        """
        # AutoFeatureExtractor normalises and returns a BatchFeature
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        # Attention mask may not always be present for single inputs
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.hubert(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        # outputs.last_hidden_state: (1, T_frames, 768)
        last_hidden = outputs.last_hidden_state  # (1, T, 768)

        # If attention mask is available, use it to ignore padding positions
        if attention_mask is not None:
            # HuBERT's attention mask is over raw waveform frames, not
            # transformer frames — we use a simple mean here
            pooled = last_hidden.mean(dim=1).squeeze(0)  # (768,)
        else:
            pooled = last_hidden.mean(dim=1).squeeze(0)  # (768,)

        return pooled  # (768,)

    def _encode_with_chunking(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Split `waveform` into `chunk_size_seconds`-long chunks, run HuBERT on
        each, and return the mean of all chunk embeddings.

        Parameters
        ----------
        waveform : np.ndarray, shape (T,)
            Full audio waveform.

        Returns
        -------
        torch.Tensor, shape (768,)
            Chunk-averaged acoustic embedding.
        """
        chunk_size = self.chunk_size_seconds * self.sample_rate
        total = len(waveform)

        chunk_embeddings: List[torch.Tensor] = []
        start = 0
        while start < total:
            end = min(start + chunk_size, total)
            chunk = waveform[start:end]
            emb = self._encode_single_waveform(chunk)
            chunk_embeddings.append(emb)
            start += chunk_size

        # Stack and average → (768,)
        stacked = torch.stack(chunk_embeddings, dim=0)  # (n_chunks, 768)
        mean_emb = stacked.mean(dim=0)                  # (768,)
        return mean_emb

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def encode_waveform(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Encode a single raw waveform (numpy) and return F_ac_pooled as a
        (768,) CPU tensor.

        Automatically uses chunking if the waveform is longer than
        ``chunk_size_seconds``.

        Parameters
        ----------
        waveform : np.ndarray, shape (T,)
            16kHz float32 mono waveform.

        Returns
        -------
        torch.Tensor, shape (768,)
        """
        threshold = self.chunk_size_seconds * self.sample_rate
        if len(waveform) > threshold:
            return self._encode_with_chunking(waveform)
        else:
            return self._encode_single_waveform(waveform)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Batch forward pass: encode a batch of padded waveforms.

        Processes each sample in the batch independently (to handle variable
        lengths), then stacks the result.

        Parameters
        ----------
        waveforms : torch.Tensor, shape (B, T)
            Batch of zero-padded 16kHz waveforms. Values should be in [-1, 1].

        Returns
        -------
        torch.Tensor, shape (B, 768)
            Batch of mean-pooled acoustic embeddings F_ac_pooled.
        """
        batch_embeddings: List[torch.Tensor] = []

        for i in range(waveforms.shape[0]):
            wf_np = waveforms[i].cpu().numpy()  # (T,)

            # Strip trailing zeros (padding)
            nonzero_mask = wf_np != 0
            if nonzero_mask.any():
                last_nonzero = np.where(nonzero_mask)[0][-1] + 1
                wf_np = wf_np[:last_nonzero]
            else:
                # All-zero waveform (should not happen in practice)
                logger.warning("All-zero waveform at batch index %d.", i)
                wf_np = wf_np

            emb = self.encode_waveform(wf_np)  # (768,) on self.device
            batch_embeddings.append(emb)

        # Stack → (B, 768) on self.device
        return torch.stack(batch_embeddings, dim=0)
