# speech_branch/__init__.py
"""
Speech Processing Branch
========================
Multimodal Alzheimer's Disease Detection Pipeline — Speech Module.

Outputs:
  - SDS  : Speech Diagnostic Score ∈ [0, 1]
  - F_fused : Fused acoustic-linguistic embedding ∈ R^{512}
  - F_ac    : Raw HuBERT acoustic embedding ∈ R^{768}
  - F_ling  : Raw BERT linguistic embedding ∈ R^{768}
"""

from inference import run_speech_branch

__all__ = ["run_speech_branch"]
__version__ = "1.0.0"
