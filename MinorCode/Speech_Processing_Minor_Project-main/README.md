# Speech Processing Branch — Alzheimer's Disease Detection

A complete, modular PyTorch implementation of the **Speech Processing Branch** for multimodal Alzheimer's Disease (AD) detection using the **DementiaBank Pitt Corpus**.

---

## Architecture Overview

```
Raw audio (.wav/.mp3)          CHAT transcript (.cha)
        │                               │
        ▼                               ▼
 AudioProcessor                  CHAT Parser
 (pydub, librosa)             (regex + cleaning)
        │                               │
        ▼                               ▼
  HuBERT (frozen)              BERT (frozen)
  ────────────────             ──────────────
  F_ac ∈ R^768                F_ling ∈ R^768
        │                               │
        └──────────┬────────────────────┘
                   ▼
         CrossModalAttention
         ───────────────────
           F_fused ∈ R^512
                   │
                   ▼
               SDSHead (MLP)
               ─────────────
           SDS ∈ [0, 1]   ←─── 0 = Control (CN) │ 1 = Dementia (AD)
```

**Trainable modules**: `CrossModalAttention` + `SDSHead`  
**Frozen encoders**: `HuBERT` (facebook/hubert-base-ls960) + `BERT` (bert-base-uncased)

---

## Project Structure

```
speech_branch/
├── data/
│   └── dataset.py              # DementiaBankDataset (PyTorch Dataset)
├── preprocessing/
│   ├── audio_processor.py      # Audio load, resample, log-Mel spec, eGeMAPS
│   ├── chat_parser.py          # CHAT .cha file parser + linguistic features
│   └── linguistic_encoder.py   # BertTokenizer preprocessing wrapper
├── models/
│   ├── acoustic_encoder.py     # HuBERT wrapper with chunk-based pooling
│   ├── linguistic_encoder.py   # BERT wrapper, [CLS] extraction
│   ├── cross_attention.py      # Bidirectional CrossModalAttention module
│   └── sds_head.py             # SDSHead MLP + full SpeechBranchModel
├── training/
│   ├── train.py                # Full training loop with early stopping
│   └── evaluate.py             # Evaluation (Acc, F1, AUC, Sens, Spec)
├── utils/
│   ├── config.py               # All hyperparameters as a dataclass
│   └── helpers.py              # Seed, logging, checkpointing utilities
├── inference.py                # Single-patient inference entry point
├── requirements.txt
└── README.md
```

---

## Dataset Setup

Download the **DementiaBank Pitt Corpus** from [TalkBank](https://dementia.talkbank.org/) (requires registration).

Arrange files in the following structure:

```
data/DementiaBank/
├── audio/
│   ├── Dementia/
│   │   ├── 001-1c.wav
│   │   └── ...
│   └── Control/
│       ├── 101-0.wav
│       └── ...
└── transcripts/
    ├── Dementia/
    │   ├── 001-1c.cha
    │   └── ...
    └── Control/
        ├── 101-0.cha
        └── ...
```

> **Note:** Audio files can be `.wav` or `.mp3`. MP3 files are automatically converted to WAV using `pydub`.  
> **Note:** Participant IDs link audio to transcript (e.g., `001-1c.wav` ↔ `001-1c.cha`).

---

## Installation

```bash
# 1. Clone / navigate to the speech_branch directory
cd speech_branch

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install PyTorch with CUDA support
#    Visit https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **ffmpeg** is required by `pydub` for MP3 conversion.  
> Install via: `sudo apt install ffmpeg` (Linux) or `winget install ffmpeg` (Windows).

---

## Training

Run training from the `speech_branch/` root directory:

```bash
# Default training (reads all parameters from utils/config.py)
python -m training.train

# Override common parameters via CLI
python -m training.train \
    --epochs 50 \
    --batch_size 8 \
    --lr 5e-5 \
    --audio_dir data/DementiaBank/audio \
    --transcript_dir data/DementiaBank/transcripts \
    --checkpoint_dir checkpoints/

# Skip eGeMAPS extraction for faster iteration
python -m training.train --no_egemaps
```

Training will:
1. Auto-detect GPU (CUDA → MPS → CPU)
2. Perform stratified 80/10/10 split
3. Apply `WeightedRandomSampler` for class-imbalance correction
4. Train with `AdamW` + `CosineAnnealingLR`
5. Save best checkpoint (by validation AUC) to `checkpoints/best_speech_branch.pt`
6. Apply early stopping (patience=5)
7. Print a per-epoch training summary table
8. Run final evaluation on the test set

### Key Hyperparameters (`utils/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | AdamW learning rate |
| `weight_decay` | `1e-2` | AdamW weight decay |
| `batch_size` | `16` | Training batch size |
| `num_epochs` | `30` | Maximum training epochs |
| `early_stopping_patience` | `5` | Epochs without AUC improvement |
| `d_model` | `256` | Cross-attention projection dim |
| `num_heads` | `4` | Multi-head attention heads |
| `sds_dropout` | `0.3` | Dropout in SDSHead |
| `chunk_size_seconds` | `30` | Max seconds per HuBERT chunk |

---

## Inference

### Single-patient inference (CLI)

```bash
python inference.py \
    --audio_path     data/DementiaBank/audio/Dementia/001-1c.wav \
    --transcript_path data/DementiaBank/transcripts/Dementia/001-1c.cha \
    --checkpoint     checkpoints/best_speech_branch.pt \
    --output_json    results/001-1c_output.json
```

### From Python (downstream pipeline integration)

```python
from inference import run_speech_branch

result = run_speech_branch(
    audio_path="data/DementiaBank/audio/Dementia/001-1c.wav",
    transcript_path="data/DementiaBank/transcripts/Dementia/001-1c.cha",
    checkpoint_path="checkpoints/best_speech_branch.pt",
)

# Primary outputs for downstream modules
sds: float       = result["SDS"]           # Speech Diagnostic Score ∈ [0, 1]
f_fused          = result["F_fused"]       # np.ndarray shape (512,)
f_ac             = result["F_ac"]          # np.ndarray shape (768,)
f_ling           = result["F_ling"]        # np.ndarray shape (768,)

# Supplementary outputs
ling_feats: dict = result["linguistic_features"]         # TTR, filler rate, etc.
egemaps          = result["acoustic_features_egemaps"]   # np.ndarray shape (88,)
patient_text     = result["patient_text"]                # Clean CHAT speech
```

### Output dictionary schema

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `SDS` | `float` | scalar | Speech Diagnostic Score ∈ [0,1] |
| `F_fused` | `np.ndarray` | `(512,)` | Fused embedding for Cross-Modal module |
| `F_ac` | `np.ndarray` | `(768,)` | Raw HuBERT acoustic embedding |
| `F_ling` | `np.ndarray` | `(768,)` | Raw BERT linguistic embedding |
| `linguistic_features` | `dict` | — | TTR, filler_count, lexical_density, etc. |
| `acoustic_features_egemaps` | `np.ndarray` | `(88,)` | eGeMAPS v02 feature vector |
| `patient_text` | `str` | — | Clean patient speech from CHAT |

---

## Linguistic Features Computed

| Feature | Description |
|---------|-------------|
| `type_token_ratio` | Vocabulary diversity (unique / total words) |
| `mean_utterance_length` | Mean words per *PAR: turn |
| `total_utterances` | Number of patient speaker turns |
| `total_words` | Total word count in patient speech |
| `filler_count` | Count of uh, um, er, hmm, hm, ah |
| `filler_rate` | filler_count / total_words |
| `lexical_density` | Content words / total words |
| `unique_noun_count` | Unique nouns (lexicon-based heuristic) |
| `unique_verb_count` | Unique verbs (lexicon-based heuristic) |
| `brunet_w_index` | Brunét's W vocabulary richness index |
| `honore_r_statistic` | Honoré's R vocabulary richness statistic |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct classification rate |
| F1-Score | Binary F1, positive class = AD |
| AUC-ROC | Area under ROC curve (primary checkpoint metric) |
| Sensitivity | True Positive Rate (recall for AD class) |
| Specificity | True Negative Rate (recall for CN class) |

---

## Model Summary

| Component | Parameters | Status |
|-----------|-----------|--------|
| HuBERT (acoustic encoder) | ~94M | **Frozen** |
| BERT (linguistic encoder) | ~110M | **Frozen** |
| CrossModalAttention | ~790K | **Trainable** |
| SDSHead | ~68K | **Trainable** |
| **Total trainable** | **~858K** | |

---

## Configuration

All hyperparameters are centralised in `utils/config.py` as a dataclass.
Edit values there — no command-line flags needed for most settings.

```python
from utils.config import Config

cfg = Config()
cfg.training.learning_rate = 5e-5
cfg.model.d_model = 512
cfg.audio.chunk_size_seconds = 20
```

---

## Downstream Integration

The `F_fused` embedding `(512,)` and `SDS` scalar are the primary outputs
consumed by downstream pipeline modules:

- **Cross-Modal Analysis Module**: concatenates `F_fused` with `F_mri` from the Neuroimaging Branch
- **Longitudinal DTW Module**: stacks `F_fused` across patient visits for temporal analysis
- **CARE-AD LLM Fusion**: provides `SDS`, `F_fused`, and `linguistic_features` to the LLM fusion layer
- **XAI Module**: uses attention weights from `CrossModalAttention` for Grad-CAM / SHAP explanations

---

## Reproducibility

Seed is set globally via `utils/helpers.set_seed()` using the value in `config.repro.seed` (default: 42).
This covers Python `random`, `numpy`, and `torch` including CUDA.

---

## License

For research use only. DementiaBank data requires a separate data use agreement from TalkBank.
