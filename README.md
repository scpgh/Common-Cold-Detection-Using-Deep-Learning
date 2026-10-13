# Speech-Based Common Cold Detection — URTIC Dataset

> **Final Year Project** | Binary cold vs. non-cold paralinguistic classification on the [INTERSPEECH 2017 ComParE Cold Sub-Challenge (URTIC)](https://www.isca-speech.org/archive/interspeech_2017/) corpus using Conformer encoders, feature fusion, and ensemble methods.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Notebook Descriptions](#notebook-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Overview

This project implements multiple deep-learning and machine-learning pipelines to detect common cold symptoms from speech audio using the **URTIC (Upper Respiratory Tract Infection Corpus)**. The core challenge is a **heavily imbalanced binary classification task** (cold ≈ 10%, non-cold ≈ 90%) evaluated using **UAR (Unweighted Average Recall = Balanced Accuracy)**.

Each notebook explores a different feature representation and model architecture, ranging from classic MFCC-based Transformers to novel Conformer models with domain-grounded nasality features.

---

## Dataset

**URTIC** (Upper Respiratory Tract Infection Corpus) from INTERSPEECH 2017 ComParE Challenge.

```
URTIC/
├── train/
│   ├── cold/         # ~970 WAV files
│   └── noncold/      # ~8,532 WAV files
├── develop/
│   ├── cold/         # ~1,011 WAV files
│   └── noncold/      # ~8,585 WAV files
└── test/
    ├── cold/         # ~894 WAV files
    └── noncold/      # ~8,655 WAV files
```

**Total:** ~28,647 audio files | **Class imbalance:** ~1:9 (cold:noncold)

> ⚠️ Update `DATASET_PATH` or `URTIC_ROOT` in each notebook to point to your local URTIC directory.

---

## Repository Structure

```
codes/
├── mfcc.ipynb                              # Baseline: MFCC + Transformer (TensorFlow)
├── ColdDetection_Breathiness_FIXED.ipynb   # Breathiness features + Transformer (PyTorch)
├── combined_feat1.ipynb                    # TEO + Voice features + Conformer
├── Combined_Feat2.ipynb                    # Multi-feature fusion + Conformer (full pipeline)
├── URTIC_Nasality_Only_Conformer.ipynb     # Dynamic Nasality + Conformer (novel contribution)
├── nonLinearFeaturesConformer_optimized.ipynb # TEO nonlinear features + Conformer (paper-faithful)
├── Ensemble_model.ipynb                    # LightGBM + Transformer ensemble
└── install_packages.bat                    # Windows dependency installer
```

---

## Notebook Descriptions

### 1. `mfcc.ipynb` — MFCC Baseline with Transformer (TensorFlow)

**Purpose:** Establishes a baseline using standard MFCC features with a multi-head attention Transformer.

| Property | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Features | 13 MFCCs + Δ + ΔΔ → 39-D per frame |
| Model | 3-layer Transformer Encoder (d=256, 8 heads) |
| Loss | Focal Loss (α=0.25, γ=2.0) |
| Imbalance Handling | Class weights |
| Input Shape | `(300 frames, 39 features)` |
| Params | ~1.36M |

**Key steps:**
1. Pre-emphasis → VAD trim → MFCC + delta + delta² extraction
2. Pad/truncate to 300 frames
3. StandardScaler normalization
4. Transformer Encoder → GlobalAveragePooling → Dense classifier

---

### 2. `ColdDetection_Breathiness_FIXED.ipynb` — Breathiness Features + Transformer (PyTorch)

**Purpose:** Extracts physiologically-motivated breathiness features, with several critical bug fixes over a baseline implementation.

| Property | Value |
|---|---|
| Framework | PyTorch |
| Features | 27-D handcrafted (PPQ, APQ, HNR, GNER, HE, spectral, MFCC stats, pitch) |
| Model | Transformer Encoder (d=256, 8 heads, dropout=0.4) |
| Loss | Focal Loss |
| Imbalance Handling | SMOTE + ROC/Youden-J threshold |
| Dataset | 28,646 files processed |

**Key bug fixes over baseline:**
- `librosa.lpc.lpc_to_lsf` → replaced with `scipy.signal.lfilter` for GNER computation
- `pyin` computed **once per file** (not per frame) — achieves 10–50× speedup
- `copy.deepcopy` for model `state_dict` (shallow copy silently broke model restore)
- UAR metric used instead of plain accuracy for imbalanced evaluation

**Features extracted:**
| Feature | Description |
|---|---|
| PPQ | Period Perturbation Quotient (voice shimmer proxy) |
| APQ | Amplitude Perturbation Quotient (vectorised) |
| HNR | Harmonic-to-Noise Ratio (via autocorrelation) |
| GNER | Glottal-to-Noise Excitation Ratio (LPC residual) |
| HE | H1/H2 Harmonic Energy Ratio |
| ZCR, Spectral features | Zero-crossing rate, centroid, rolloff, bandwidth, RMS |
| MFCC (1–13) | Mean + Std of 13 MFCCs |
| Pitch stats | F0 mean/std, voiced fraction, jitter |

---

### 3. `combined_feat1.ipynb` — TEO + Voice Features + Conformer (v1)

**Purpose:** Combines MFCC/delta features with TEO (Teager Energy Operator) nonlinear features and lightweight voice quality proxies in a compact Conformer architecture.

| Property | Value |
|---|---|
| Framework | PyTorch |
| Features | 65-D per frame (MFCC×39 + voice×8 + TEO×18) |
| Model | 2-layer Conformer (d=80, 4 heads, temporal_pool=4) |
| Loss | Focal Loss (γ=1.5) |
| Imbalance Handling | WeightedRandomSampler |
| Caching | Per-file `.npz` with SHA-256 fingerprint validation |

**65-D Feature breakdown:**
- **39 dims:** MFCC (13) + Δ-MFCC (13) + ΔΔ-MFCC (13)
- **8 dims:** log-RMS, ZCR, spectral centroid, rolloff@85%, flatness, F0, HNR proxy, CPP proxy
- **18 dims:** TEO FM-variance (1) + TEO autocorrelation envelope area (1) + 16 TEO critical-band features

**Dev UAR achieved:** ~64.3%

---

### 4. `Combined_Feat2.ipynb` — Multi-Feature Fusion + Conformer (Full Pipeline)

**Purpose:** A complete self-contained pipeline combining MFCC/delta, log-Mel, pitch/voice quality, nasality, and temporal modulation features, all fed into a Conformer encoder.

| Property | Value |
|---|---|
| Framework | PyTorch |
| Features | ~130-D per frame (see below) |
| Model | Conformer (d=128, 4 heads, 3 layers) with PCA to 64-D |
| Loss | Focal Loss (γ=2.0) |
| Imbalance Handling | WeightedRandomSampler |
| Normalization | Train-only StandardScaler + IncrementalPCA |
| Caching | Atomic `.tmp.npz` → `.npz` writes with config fingerprinting |

**Feature groups:**
| Group | Features |
|---|---|
| MFCCs | 13 coefficients + Δ + ΔΔ (39 total) |
| Log-Mel | 40 mel-filterbank energies |
| Pitch & Voice Quality | F0, F0 stats, HNR, CPPS, jitter, shimmer |
| Formants | F1, F2, F3 (LPC-based) |
| Spectral | Tilt, flux, entropy, LF/HF ratio |
| Nasality (NRE, NOER) | Nasal resonance energy, antiformant frequency/depth, NCI components |
| Temporal Modulation | Modulation spectrum, DCT modulation, modulation energy/entropy |

**Test UAR achieved:** ~65.8%

---

### 5. `URTIC_Nasality_Only_Conformer.ipynb` — Dynamic Nasality + Conformer ⭐ *Novel Contribution*

**Purpose:** Domain-grounded dynamic nasality feature pipeline with a Conformer classifier using Gated Attention Pooling. This is the primary research contribution of the project.

| Property | Value |
|---|---|
| Framework | PyTorch |
| Features | 81-D per frame (nasality-focused, 4 feature groups) |
| Model | Conformer + **Gated Attention Pooling** (Ilse et al., 2018) |
| Loss | Focal Loss / Weighted CE (ablatable) |
| Imbalance Handling | WeightedRandomSampler + Focal Loss |
| Reproducibility | 5-seed evaluation (2026–2030), mean ± std reporting |
| Caching | 2-stage fingerprint-validated cache with atomic writes |

**Key architectural contributions:**

1. **Dynamic Nasality Trajectory Modeling** — extends classical static measures (A₁-P₀, A₁-P₁) with:
   - **Distributional functionals** (Group A): 11 statistics over NRE trajectory
   - **Trajectory dynamics** (Group B): velocity, acceleration, transition rate, persistence ratio
   - **Markovian state transitions** (Group C): 3-state (low/med/high) transition entropy and run lengths
   - **Cross-feature coupling** (Group D): temporal cross-correlation with lag asymmetry between NRE and antiformant trajectories

2. **Gated Attention Pooling** — learned temporal focus on nasalization anomalies (vs. naive mean/max pooling)

3. **Two-stage robust caching** — prevents stale cache and eliminates DataLoader bottlenecks

**Feature schema (81-D):**

| Group | Count | Description |
|---|---|---|
| Core nasality frame proxies | 20 | NRE, NOER, P0/P1 frequency, A1-P0/A1-P1, antiformant, NHA, HAR, NFD, coupling composite |
| Trajectory derivatives + persistence | 6 | NRE velocity/acceleration, NFD velocity/acceleration, nasality state/persistence |
| Local sliding-window context | 4 | 15-frame local mean and std of NRE and antiformant depth |
| Modulation spectrum | 16 | 4 band energies, centroid, spread, entropy, flatness + 8 DCT coefficients |
| Group A: Distributional | 11 | mean, median, std, IQR, CV, skew, kurtosis, P10, Q25, P90, entropy |
| Group B: Dynamics | 10 | velocity stats, excursions, transition rate, onset/recovery duration |
| Group C: Markovian | 12 | state occupancy, transition probabilities, entropy, max run lengths |
| Group D: Cross-feature coupling | 8 | Pearson, Spearman, optimal lag, max correlation, lag asymmetry |

**Ablation study options:**

| Config | Pooling | Loss | Use |
|---|---|---|---|
| A | mean | focal | Baseline |
| B | gated_attention | focal | Gated pooling effect |
| C | gated_attention | weighted_ce | Loss function comparison |
| D | mean | weighted_ce | Full ablation |

---

### 6. `nonLinearFeaturesConformer_optimized.ipynb` — TEO Nonlinear Features + Conformer (Paper-Faithful)

**Purpose:** Paper-faithful implementation of the Zhou-Hansen-Kaiser (2001) TEO feature extraction pipeline with a Conformer classifier. Audits and fixes issues from earlier notebooks.

| Property | Value |
|---|---|
| Framework | PyTorch |
| Features | 21-D TEO per frame (1 FM-var + 4 autocorr-env + 16 critical-band) |
| Model | Conformer (d=64, 4 heads, 2 layers, depthwise conv kernel=9) |
| Pooling | mean_max |
| Loss | Weighted BCE / Focal Loss |
| Imbalance Handling | Hard Negative Mining (from epoch 3) |
| Target UAR | 0.80 |
| Caching | Per-file `.npz` (versioned) with joblib parallel extraction |

**TEO Feature breakdown (21-D per frame):**
| Feature | Description |
|---|---|
| `teo_fm_var` (1-D) | FM-frequency variance from Teager Energy Operator |
| `teo_auto_env` (4-D) | Autocorrelation-envelope area in 4 frequency bands (0–1kHz, 1–2kHz, 2–3kHz, 3–4kHz) |
| `teo_cb_auto_env` (16-D) | Critical-band ACF-envelope using 16 Gabor-filtered bands (140–3400 Hz) |

**Audited fixes vs. prior implementations:**
- Fixed double imbalance compensation (balanced sampler + cold-heavy focal loss was double-counting)
- Fixed non-versioned compressed split cache (now uses `PREPROCESSING_VERSION` tag)
- Fixed attention masking (padding-safe mask-safe pooling)
- Fixed potential test leakage in threshold selection (dev-only Youden-J)

---

### 7. `Ensemble_model.ipynb` — LightGBM + Transformer Ensemble (71%+ UAR Target)

**Purpose:** Speed-optimized ensemble combining LightGBM (tabular) and Transformer (sequence) models with 4× augmentation of cold training samples.

| Property | Value |
|---|---|
| Framework | PyTorch + LightGBM |
| Features | ~90-D tabular (MFCC×78 + nonlinear×7 + harmonic×3 + voice×1) |
| Models | LightGBM (n_estimators=500) + Transformer Encoder |
| Ensemble | Weighted average (LGBM-heavy) |
| Augmentation | 4× cold samples: noise, time-stretch ×2, pitch shift |
| Parallel extraction | joblib (loky backend, 6 workers) |
| Target UAR | ≥71% |

**Feature groups (FastFeatureExtractor):**
| Group | Features |
|---|---|
| MFCC | 13 coefficients × 3 (mean+std) × 3 (mfcc, Δ, ΔΔ) = 78 |
| Nonlinear | Shannon entropy, spectral entropy, Rényi entropy, TEO mean/std, approx entropy, sample entropy, permutation entropy |
| Harmonic | Harmonic energy, HSR (harmonic-to-signal ratio), HERS |
| Voice | HNR (via autocorrelation) |

**Augmentation pipeline (cold train only, 4× per file):**
1. Gaussian noise at SNR 20–30 dB
2. Time-stretch at rate 0.9
3. Time-stretch at rate 1.1
4. Pitch shift +1 semitone

**71% guard:** If dev UAR < 71%, LightGBM is retrained with more trees automatically.

---

## Installation

### Option 1: Windows batch script
```bat
install_packages.bat
```

### Option 2: Manual pip install
```bash
pip install numpy pandas scipy librosa soundfile scikit-learn \
            torch torchaudio imbalanced-learn matplotlib seaborn \
            tqdm scipy joblib lightgbm tensorflow
```

### Python version
All notebooks were tested on **Python 3.13** (Windows). Some notebooks use Python 3.14 syntax.

---

## Usage

### Step 1: Set your dataset path

In each notebook, update the dataset path variable:
```python
# In most notebooks:
DATASET_PATH = r"E:\Your\Path\To\URTIC"
# or
URTIC_ROOT = Path(r"E:\Your\Path\To\URTIC")
```

### Step 2: Run notebook cells top-to-bottom

All notebooks are designed to be **self-contained** — restart the kernel and run all cells in order. Feature caches are built on first run and reused on subsequent runs.

### Step 3: Monitor caching

First-run feature extraction can take **30 minutes to several hours** depending on the notebook and your hardware. Subsequent runs use cached features and are nearly instant.

---

## Results Summary

| Notebook | Features | Model | Dev UAR | Test UAR |
|---|---|---|---|---|
| `mfcc.ipynb` | MFCC+Δ+ΔΔ (39-D) | Transformer (TF) | — | — |
| `ColdDetection_Breathiness_FIXED.ipynb` | Breathiness (27-D) | Transformer (PyTorch) | — | — |
| `combined_feat1.ipynb` | TEO+Voice (65-D) | Conformer | ~64.3% | ~62.7% |
| `Combined_Feat2.ipynb` | Multi-feature (130-D→64-D PCA) | Conformer | ~67.9% | ~65.8% |
| `URTIC_Nasality_Only_Conformer.ipynb` | Dynamic Nasality (81-D) | Conformer + Gated Attn | — | — |
| `nonLinearFeaturesConformer_optimized.ipynb` | TEO (21-D) | Conformer | — | — |
| `Ensemble_model.ipynb` | Tabular (90-D) | LGBM + Transformer | ≥71% target | — |

> **Metric:** UAR = Unweighted Average Recall = Balanced Accuracy. Required for imbalanced cold dataset.
> Results may vary by run; use development-set Youden-J threshold for final test evaluation.

---

## Key Design Decisions

### Imbalance handling
The URTIC dataset has a **~1:9 cold:noncold ratio**. Every notebook addresses this via one or more of:
- **SMOTE** oversampling (Breathiness notebook)
- **WeightedRandomSampler** (Conformer notebooks)
- **Focal Loss** (γ=1.5–2.0) — down-weights easy negatives
- **Youden-J threshold selection** on dev set — optimal ROC operating point

### Evaluation protocol
- **Train**: train split only
- **Threshold selection**: development set only (Youden-J index on ROC curve)
- **Final reporting**: test set with dev-selected threshold
- **No test labels consulted** during training or threshold selection

### Feature caching
All computationally expensive feature extractions use **per-file `.npz` caching** with SHA-256 fingerprinting of the config. Stale caches are automatically detected and rebuilt.

### Train-only normalization
All `StandardScaler` and `PCA` fits are performed **on the training split only**, then applied to dev/test. This prevents data leakage.

---

## Dependencies

| Package | Version | Used in |
|---|---|---|
| `torch` | ≥2.2 | All PyTorch notebooks |
| `torchaudio` | ≥2.2 | Breathiness notebook |
| `tensorflow` | ≥2.x | `mfcc.ipynb` |
| `librosa` | ≥0.10 | All notebooks |
| `soundfile` | ≥0.12 | All notebooks |
| `numpy` | ≥1.26 | All notebooks |
| `pandas` | ≥2.1 | All notebooks |
| `scikit-learn` | ≥1.3 | All notebooks |
| `scipy` | ≥1.11 | Most notebooks |
| `imbalanced-learn` | ≥0.14 | Breathiness + Ensemble |
| `lightgbm` | ≥4.0 | `Ensemble_model.ipynb` |
| `joblib` | ≥1.3 | Most notebooks |
| `tqdm` | ≥4.66 | All notebooks |
| `matplotlib` | ≥3.7 | All notebooks |
| `seaborn` | ≥0.12 | Most notebooks |

---

## References

- Schuller, B. et al. (2017). *The INTERSPEECH 2017 Computational Paralinguistics Challenge: Addressee, Cold & Snoring.* INTERSPEECH 2017.
- Zhou, G., Hansen, J.H.L., Kaiser, J.F. (2001). *Nonlinear feature based classification of speech under stress.* IEEE TSAP.
- Ilse, M., Tomczak, J.M., Welling, M. (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
- Gulati, A. et al. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition.* INTERSPEECH.
