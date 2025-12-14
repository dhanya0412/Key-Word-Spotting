
# Keyword Spotting (Speech Command Detection)

A machine learning project for **noise-resilient keyword spotting (KWS)** that detects short spoken commands from audio.  
This project presents a **comparative evaluation of classical and deep learning models** for real-world speech command recognition under clean and noisy conditions.

The system is designed with a strong emphasis on **robustness, low latency, and practical deployment scenarios** such as smart assistants, IoT devices, and voice-controlled interfaces.

---

## Project Overview

Keyword Spotting (KWS) enables devices to continuously listen for predefined voice commands and trigger actions when detected.  
Real-world environments introduce challenges such as background noise, speaker variability, and microphone distortion.

In this project, we:
- Compare **classical acoustic models** with **modern deep learning architectures**
- Evaluate performance under **clean and noise-augmented audio**
- Propose and validate a **BiLSTM–DenseNet hybrid architecture** for robust keyword detection

---

## Dataset

- **Google Speech Commands Dataset v2**
- ~1 second audio clips of spoken keywords
- **35 keyword classes**
- Train / Validation / Test split: **80% / 10% / 10%**
- Multiple speakers with varied accents and recording conditions

---

## Methodology

### Audio Preprocessing
- Resampling to **16 kHz** (CNN uses 8 kHz)
- Amplitude normalization
- Silence trimming
- Framing (20–25 ms windows)

### Feature Extraction (MFCC-based models)
- **12 MFCC coefficients**
- First-order (Δ) and second-order (ΔΔ) derivatives
- Final feature size: **36 × T**
- FFT → Mel filterbank → log energies → DCT

### Noise Augmentation
To evaluate robustness:
- Applied with **10% probability**
- Mixed background noise at **9:1 (clean : noise)** ratio
- MFCCs recomputed after augmentation

Formula:
```
y_aug[n] = 0.9 * clean[n] + 0.1 * noise[n]
```

---

## Model Architectures

### HMM-GMM (Baseline)
- 3 hidden states per keyword
- 2 Gaussian mixtures per state
- Trained using Baum–Welch + EM
- Uses MFCC + Δ + ΔΔ features

### CNN (Raw Waveform)
- 1D CNN inspired by **M5 architecture**
- Operates directly on raw audio (8 kHz)
- Global average pooling + fully connected classifier
- Optimized using Adam + NLL loss

### LSTM
- Unidirectional LSTM
- Hidden size: 128
- Captures long-term temporal dependencies in MFCC sequences

### BiLSTM + Attention
- Bidirectional LSTM encoder
- Dot-product attention mechanism
- Focuses on speech-rich temporal regions
- Improves localization of keywords within audio clips

### BiLSTM + DenseNet (Proposed Model)
- **2-layer BiLSTM front-end** for bidirectional temporal modeling
- **1D DenseNet feature extractor** for multi-scale feature reuse
- Adaptive average pooling for variable-length input
- Achieves strongest performance across all settings

---

## Experimental Results

| Model | Accuracy (Clean) | Accuracy (With Noise) |
|------|-----------------|-----------------------|
| HMM-GMM | ~65% | ~62% |
| CNN (Raw Audio) | ~85% | ~83–84% |
| LSTM | ~88–89% | ~87% |
| BiLSTM + Attention | ~91–92% | ~91% |
| **BiLSTM + DenseNet** | **92.56%** | **92.43%** |

### Key Observations
- Deep models outperform HMM-GMM by **25–30 percentage points**
- Noise causes **<1% degradation** in BiLSTM + DenseNet
- Attention and dense connectivity significantly improve robustness

---

## Analysis & Insights

- **Model capacity correlates strongly with accuracy**
- CNNs benefit from raw waveform learning but lack long-range context
- BiLSTMs capture richer temporal dependencies
- DenseNet layers enable efficient feature reuse and multi-scale learning
- Noise augmentation minimally affects deep architectures

---

## Evaluation Tools

- Accuracy metrics on held-out test set
- Confusion matrices for error analysis
- Clean vs noise-augmented comparisons

---

## Tech Stack

- **Python**
- **PyTorch**
- MFCC, FFT, Mel Filterbanks
- CNN, LSTM, BiLSTM, Attention
- DenseNet (1D)
- Google Speech Commands Dataset v2

---

## Applications

- Voice assistants (wake-word detection)
- Smart home & IoT control
- Automotive voice commands
- Low-latency speech interfaces

---

## Authors

**Dhanya Girdhar**  
**Raashi Sharma**   

---


## 📎 License

This project is intended for academic and research purposes.
