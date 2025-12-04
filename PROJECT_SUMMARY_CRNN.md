# Speech Emotion Recognition: Project Summary

**Project:** CS 3540 Speech Emotion Recognition  
**Date:** December 2024  
**Current Best Model:** CRNN + Bidirectional LSTM with Attention  
**Best Validation Accuracy:** 81.29% (4-class), 76.76% (6-class)

---

## Table of Contents

1. [Current Model Architecture](#current-model-architecture)
2. [Project Evolution](#project-evolution)
3. [Datasets](#datasets)
4. [Research Citations](#research-citations)
5. [Technical Specifications](#technical-specifications)

---

## Current Model Architecture

### Overview

Our best-performing model is a **Convolutional Recurrent Neural Network (CRNN)** with bidirectional LSTM and attention mechanisms. This architecture achieves **81.29% validation accuracy** on 4-class emotion recognition (Anger, Happy, Neutral, Sad) and **76.76%** on 6-class recognition (adding Fear and Surprise).

### Architecture Details

```
Input: Mel Spectrogram [batch, 1, 96, 173]
    - 96 mel frequency bands
    - 173 time frames (4 seconds at 22050 Hz, hop_length=512)
    ↓
CNN Feature Extraction (3 layers):
    Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    Output: [batch, 128, 12, 21]  (12 = 96/8, 21 = 173/8)
    ↓
Reshape for LSTM:
    Permute and flatten: [batch, 21, 128×12] = [batch, 21, 1536]
    ↓
Bidirectional LSTM (2 layers):
    LSTM(1536 → 128, num_layers=2, bidirectional=True)
    Output: [batch, 21, 256]  (256 = 128×2 for bidirectional)
    ↓
Attention Mechanism:
    Linear(256 → 64) → Tanh → Linear(64 → 1) → Softmax
    Weighted sum: [batch, 256]
    ↓
Fully Connected Classifier:
    Linear(256 → 64) → ReLU → Dropout(0.3)
    Linear(64 → num_classes) → Logits
```

### Model Parameters

- **Total Parameters:** ~2.5M
- **CNN Channels:** 32 → 64 → 128
- **LSTM Hidden Size:** 128 (256 with bidirectional)
- **LSTM Layers:** 2
- **Attention Dimension:** 64
- **Dropout:** 0.3

### Key Components

1. **CNN Layers**: Extract local spectral-temporal patterns from mel spectrograms
2. **Bidirectional LSTM**: Models temporal dynamics in both forward and backward directions
3. **Attention Mechanism**: Focuses on emotion-relevant time segments
4. **Fully Connected Layers**: Final classification with dropout regularization

### Audio Preprocessing

```python
1. Load audio at 22050 Hz, mono
2. Noise reduction (spectral gating)
3. Trim silence (top_db=20)
4. RMS normalization (target_rms=0.1)
5. Extract mel spectrogram:
   - 96 mel bands
   - 512 hop length
   - 2048 FFT window
   - Max duration: 4 seconds
6. Normalize to [-1, 1]
7. Pad/truncate to fixed length (173 frames)
```

### Data Augmentation

**Audio-level augmentation** (during training):
- Pitch shift: ±4 semitones
- Time stretch: 0.85× to 1.15×
- Add noise: SNR 15-30 dB
- Volume variation: 0.5× to 1.5×
- Low-pass filter: 4-8 kHz cutoff

**Spectrogram augmentation** (SpecAugment):
- Frequency masking: up to 10 mel bands
- Time masking: up to 20 frames

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| **Batch Size** | 40 |
| **Learning Rate** | 0.001 |
| **LR Schedule** | Cosine Annealing with Warm Restarts (T_0=10, T_mult=2) |
| **Optimizer** | Adam (weight_decay=0.01) |
| **Max Epochs** | 150 |
| **Early Stopping** | Patience=28 epochs |
| **Warmup Epochs** | 5 |
| **Gradient Clipping** | Max norm=1.0 |
| **Dropout** | 0.3 |
| **Label Smoothing** | 0.1 |

### Loss Function

**Distance-Weighted Cross-Entropy Loss** with class weights:

```python
# Emotion relationship matrix (arousal-valence space)
EMOTION_DISTANCE_MATRIX = [
    [1.0, 0.9, 0.7, 0.6],  # Anger
    [0.9, 1.0, 0.6, 0.8],  # Happy
    [0.7, 0.6, 1.0, 0.8],  # Neutral
    [0.6, 0.8, 0.8, 1.0],  # Sad
]

# Class weights: [Anger, Happy, Neutral, Sad]
class_weights = [0.7, 1.3, 1.0, 1.0]
```

The distance-weighted loss reduces penalty for predicting similar emotions (e.g., predicting Sad when true is Anger gets 0.6× penalty instead of full penalty).

---

## Project Evolution

### Phase 1: Feature Extraction with Traditional ML (Initial Approach)

**Timeline:** Early project phase  
**Accuracy:** 55-67% (baseline), 73% (with relative features), 91% (4-class with speaker normalization)

**Approach:**
- Hand-crafted acoustic features (253 features total)
- Categories: MFCCs, spectral features, prosodic (F0), voice quality, formants, energy, rhythm
- Dense neural network classifier (3 layers, ~500K parameters)

**Strengths:**
- Interpretable features
- Fast inference (once features extracted)
- Low memory footprint

**Limitations:**
- Feature extraction bottleneck (~100-200ms per sample)
- Limited to known acoustic patterns
- Speaker dependency for best results

### Phase 2: Transition to Deep Learning (CRNN)

**Timeline:** Mid-project  
**Motivation:**
- Real-time performance requirements
- Better generalization potential
- End-to-end learning without manual feature engineering

**Initial CRNN:**
- 3-layer CNN (32→64→128 channels)
- 2-layer bidirectional LSTM (128 hidden)
- Attention mechanism
- 80 mel bands → later increased to 96

**Performance:** 75.92% validation accuracy (4-class)

### Phase 3: Multi-Dataset Training

**Timeline:** Mid-to-late project  
**Evolution:**
- Started with single dataset (RAVDESS)
- Expanded to multiple English datasets (CREMA-D, RAVDESS, SAVEE, TESS)
- Added conversational dataset (IEMOCAP)
- **Total samples:** ~21,000+ across 6 datasets

**Impact:**
- Improved generalization
- Better class balance
- More diverse speaker characteristics

### Phase 4: Spectrogram-Based Input

**Timeline:** Mid-project  
**Change:** Switched from feature extraction to raw spectrogram input

**Benefits:**
- Faster preprocessing (5-10ms vs 100-200ms)
- End-to-end learning
- Model learns optimal representations
- Better real-time performance

### Phase 5: Multilingual Expansion

**Timeline:** Late project  
**Added Languages:**
- **Polish:** nEMO dataset (~4,500 samples)
- **German:** EmoDB dataset (~535 samples)

**Rationale:**
- Research shows universal acoustic patterns in emotion expression
- Cross-linguistic training forces model to learn emotion patterns, not language-specific features
- Better generalization to unseen speakers and languages

**Total Languages:** 3 (English, Polish, German)  
**Total Datasets:** 8 (including RAVDESS Songs)

### Phase 6: Singing Data Integration

**Timeline:** Recent  
**Addition:** RAVDESS Emotional Song Audio (~2,000 samples)

**Motivation:**
- Address real-world use case (singing emotion recognition)
- Improve model robustness to different vocal modalities
- Address observed issue: happy songs misclassified as angry/neutral

**Impact:**
- Model now handles both speech and singing
- Better generalization to musical contexts
- Total samples: ~25,800+ across 8 datasets

### Phase 7: Intensity-Based Weighting (Current)

**Timeline:** Latest  
**Feature:** CREMA-D intensity scores for sample weighting

**Implementation:**
- CREMA-D provides VoiceLevel intensity ratings (0-100)
- High intensity samples (70-100) → weight 1.0-2.0
- Medium intensity (50-70) → weight 0.7-1.3
- Low intensity (0-50) → weight 0.5-0.7
- Other datasets: uniform weights (1.0)

**Status:** Implemented in loaders, ready for training script integration

---

## Datasets

### Dataset Summary

| Dataset | Language | Samples | Emotions | Type | Citation |
|---------|----------|---------|----------|------|----------|
| **CREMA-D** | English (US) | 7,442 | 6 | Acted | [Cao et al., 2014](#crema-d) |
| **RAVDESS** | English (US) | 2,880 | 8 | Acted | [Livingstone & Russo, 2018](#ravdess) |
| **RAVDESS Songs** | English (US) | 2,024 | 8 | Singing | [Livingstone & Russo, 2018](#ravdess) |
| **SAVEE** | English (UK) | 480 | 7 | Acted | [Jackson & Haq, 2014](#savee) |
| **TESS** | English (CA) | 5,600 | 7 | Acted | [Dupuis & Pichora-Fuller, 2010](#tess) |
| **IEMOCAP** | English (US) | 4,901 | 6 | Conversational | [Busso et al., 2008](#iemocap) |
| **nEMO** | Polish | 4,481 | 6 | Acted | [Christop, 2024](#nemo) |
| **EmoDB** | German | 535 | 5 | Acted | [Burkhardt et al., 2005](#emodb) |

**Total:** ~25,828 samples across 3 languages (English, Polish, German)

### Dataset Details

#### CREMA-D
- **Full Name:** Crowd-sourced Emotional Multimodal Actors Dataset
- **Actors:** 91 (48 male, 43 female), ages 20-74
- **Emotions:** Anger, Disgust, Fear, Happy, Neutral, Sad
- **Special Features:** Intensity ratings (0-100) from crowd-sourced annotations
- **Modalities:** Audio, video, audio-visual
- **Quality:** High quality, professional actors

#### RAVDESS
- **Full Name:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Actors:** 24 (12 male, 12 female), professional actors
- **Emotions:** Anger, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Special Features:** Includes both speech and song versions
- **Quality:** Very high quality, standardized recording conditions

#### SAVEE
- **Full Name:** Surrey Audio-Visual Expressed Emotion
- **Actors:** 4 male speakers
- **Emotions:** Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Accent:** British English
- **Quality:** High quality, controlled environment

#### TESS
- **Full Name:** Toronto Emotional Speech Set
- **Actors:** 2 female speakers (older adults)
- **Emotions:** Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Accent:** Canadian English
- **Quality:** High quality, clear recordings

#### IEMOCAP
- **Full Name:** Interactive Emotional Dyadic Motion Capture
- **Actors:** 10 speakers (5 male, 5 female) in dyadic conversations
- **Emotions:** Anger, Happy, Neutral, Sad, Fear, Surprise (and others)
- **Special Features:** Conversational, naturalistic interactions
- **Quality:** High quality, multimodal (audio + video + motion capture)

#### nEMO
- **Full Name:** Dataset of Emotional Speech in Polish
- **Actors:** 9 actors
- **Emotions:** Anger, Fear, Happy, Sad, Surprise, Neutral
- **Language:** Polish
- **Special Features:** Includes transcriptions for ASR tasks
- **Quality:** High quality, carefully selected phonetic material

#### EmoDB
- **Full Name:** Berlin Database of Emotional Speech
- **Actors:** 10 speakers (5 male, 5 female)
- **Emotions:** Anger, Happy, Neutral, Sad, Fear
- **Language:** German
- **Special Features:** Does not include Surprise
- **Quality:** High quality, acted emotions

---

## Research Citations

### Architecture & Methodology

#### CRNN for Speech Emotion Recognition

The CRNN (Convolutional Recurrent Neural Network) architecture combines CNN layers for spatial feature extraction with LSTM layers for temporal modeling, which has been shown effective for sequential data classification tasks.

**Related Work:**
- **Shi, B., et al. (2016).** "Very Deep Convolutional Networks for End-to-End Speech Recognition." *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 25(12), 2260-2273.
  - Demonstrates effectiveness of deep CNNs for speech tasks

- **Graves, A., & Schmidhuber, J. (2005).** "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks*, 18(5-6), 602-610.
  - Introduces bidirectional LSTM for sequence modeling

- **Bahdanau, D., Cho, K., & Bengio, Y. (2014).** "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv preprint arXiv:1409.0473*.
  - Attention mechanism for sequence-to-sequence tasks

#### Attention Mechanisms

- **Vaswani, A., et al. (2017).** "Attention is All You Need." *Advances in Neural Information Processing Systems*, 30.
  - Transformer architecture with self-attention (foundational work)

- **Luong, M. T., Pham, H., & Manning, C. D. (2015).** "Effective Approaches to Attention-based Neural Machine Translation." *Proceedings of EMNLP*.
  - Attention mechanisms for neural networks

#### Mel Spectrograms

- **Davis, S., & Mermelstein, P. (1980).** "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357-366.
  - Mel-frequency cepstral coefficients (MFCCs) and mel scale

#### SpecAugment

- **Park, D. S., et al. (2019).** "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *Interspeech 2019*.
  - Spectrogram augmentation for speech recognition

### Speech Emotion Recognition Surveys

- **El Ayadi, M., Kamel, M. S., & Karray, F. (2011).** "Survey on speech emotion recognition: Features, classification schemes, and databases." *Pattern Recognition*, 44(3), 572-587.
  - Comprehensive survey on SER methods and datasets
  - Discusses universal acoustic correlates of emotion

- **Schuller, B., et al. (2013).** "Cross-corpus acoustic emotion recognition: Variances and strategies." *IEEE Transactions on Affective Computing*, 1(2), 119-131.
  - Cross-corpus emotion recognition challenges

### Multilingual & Cross-Linguistic Emotion Recognition

- **Pell, M. D., Monetta, L., Paulmann, S., & Kotz, S. A. (2009).** "Recognizing emotions in a foreign language." *Journal of Nonverbal Behavior*, 33(2), 107-120.
  - Monolingual Spanish speakers recognizing emotions in English, German, Arabic
  - Available: https://link.springer.com/article/10.1007/s10919-008-0065-7

- **Laukka, P., Elfenbein, H. A., Chui, W., et al. (2018).** "Cross-cultural decoding of positive and negative non-linguistic emotion vocalizations." *Frontiers in Psychology*, 9, 1158.
  - English speakers recognizing emotions in Spanish, Chinese, Arabic
  - Available: https://pubmed.ncbi.nlm.nih.gov/29904120/

- **Scherer, K. R., Banse, R., & Wallbott, H. G. (2001).** "Emotion inferences from vocal expression correlate across languages and cultures." *Journal of Cross-Cultural Psychology*, 32(1), 76-92.
  - Cross-cultural emotion recognition studies

- **Banse, R., & Scherer, K. R. (1996).** "Acoustic profiles in vocal emotion expression." *Journal of Personality and Social Psychology*, 70(3), 614-636.
  - Universal acoustic profiles in emotion expression

### Loss Functions & Training

- **Lin, T. Y., et al. (2017).** "Focal Loss for Dense Object Detection." *Proceedings of ICCV*.
  - Focal loss for addressing class imbalance (referenced in improved model experiments)

- **Loshchilov, I., & Hutter, F. (2016).** "SGDR: Stochastic Gradient Descent with Warm Restarts." *arXiv preprint arXiv:1608.03983*.
  - Cosine annealing with warm restarts learning rate schedule

---

## Dataset Citations

### CREMA-D {#crema-d}

**Citation:**
```
Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). 
CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. 
IEEE Transactions on Affective Computing, 5(4), 377-390.
DOI: 10.1109/TAFFC.2014.2336244
```

**Paper:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/

**Additional Reference:**
```
Keutmann, M. K., Moore, S. L., Savitt, A., & Gur, R. C. (2015). 
Generating an item pool for translational social cognition research: methodology and initial validation. 
Behavior Research Methods, 47(1), 228-234.
```

### RAVDESS {#ravdess}

**Citation:**
```
Livingstone, S. R., & Russo, F. A. (2018). 
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): 
A dynamic, multimodal set of facial and vocal expressions in North American English. 
PLoS ONE, 13(5), e0196391.
DOI: 10.1371/journal.pone.0196391
```

**Paper:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391

**Dataset:** https://zenodo.org/record/1188976

### SAVEE {#savee}

**Citation:**
```
Jackson, P., & Haq, S. (2014). 
Surrey Audio-Visual Expressed Emotion (SAVEE) Database. 
University of Surrey, Guildford, UK.
```

**Dataset:** http://kahlan.eps.surrey.ac.uk/savee/

### TESS {#tess}

**Citation:**
```
Dupuis, K., & Pichora-Fuller, M. K. (2010). 
Toronto Emotional Speech Set (TESS). 
University of Toronto, Psychology Department.
```

**Dataset:** https://tspace.library.utoronto.ca/handle/1807/24487

### IEMOCAP {#iemocap}

**Citation:**
```
Busso, C., Bulut, M., Lee, C. C., Kazemzadeh, A., Mower, E., Kim, S., 
Chang, J. N., Lee, S., & Narayanan, S. S. (2008). 
IEMOCAP: Interactive emotional dyadic motion capture database. 
Language Resources and Evaluation, 42(4), 335-359.
DOI: 10.1007/s10579-008-9076-6
```

**Paper:** https://link.springer.com/article/10.1007/s10579-008-9076-6

**Dataset:** https://sail.usc.edu/iemocap/

### nEMO {#nemo}

**Citation:**
```
@inproceedings{christop-2024-nemo-dataset,
    title = "n{EMO}: Dataset of Emotional Speech in {P}olish",
    author = "Christop, Iwona",
    booktitle = "Proceedings of the 2024 Joint International Conference on 
                 Computational Linguistics, Language Resources and Evaluation 
                 (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1059",
    pages = "12111--12116",
}
```

**Paper:** https://aclanthology.org/2024.lrec-main.1059

**Dataset:** https://github.com/iwona-christop/nEMO

### EmoDB {#emodb}

**Citation:**
```
Burkhardt, F., Paeschke, A., Rolfes, M., Sendlmeier, W. F., & Weiss, B. (2005). 
A database of German emotional speech. 
Proceedings of Interspeech 2005, 1517-1520.
```

**Paper:** https://www.isca-speech.org/archive/interspeech_2005/

**Dataset:** http://www.emodb.bilderbar.info/

---

## Technical Specifications

### Current Best Model

**File:** `scripts/training/Train_CRNN_MultiDataset.py`  
**Checkpoint:** `models/crnn_multi/crnn_multi_dataset.pth`  
**Best Accuracy:** 81.29% validation (4-class), 76.76% (6-class)

### Model Checkpoints

- **4-class model:** `models/crnn_multi/crnn_multi_dataset.pth` (81.29%)
- **6-class model:** `models/crnn_multi_6class/crnn_multi_dataset_6class.pth` (76.76%)
- **Legacy model:** `demo/checkpoints/4class/crnn_emotion_model.pth` (77.26%)

### Dependencies

```
torch >= 2.0
librosa >= 0.10
numpy >= 1.21
scikit-learn >= 1.0
scipy >= 1.7
tqdm >= 4.60
noisereduce >= 2.0
```

### Hardware Requirements

- **Training:** GPU recommended (CUDA or MPS)
- **Inference:** CPU or GPU (both fast, ~10-20ms per sample)
- **Memory:** ~4GB RAM for training, <1GB for inference

### Performance Metrics

**4-Class Model (Anger, Happy, Neutral, Sad):**
- Validation Accuracy: 81.29%
- Training Time: ~45-60 minutes (150 epochs)
- Inference: ~10-20ms per sample

**6-Class Model (+ Fear, Surprise):**
- Validation Accuracy: 76.76%
- Training Time: ~60-75 minutes (150 epochs)
- Inference: ~10-20ms per sample

### Data Statistics

**Total Training Samples:** ~25,828
- CREMA-D: 7,442
- RAVDESS: 2,880 (speech)
- RAVDESS Songs: 2,024
- SAVEE: 480
- TESS: 5,600
- IEMOCAP: 4,901
- nEMO: 4,481
- EmoDB: 535

**Languages:** 3 (English, Polish, German)  
**Modalities:** Speech + Singing  
**Split:** 60% train / 20% validation / 20% test

---

## Key Achievements

1. **Architecture Evolution:** From feature extraction (55-67%) to CRNN (81.29%)
2. **Dataset Expansion:** From single dataset to 8 datasets across 3 languages
3. **Modality Expansion:** Added singing data for better real-world generalization
4. **Multilingual Training:** Cross-linguistic training for universal emotion patterns
5. **Intensity Weighting:** Implemented CREMA-D intensity-based sample weighting
6. **Real-time Performance:** 5-10× faster than feature extraction approach

---

## Future Directions

1. **Architecture Improvements:**
   - Deeper CNNs with residual connections
   - Transformer-based attention
   - Multi-head attention mechanisms

2. **Training Enhancements:**
   - Focal loss for better class imbalance handling
   - Mixup augmentation
   - Contrastive learning for embeddings

3. **Data Expansion:**
   - Additional languages (Italian, Spanish, etc.)
   - More singing datasets
   - Real-world conversational data

4. **Deployment Optimization:**
   - Model quantization (INT8)
   - Model pruning
   - Edge device optimization

---

**Last Updated:** December 2024  
**Project Status:** Active development  
**Best Model:** CRNN Multi-Dataset (81.29% validation accuracy)

