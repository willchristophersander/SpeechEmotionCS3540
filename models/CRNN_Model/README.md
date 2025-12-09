# Speech Emotion Recognition Project - CRNN Models

This is a **machine learning project** that focuses on emotional analysis of vocal data using deep learning techniques. The project implements Convolutional Recurrent Neural Networks (CRNN) with attention mechanisms to classify emotions from speech audio signals.

## Project Structure

```
├── config/                 # Configuration files
│   └── config.py          # Paths, settings, and constants
│
├── src/                    # Source code
│   ├── models/            # Model definitions
│   │   ├── CNN_Experimental.py
│   │   ├── Neural_Network_Experimental.py
│   │   ├── Ensemble_Classifier.py
│   │   └── ...
│   ├── preprocessing/     # Data preprocessing and feature extraction
│   │   ├── extract_features.py
│   │   ├── prepare_features_and_labels.py
│   │   └── ...
│   └── utils/            # Utility functions
│       └── paths.py      # Path helper functions
│
├── scripts/               # Executable scripts
│   ├── training/         # Model training scripts
│   │   ├── Train_*.py
│   │   ├── Optimize_*.py
│   │   └── ...
│   ├── evaluation/       # Model evaluation scripts
│   │   └── Evaluate_*.py
│   └── analysis/         # Data analysis scripts
│       ├── *_analysis.py
│       └── ...
│
├── features/              # Extracted features (CSV files)
├── models/                # Saved model files (.pth)
├── results/               # Results and outputs
├── visualizations/        # Generated plots and figures
└── requirements.txt       # Python dependencies

```

## Best CRNN Models

This folder contains the best-performing CRNN models:

### 4-Class Model (`crnn_multi_dataset.pth`)
- **Validation Accuracy:** 81.29%
- **Emotions:** Anger, Happy, Neutral, Sad
- **Architecture:** 3-layer CNN (32→64→128) + 2-layer BiLSTM (128) + Attention
- **Training Script:** `Train_CRNN_MultiDataset.py` (also available in `scripts/training/`)
- **Datasets:** CREMA-D, RAVDESS, SAVEE, TESS, IEMOCAP, nEMO

### 6-Class Model (`crnn_multi_dataset_6class.pth`)
- **Validation Accuracy:** 81.46%
- **Test Accuracy:** 81.0%
- **Emotions:** Anger, Happy, Neutral, Sad, Fear, Surprise
- **Architecture:** 3-layer CNN (32→64→128) + 2-layer BiLSTM (128) + Attention
- **Training Script:** `Train_CRNN_MultiDataset_6Class.py` (also available in `scripts/training/`)
- **Datasets:** CREMA-D, RAVDESS, RAVDESS Songs, SAVEE, TESS, IEMOCAP, nEMO, EmoDB

## Dataset

- **CREMA-D**: Crowd Sourced Emotional Multimodal Actors Dataset
- Contains audio files with 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
- Multiple emotion levels: Low, Medium, High
- Audio formats: WAV, MP3

## Features

- **Deep Learning Models**: 
  - CRNN (Convolutional Recurrent Neural Network) with attention mechanisms
  - End-to-end training from mel spectrograms
  - Support for 4-class and 6-class emotion classification
- **Comprehensive Dataset Support**: 
  - 8 emotion recognition datasets: CREMA-D, EmoDB, RAVDESS, RAVDESS Songs, SAVEE, TESS, IEMOCAP, nEMO
  - Unified preprocessing pipeline
- **Advanced ML Techniques**:
  - Mel spectrogram feature extraction
  - Data augmentation (SpecAugment, pitch shift, time stretch, noise injection)
  - Class-weighted loss functions for imbalanced data
  - Distance-weighted loss for emotion similarity modeling
- **Model Training & Evaluation**:
  - Checkpointing and resume training
  - Early stopping with validation monitoring
  - Per-class accuracy reporting
  - Comprehensive logging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure paths (optional):
   - Edit `config/config.py` to set your CREMA-D audio directory
   - Or set the `CREMA_D_AUDIO_DIR` environment variable

## Usage

### Feature Extraction
```bash
# Extract features from audio files
python src/preprocessing/extract_features.py

# Prepare features and labels for ML training
python src/preprocessing/prepare_features_and_labels.py

# Clean dataset files (if needed)
python src/preprocessing/clean_dataset.py
```

### Model Training

**Training scripts are available in this folder:**

```bash
# Train CRNN model (4-class) - from this folder
python models/CRNN_Model/Train_CRNN_MultiDataset.py

# Train CRNN model (6-class) - from this folder
python models/CRNN_Model/Train_CRNN_MultiDataset_6Class.py

# Or from the project root using the scripts folder
python scripts/training/Train_CRNN_MultiDataset.py
python scripts/training/Train_CRNN_MultiDataset_6Class.py

# Legacy training scripts (feature-based models)
python scripts/training/Train_Dense_Ensemble.py
python scripts/training/Train_LSTM_Ensemble.py
```

### Model Inference
```bash
# Use the trained models for inference
python scripts/interpret_model.py --model CRNN_Model/crnn_multi_dataset.pth --audio path/to/audio.wav
```

### Evaluation
```bash
# Evaluate ensemble models
python scripts/evaluation/Evaluate_4Way_Ensemble.py
```

### Analysis
```bash
# Run analysis scripts
python scripts/analysis/logistic_regression_analysis.py
python scripts/analysis/pitch_analysis.py
python scripts/analysis/pairwise_discriminatory_analysis.py
```

## Configuration

The project uses a centralized configuration file (`config/config.py`) for:
- Data paths (audio directory, feature files)
- Model paths (saved models directory)
- Hyperparameters (random seed, train/test split, etc.)
- Emotion and intensity mappings

To customize paths, edit `config/config.py` or set environment variables:
```bash
export CREMA_D_AUDIO_DIR="/path/to/your/audio/files"
```

## Model Performance

**Current Best Performance:**
- **4-Class Model**: 81.29% validation accuracy
  - Emotions: Anger, Happy, Neutral, Sad
- **6-Class Model**: 81.46% validation accuracy, 81.0% test accuracy
  - Emotions: Anger, Happy, Neutral, Sad, Fear, Surprise

**Model Architecture:**
- CRNN with 3 CNN blocks (32→64→128 channels)
- Bidirectional LSTM (2 layers, 128 hidden units)
- Attention mechanism for temporal focus
- Fully connected classification layers

See `PROJECT_SUMMARY_CRNN.md` for detailed architecture, evolution, and research citations.

## Model Interpretability

The project includes comprehensive interpretability tools:

```bash
# Visualize attention patterns for a single audio file
python scripts/interpret_model.py --model CRNN_Model/crnn_multi_dataset_6class.pth --audio path/to/audio.wav

# Visualize all 6 emotions
python scripts/visualize_all_emotions.py --model CRNN_Model/crnn_multi_dataset_6class.pth

# Generate all visualizations (attention, Grad-CAM, feature maps)
python scripts/interpret_model.py --model CRNN_Model/crnn_multi_dataset_6class.pth --audio path/to/audio.wav --all
```

See `scripts/INTERPRETABILITY_GUIDE.md` for detailed usage instructions.

## Dependencies

**Core ML/AI Libraries:**
- **PyTorch** - Deep learning framework for CRNN models
- **librosa** - Audio processing and mel spectrogram extraction
- **scikit-learn** - Traditional ML models and utilities

**Data Processing:**
- numpy, pandas - Data handling and manipulation
- scipy - Signal processing

**Visualization:**
- matplotlib, seaborn - Plotting and analysis

See `requirements.txt` for complete list with versions.

## Notes

- Model files (`.pth`) are saved in the `models/` directory
- Best models are also stored in `CRNN_Model/` directory
- **Training scripts** for both 4-class and 6-class models are included in this folder
- Results and outputs are saved in the `results/` directory
- Visualizations are saved in the `visualizations/` directory
- Feature files are stored in the `features/` directory

## References

For detailed information about:
- **Architecture and Evolution:** See `PROJECT_SUMMARY_CRNN.md`
- **Model Performance:** See `models/MODEL_PERFORMANCE.md`
- **Interpretability:** See `scripts/INTERPRETABILITY_GUIDE.md`
- **Data Loaders:** See `scripts/data/loaders/FOLDER_CONTENTS.md`
