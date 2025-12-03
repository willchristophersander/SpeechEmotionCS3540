# Speech Emotion Recognition Project

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
  - 8 emotion recognition datasets: CREMA-D, EmoDB, RAVDESS, SAVEE, TESS, IEMOCAP, nEMO, MELD
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
```bash
# Train CRNN model (4-class)
cd demo
python train_unified.py

# Train CRNN model (6-class)
python train_unified_6class.py

# Train enhanced CRNN model
python train_unified_enhanced.py

# Legacy training scripts (feature-based models)
python scripts/training/Train_Dense_Ensemble.py
python scripts/training/Train_LSTM_Ensemble.py
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
- **4-Class Model**: 77.49% test accuracy, 77.26% validation accuracy
  - Emotions: Anger, Happy, Neutral, Sad
- **6-Class Model**: Training in progress
  - Emotions: Anger, Happy, Neutral, Sad, Fear, Surprise

**Model Architecture:**
- CRNN with 3 CNN blocks (32→64→128 channels)
- Bidirectional LSTM (2 layers, 128 hidden units)
- Attention mechanism for temporal focus
- Fully connected classification layers

See `MODEL_EVOLUTION_SUMMARY.md` for detailed evolution from feature extraction to deep learning approaches.

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
- Results and outputs are saved in the `results/` directory
- Visualizations are saved in the `visualizations/` directory
- Feature files are stored in the `features/` directory
