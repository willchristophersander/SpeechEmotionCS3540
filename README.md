# Speech Emotion Recognition Project

This project focuses on emotional analysis of vocal data using the CREMA-D dataset for machine learning classification.

## Dataset
- **CREMA-D**: Crowd Sourced Emotional Multimodal Actors Dataset
- Contains audio files with 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
- Multiple emotion levels: Low, Medium, High
- Audio formats: WAV, MP3

## Features
- Vocal feature extraction using librosa
- Machine learning pipeline for emotion classification
- Data preprocessing and visualization utilities

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
# Run the main feature extraction pipeline
python extract_features.py

# Prepare features and labels for ML training
python prepare_features_and_labels.py

# Clean dataset files (if needed)
python clean_dataset.py
```
