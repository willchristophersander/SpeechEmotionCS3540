# CS-FAIR-DEMO: Speech Emotion Recognition Demo

Real-time web demo for 4-class emotion recognition using CRNN (Convolutional Recurrent Neural Network).

## Overview

This is a standalone demo package for the Speech Emotion Recognition project. It includes:
- Web-based real-time emotion recognition interface
- Pre-trained 4-class CRNN model (Anger, Happy, Neutral, Sad)
- Complete inference pipeline (no training data required)

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo**:
   ```bash
   python app.py
   ```

3. **Open in browser**:
   - URL: http://localhost:5001
   - The demo will automatically load the pre-trained model

## Project Structure

```
CS-FAIR-DEMO/
├── app.py                      # Flask web application
├── templates/
│   └── index.html             # Web interface
├── checkpoints/
│   └── 4class/
│       └── crnn_emotion_model.pth  # Pre-trained model
├── ser/                       # Speech Emotion Recognition package
│   ├── models/                # Model architectures
│   ├── utils/                 # Audio preprocessing
│   ├── data/                  # Dataset classes (no data loaders)
│   ├── losses/                # Loss functions
│   └── augmentation/          # Data augmentation
├── 4class_model/              # 4-class model training code
│   ├── config_4class.py       # Hyperparameters
│   ├── trainer_4class.py     # Training logic
│   └── train_unified.py       # Training script
└── requirements.txt          # Python dependencies
```

## Model Information

- **Architecture**: CRNN (CNN + Bidirectional LSTM + Attention)
- **Emotions**: Anger, Happy, Neutral, Sad (4 classes)
- **Input**: Audio recordings (WAV format, auto-resampled to 22050 Hz)
- **Preprocessing**: Noise reduction, volume normalization, trimming
- **Performance**: ~77% validation accuracy

## Usage

### Web Interface

1. Click the microphone button or upload an audio file
2. Record or select audio (recommended: 2-5 seconds)
3. View the predicted emotion with confidence scores
4. See probability distribution across all 4 emotions

### API Endpoints

- `GET /` - Main demo page
- `POST /predict` - Predict emotion from audio (base64 encoded WAV)
- `GET /health` - Health check and model status

### Example API Usage

```python
import requests
import base64

# Load audio file
with open('audio.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')

# Predict emotion
response = requests.post('http://localhost:5001/predict', 
                        json={'audio': audio_b64})
result = response.json()

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
```

## Technical Details

- **Sample Rate**: 22050 Hz
- **Mel Bands**: 96
- **Spectrogram**: Log-mel spectrogram with normalization
- **Preprocessing**: 
  - Noise reduction (spectral gating)
  - RMS normalization (target: 0.1)
  - Automatic trimming of silence
- **Model Architecture**:
  - 3 CNN blocks (32→64→128 channels)
  - Bidirectional LSTM (2 layers, 128 hidden units)
  - Attention mechanism
  - Fully connected classification head

## Training Code

The `4class_model/` folder contains the complete training code:
- Configuration and hyperparameters
- Training loop with checkpointing
- Early stopping and learning rate scheduling

To retrain the model, you would need the training datasets (not included in this demo).

## Notes

- This demo package does **not** include training datasets (CREMA-D, RAVDESS, etc.)
- Data loaders are excluded as they reference dataset directories
- The model checkpoint is included and ready to use
- All inference code is self-contained

## Troubleshooting

- **Model not loading**: Check that `checkpoints/4class/crnn_emotion_model.pth` exists
- **Import errors**: Make sure you're running from the `CS-FAIR-DEMO/` directory
- **Audio issues**: Ensure audio is in WAV format and not corrupted
- **Port already in use**: Change the port in `app.py` (line 211)

## License

This demo is part of the Speech Emotion Recognition project for CS 3540.
