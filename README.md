# SpeechEmotionCS3540

**William Sander and Seth Shienbrood CS3540 Project**

A comprehensive speech emotion recognition research project using deep learning techniques, focusing on Convolutional Recurrent Neural Networks (CRNN) with attention mechanisms.

## Overview

This project explores various approaches to speech emotion recognition, from feature-based machine learning to CRNN spectogram architectures. 
## Key Components

###  Best Models
- **CRNN 4-Class Model**: 81.29% validation accuracy (Anger, Happy, Neutral, Sad)
- **CRNN 6-Class Model**: 81.46% validation accuracy (Anger, Happy, Neutral, Sad, Fear, Surprise)

Best models are stored in the [`models/CRNN_Model/`](models/CRNN_Model/) directory. See [`models/CRNN_Model/README.md`](models/CRNN_Model/README.md) for detailed information.

###  Datasets
The project supports 8 emotion recognition datasets:
- **CREMA-D** - ~7,400 samples (English, professional actors)
- **RAVDESS** - ~1,400 samples (English, high quality speech)
- **RAVDESS Songs** - ~1,700 samples (English, emotional songs)
- **SAVEE** - ~480 samples (British English)
- **TESS** - ~2,400 samples (English, older females)
- **IEMOCAP** - ~4,900 samples (English, conversational)
- **nEMO** - ~4,500 samples (Polish, acted)
- **EmoDB** - ~535 samples (German, acted)

###  Architecture
- **3-layer CNN** (32→64→128 channels) for feature extraction
- **2-layer Bidirectional LSTM** (128 hidden units) for temporal modeling
- **Attention mechanism** for focusing on important time regions
- **Fully connected layers** for classification

###  Research & Development
- Model interpretability tools (attention visualization, Grad-CAM)
- Comprehensive data loading system with modular dataset loaders
- Performance tracking and model registry
- Self-documenting codebase with auto-generated documentation

## Documentation

- **[models/CRNN_Model/README.md](models/CRNN_Model/README.md)** - Best models and usage
- **[PROJECT_SUMMARY_CRNN.md](PROJECT_SUMMARY_CRNN.md)** - Architecture, evolution, and research
- **[models/MODEL_PERFORMANCE.md](models/MODEL_PERFORMANCE.md)** - Performance leaderboard
- **[scripts/INTERPRETABILITY_GUIDE.md](scripts/INTERPRETABILITY_GUIDE.md)** - Model visualization guide
- **[scripts/data/loaders/FOLDER_CONTENTS.md](scripts/data/loaders/FOLDER_CONTENTS.md)** - Dataset loader documentation

## Research Contributions

This project implements and extends research on:
- CRNN architectures for speech emotion recognition
- Attention mechanisms for temporal modeling
- Multi-dataset training and generalization
- Model interpretability for emotion recognition

See `PROJECT_SUMMARY_CRNN.md` for detailed citations and references.

## License

[Add your license here]

## Contact

[Add contact information here]
