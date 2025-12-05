# SpeechEmotionCS3540

**William Sander and Seth Shienbrood CS3540 Project**

A comprehensive speech emotion recognition research project using multiple deep learning architectures, techniques, data refinement methods and data ingestion methods.

## Overview

This project explores various approaches to speech emotion recognition, from feature-based machine learning to CRNN spectogram architectures. 
## Key Components

###  Models
- **CRNN 4-Class Model**: 81.29% validation accuracy (Anger, Happy, Neutral, Sad)
- **CRNN 6-Class Model**: 81.46% validation accuracy (Anger, Happy, Neutral, Sad, Fear, Surprise)

Best models are stored in the models directory. 

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

###  Research & Development
- Model interpretability tools (attention visualization, Grad-CAM)
- Comprehensive data loading system with modular dataset loaders
- Performance tracking and model registry
- Self-documenting codebase with auto-generated documentation

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
