# CS-FAIR-DEMO Setup Guide

## What's Included

This demo package contains everything needed to run the Speech Emotion Recognition web application:

✅ **Web Application**
- `app.py` - Flask web server
- `templates/index.html` - User interface

✅ **Pre-trained Model**
- `checkpoints/4class/crnn_emotion_model.pth` - Best 4-class model (~77% accuracy)

✅ **Code Package**
- `ser/` - Complete SER package (models, preprocessing, utilities)
- `4class_model/` - Training code (for reference/documentation)

✅ **Documentation**
- `README.md` - Main documentation
- `requirements.txt` - Python dependencies

## What's NOT Included

❌ **Training Datasets**
- CREMA-D, RAVDESS, SAVEE, TESS, IEMOCAP, EmoDB, nEMO, MELD
- Data loaders are stubbed out (not functional without datasets)

❌ **Development Files**
- Logs, debug files, cache files (excluded via .gitignore)

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo**:
   ```bash
   python app.py
   ```

3. **Open browser**: http://localhost:5001

## File Structure

```
CS-FAIR-DEMO/
├── app.py                 # Main web application
├── templates/
│   └── index.html        # Web UI
├── checkpoints/
│   └── 4class/
│       └── crnn_emotion_model.pth  # Model (26MB)
├── ser/                  # SER package
│   ├── models/           # CRNN model definitions
│   ├── utils/            # Audio preprocessing
│   ├── data/             # Dataset classes (no loaders)
│   ├── losses/           # Loss functions
│   └── augmentation/     # Data augmentation
├── 4class_model/         # Training code
│   ├── config_4class.py  # Hyperparameters
│   ├── trainer_4class.py # Training logic
│   └── train_unified.py  # Training script
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── .gitignore           # Git ignore rules
```

## Verification

To verify everything is set up correctly:

1. Check model file exists:
   ```bash
   ls -lh checkpoints/4class/crnn_emotion_model.pth
   ```
   Should show ~26MB file

2. Test imports:
   ```bash
   python -c "from ser.models import CRNN; from ser.utils import AudioPreprocessor; print('✓ Imports work!')"
   ```

3. Test health endpoint (after starting app):
   ```bash
   curl http://localhost:5001/health
   ```

## Troubleshooting

- **Import errors**: Make sure you're in the CS-FAIR-DEMO directory
- **Model not found**: Verify `checkpoints/4class/crnn_emotion_model.pth` exists
- **Port in use**: Change port in `app.py` line 211

## Notes

- This is a **standalone demo** - no external data files needed
- The model is pre-trained and ready to use
- Training code is included for reference but requires datasets to run
- All inference code is self-contained

