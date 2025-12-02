# 4-Class Emotion Recognition Model

This folder contains all components for training the 4-class CRNN emotion recognition model.

## Files

- **`config_4class.py`**: Hyperparameters and training configuration
  - Model settings (n_mels, dropout, num_classes)
  - Optimizer settings (learning_rate, weight_decay)
  - Scheduler settings
  - Training parameters (max_epochs, early_stop_patience, etc.)
  - Loss function settings (class_weights, label_smoothing)

- **`trainer_4class.py`**: PyTorch training logic
  - `Trainer4Class` class with complete training loop
  - Training, validation, checkpoint management
  - Early stopping and learning rate scheduling

- **`train_unified.py`**: Main training script
  - Data loading and preparation
  - Model initialization
  - Orchestrates the training process

## Usage

To train the 4-class model:

```bash
cd demo/scripts/training/4class_model
python train_unified.py
```

Or from the training directory:

```bash
python -m 4class_model.train_unified
```

## Emotion Classes

The 4-class model recognizes:
- **Anger** (class 0)
- **Happy** (class 1)
- **Neutral** (class 2)
- **Sad** (class 3)

## Model Output

Trained models are saved to:
- Best model: `demo/checkpoints/4class/crnn_emotion_model.pth`
- Periodic checkpoints: `demo/checkpoints/4class/checkpoint_epoch_*.pth`

