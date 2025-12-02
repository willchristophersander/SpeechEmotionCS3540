"""
4-Class Emotion Recognition Model
==================================

This module contains all components for training the 4-class CRNN emotion recognition model:
- config_4class: Hyperparameters and configuration
- trainer_4class: PyTorch training logic
- train_unified: Main training script
"""

from .config_4class import TrainingConfig, EMOTIONS, NUM_CLASSES, DEFAULT_CONFIG
from .trainer_4class import Trainer4Class

__all__ = [
    'TrainingConfig',
    'EMOTIONS',
    'NUM_CLASSES',
    'DEFAULT_CONFIG',
    'Trainer4Class'
]

