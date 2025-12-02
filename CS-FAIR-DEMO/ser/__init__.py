"""
Speech Emotion Recognition (SER) Package - Demo Version

Main package for speech emotion recognition using deep learning.
This is a demo version that excludes data loaders (no dataset files required).
"""

__version__ = "1.0.0-demo"

from .models import CRNN
from .data import EmotionDataset
from .losses import DistanceWeightedLoss, DistanceWeightedLoss6Class, EMOTION_DISTANCE_MATRIX, EMOTION_DISTANCE_MATRIX_6CLASS
from .augmentation import SpecAugment
from .utils import AudioPreprocessor

__all__ = [
    'CRNN',
    'EmotionDataset',
    'DistanceWeightedLoss', 'DistanceWeightedLoss6Class',
    'EMOTION_DISTANCE_MATRIX', 'EMOTION_DISTANCE_MATRIX_6CLASS',
    'SpecAugment',
    'AudioPreprocessor',
]
