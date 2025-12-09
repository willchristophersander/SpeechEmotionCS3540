"""
Data Loading Package
====================

Modular data loading for speech emotion recognition.

Usage:
    from scripts.data import load_datasets, load_4class, load_6class
    
    # Load 4-class emotions
    paths, labels, mapping, datasets = load_4class(
        datasets=['CREMA-D', 'RAVDESS', 'TESS']
    )
    
    # Load specific emotions from specific datasets
    paths, labels, mapping, datasets = load_datasets(
        emotions=['Anger', 'Happy', 'Neutral', 'Sad', 'Fear'],
        datasets=['CREMA-D', 'RAVDESS']
    )
"""

MODULE_INFO = {
    'description': 'Data loading package for speech emotion recognition',
    'status': 'production'
}

from .loaders.unified import (
    load_datasets,
    load_4class,
    load_6class,
    check_emotion_availability,
)
from .loaders import (
    AVAILABLE_LOADERS,
    EmotionNotAvailableError,
    get_dataset_info,
)

__all__ = [
    'load_datasets',
    'load_4class', 
    'load_6class',
    'check_emotion_availability',
    'AVAILABLE_LOADERS',
    'EmotionNotAvailableError',
    'get_dataset_info',
]





