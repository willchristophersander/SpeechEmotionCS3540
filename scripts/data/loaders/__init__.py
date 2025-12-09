"""
Dataset Loaders
===============

Modular data loading for speech emotion recognition datasets.

Each loader accepts a list of emotions and returns only data for those emotions.
If a requested emotion isn't available, an error is raised with helpful info.

Example usage:

    from scripts.data.loaders import load_datasets, AVAILABLE_LOADERS
    
    # Load specific emotions from specific datasets
    paths, labels, mapping = load_datasets(
        emotions=['Anger', 'Happy', 'Neutral', 'Sad'],
        datasets=['CREMA-D', 'RAVDESS', 'TESS'],
    )
    
    # See what emotions each dataset has
    for name, loader in AVAILABLE_LOADERS.items():
        print(f"{name}: {loader.AVAILABLE_EMOTIONS}")
"""

MODULE_INFO = {
    'description': 'Unified dataset loading system with emotion validation',
    'inputs': ['List of emotions', 'List of datasets'],
    'outputs': ['File paths, labels, and emotion mapping'],
    'status': 'production'
}

from .base import BaseDatasetLoader, EmotionNotAvailableError
from .cremad import CREMADLoader
from .ravdess import RAVDESSLoader
from .ravdess_songs import RAVDESSSongsLoader
from .savee import SAVEELoader
from .tess import TESSLoader
from .iemocap import IEMOCAPLoader
from .nemo import nEMOLoader
from .emodb import EmoDBLoader

# Registry of all available loaders
AVAILABLE_LOADERS = {
    'CREMA-D': CREMADLoader,
    'RAVDESS': RAVDESSLoader,
    'RAVDESS Songs': RAVDESSSongsLoader,
    'SAVEE': SAVEELoader,
    'TESS': TESSLoader,
    'IEMOCAP': IEMOCAPLoader,
    'nEMO': nEMOLoader,
    'EmoDB': EmoDBLoader,
}

__all__ = [
    'BaseDatasetLoader',
    'EmotionNotAvailableError',
    'CREMADLoader',
    'RAVDESSLoader',
    'RAVDESSSongsLoader',
    'SAVEELoader',
    'TESSLoader',
    'IEMOCAPLoader',
    'nEMOLoader',
    'EmoDBLoader',
    'AVAILABLE_LOADERS',
    'load_datasets',
    'get_dataset_info',
]


def get_dataset_info():
    """Print information about all available datasets."""
    print("=" * 70)
    print("Available Datasets for Speech Emotion Recognition")
    print("=" * 70)
    
    for name, loader_class in AVAILABLE_LOADERS.items():
        emotions = loader_class.AVAILABLE_EMOTIONS
        print(f"\n{name}:")
        print(f"  Emotions: {', '.join(emotions)}")
    
    print("\n" + "=" * 70)

