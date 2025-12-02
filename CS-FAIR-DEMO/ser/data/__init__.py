"""Data loading and dataset classes for speech emotion recognition."""

from .dataset import EmotionDataset

# Note: Data loaders (load_cremad, etc.) are not included in demo
# as they require dataset files that are not part of this demo package

__all__ = [
    'EmotionDataset',
]
