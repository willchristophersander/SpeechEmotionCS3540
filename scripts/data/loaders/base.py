"""
Base Dataset Loader
===================

Defines the interface all dataset loaders must follow.

Each dataset loader must:
1. Define AVAILABLE_EMOTIONS - list of emotions this dataset contains
2. Implement load(emotions) - returns (file_paths, labels) for requested emotions
3. Raise EmotionNotAvailableError if a requested emotion isn't available
"""

MODULE_INFO = {
    'description': 'Base class and utilities for dataset loaders',
    'inputs': ['Emotion classes to load'],
    'outputs': ['File paths and labels'],
    'dependencies': [],
    'status': 'production'
}

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set
from pathlib import Path


class EmotionNotAvailableError(Exception):
    """Raised when a requested emotion is not available in a dataset."""
    
    def __init__(self, dataset_name: str, requested: List[str], available: List[str]):
        self.dataset_name = dataset_name
        self.requested = requested
        self.available = available
        
        missing = set(requested) - set(available)
        
        message = f"""
{'='*60}
ERROR: Emotion(s) not available in {dataset_name}
{'='*60}

Requested emotions: {', '.join(requested)}
Missing emotions:   {', '.join(missing)}

{dataset_name} only has these emotions:
  → {', '.join(available)}

Options:
  1. Remove {', '.join(missing)} from your requested emotions
  2. Use a different dataset that has {', '.join(missing)}
  3. Use the 'skip_missing=True' option to load available emotions only

{'='*60}
"""
        super().__init__(message)


class BaseDatasetLoader(ABC):
    """
    Base class for all dataset loaders.
    
    Subclasses must implement:
        - DATASET_NAME: str
        - AVAILABLE_EMOTIONS: List[str]
        - _load_all() -> Tuple[List[str], List[str]]  # returns (paths, emotion_names)
    """
    
    DATASET_NAME: str = "Unknown"
    AVAILABLE_EMOTIONS: List[str] = []
    
    def __init__(self, data_root: Path):
        """
        Initialize the loader.
        
        Args:
            data_root: Path to the DataSets directory
        """
        self.data_root = Path(data_root)
    
    @abstractmethod
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """
        Load all data from this dataset.
        
        Args:
            return_weights: If True, also return sample weights
        
        Returns:
            If return_weights=False: Tuple of (file_paths, emotion_names)
            If return_weights=True: Tuple of (file_paths, emotion_names, weights)
            emotion_names should be standardized: 'Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise'
            weights should be List[float] with same length as file_paths
        """
        pass
    
    def load(
        self, 
        emotions: List[str],
        skip_missing: bool = False,
        return_weights: bool = False
    ) -> Tuple[List[str], List[int], Dict[str, int], List[float]]:
        """
        Load data for specified emotions.
        
        Args:
            emotions: List of emotion names to load (e.g., ['Anger', 'Happy', 'Neutral', 'Sad'])
            skip_missing: If True, skip emotions not in dataset. If False, raise error.
            return_weights: If True, return sample weights (default: False for backward compatibility)
        
        Returns:
            Tuple of:
                - file_paths: List of audio file paths
                - labels: List of integer labels (indices into emotions list)
                - emotion_to_idx: Dict mapping emotion name to label index
                - weights: List of sample weights (all 1.0 by default, unless overridden by subclass)
        
        Raises:
            EmotionNotAvailableError: If requested emotion not available and skip_missing=False
        """
        # Normalize emotion names
        emotions = [self._normalize_emotion(e) for e in emotions]
        
        # Check availability
        available_set = set(self._normalize_emotion(e) for e in self.AVAILABLE_EMOTIONS)
        requested_set = set(emotions)
        missing = requested_set - available_set
        
        if missing and not skip_missing:
            raise EmotionNotAvailableError(
                self.DATASET_NAME,
                emotions,
                self.AVAILABLE_EMOTIONS
            )
        
        # Filter to available emotions if skip_missing
        if skip_missing:
            emotions = [e for e in emotions if e in available_set]
            if not emotions:
                print(f"Warning: No requested emotions available in {self.DATASET_NAME}")
                return [], [], {}, []
        
        # Create emotion to index mapping
        emotion_to_idx = {e: i for i, e in enumerate(emotions)}
        
        # Load all data (paths, emotions, and optionally weights)
        load_result = self._load_all(return_weights=return_weights)
        if return_weights and len(load_result) == 3:
            all_paths, all_emotions, all_weights = load_result
        else:
            all_paths, all_emotions = load_result[:2]
            all_weights = None
        
        # Filter to requested emotions
        file_paths = []
        labels = []
        weights = []
        
        for idx, (path, emotion) in enumerate(zip(all_paths, all_emotions)):
            norm_emotion = self._normalize_emotion(emotion)
            if norm_emotion in emotion_to_idx:
                file_paths.append(path)
                labels.append(emotion_to_idx[norm_emotion])
                # Always add weight (1.0 default if not provided by loader)
                if return_weights and all_weights is not None:
                    weights.append(all_weights[idx])
                else:
                    weights.append(1.0)  # Default weight
        
        return file_paths, labels, emotion_to_idx, weights
    
    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion name to standard format."""
        mapping = {
            # Anger variants
            'anger': 'Anger', 'angry': 'Anger', 'ang': 'Anger',
            # Happy variants
            'happy': 'Happy', 'happiness': 'Happy', 'hap': 'Happy', 'joy': 'Happy',
            # Neutral variants
            'neutral': 'Neutral', 'neu': 'Neutral', 'calm': 'Neutral',
            # Sad variants
            'sad': 'Sad', 'sadness': 'Sad',
            # Fear variants
            'fear': 'Fear', 'fearful': 'Fear', 'afraid': 'Fear', 'fea': 'Fear',
            # Surprise variants
            'surprise': 'Surprise', 'surprised': 'Surprise', 'sur': 'Surprise',
            # Disgust (maps to Anger in 4-class)
            'disgust': 'Disgust', 'dis': 'Disgust',
        }
        return mapping.get(emotion.lower(), emotion.title())
    
    def get_info(self) -> Dict:
        """Get information about this dataset."""
        return {
            'name': self.DATASET_NAME,
            'available_emotions': self.AVAILABLE_EMOTIONS,
            'path': str(self.data_root),
        }
    
    def __repr__(self):
        return f"{self.DATASET_NAME}Loader(emotions={self.AVAILABLE_EMOTIONS})"

