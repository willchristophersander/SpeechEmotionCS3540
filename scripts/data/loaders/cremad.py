"""
CREMA-D Dataset Loader
======================

Crowd-sourced Emotional Multimodal Actors Dataset
~7,400 samples from 91 actors (48 male, 43 female)
English, professional actors, high quality

File format: XXXX_XXX_EMO_XX.wav
Example: 1001_IEO_ANG_HI.wav

Emotions available:
    - ANG: Anger
    - HAP: Happy  
    - NEU: Neutral
    - SAD: Sad
    - FEA: Fear
    - DIS: Disgust (mapped to Anger in 4-class)
"""

MODULE_INFO = {
    'description': 'CREMA-D dataset loader - 7,400 samples, 6 emotions, English',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Disgust'],
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple, Optional
import csv

from .base import BaseDatasetLoader


class CREMADLoader(BaseDatasetLoader):
    """Loader for CREMA-D dataset."""
    
    DATASET_NAME = "CREMA-D"
    AVAILABLE_EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Disgust']
    
    # Mapping from filename codes to emotion names
    EMOTION_MAP = {
        'ANG': 'Anger',
        'HAP': 'Happy',
        'NEU': 'Neutral',
        'SAD': 'Sad',
        'FEA': 'Fear',
        'DIS': 'Disgust',
    }
    
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """
        Load all CREMA-D data with optional intensity-based weights.
        
        Args:
            return_weights: If True, return intensity-based weights from VoiceLevel ratings
        
        Returns:
            If return_weights=False: (file_paths, emotions)
            If return_weights=True: (file_paths, emotions, weights)
        """
        audio_dir = self.data_root / 'CREMA-D' / 'AudioWAV'
        
        if not audio_dir.exists():
            print(f"Warning: CREMA-D directory not found at {audio_dir}")
            if return_weights:
                return [], [], []
            return [], []
        
        # Load intensity scores if weights are requested
        intensity_map = {}
        if return_weights:
            summary_file = self.data_root / 'CREMA-D' / 'processedResults' / 'summaryTable.csv'
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            filename = row['FileName']
                            voice_level = row.get('VoiceLevel', '').strip()
                            if voice_level:
                                # Handle colon-separated values (take first/main value)
                                try:
                                    intensity = float(voice_level.split(':')[0])
                                    # Normalize intensity (0-100) to weight (0.5-2.0)
                                    # High intensity (70-100) -> 1.3-2.0
                                    # Medium (50-70) -> 1.0-1.3
                                    # Low (0-50) -> 0.5-1.0
                                    if intensity >= 70:
                                        weight = 1.0 + (intensity - 70) / 30.0 * 1.0  # 1.0 to 2.0
                                    elif intensity >= 50:
                                        weight = 0.7 + (intensity - 50) / 20.0 * 0.6  # 0.7 to 1.3
                                    else:
                                        weight = 0.5 + (intensity / 50.0) * 0.2  # 0.5 to 0.7
                                    intensity_map[filename] = weight
                                except (ValueError, IndexError):
                                    intensity_map[filename] = 1.0  # Default if parsing fails
                except Exception as e:
                    print(f"  Warning: Could not load CREMA-D intensity scores: {e}")
                    print(f"  Using uniform weights (1.0) for all samples")
        
        file_paths = []
        emotions = []
        weights = []
        
        for wav_file in audio_dir.glob('*.wav'):
            # Parse filename: 1001_IEO_ANG_HI.wav
            parts = wav_file.stem.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in self.EMOTION_MAP:
                    file_paths.append(str(wav_file))
                    emotions.append(self.EMOTION_MAP[emotion_code])
                    
                    # Get weight from intensity map if available
                    if return_weights:
                        # Match filename: 1001_IEO_ANG_HI.wav -> 1001_IEO_ANG_HI
                        filename_key = wav_file.stem
                        weight = intensity_map.get(filename_key, 1.0)
                        weights.append(weight)
        
        print(f"  CREMA-D: Found {len(file_paths)} samples")
        if return_weights and weights:
            avg_weight = sum(weights) / len(weights)
            print(f"    Intensity-based weights: avg={avg_weight:.2f}, range=[{min(weights):.2f}, {max(weights):.2f}]")
        
        if return_weights:
            return file_paths, emotions, weights
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False, return_weights: bool = False):
    """
    Convenience function to load CREMA-D data.
    
    Args:
        emotions: List of emotions to load (e.g., ['Anger', 'Happy', 'Neutral', 'Sad'])
        data_root: Path to DataSets directory (default: auto-detect)
        skip_missing: If True, skip unavailable emotions instead of raising error
        return_weights: If True, return intensity-based sample weights
    
    Returns:
        Tuple of (file_paths, labels, emotion_to_idx, weights)
        weights are intensity-based for CREMA-D, all 1.0 if return_weights=False
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = CREMADLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing, return_weights=return_weights)


# Allow running as script to test
if __name__ == '__main__':
    print("Testing CREMA-D Loader...")
    print(f"Available emotions: {CREMADLoader.AVAILABLE_EMOTIONS}")
    
    # Test loading 4-class
    try:
        paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
        print(f"\n4-class load: {len(paths)} samples")
        print(f"Label mapping: {mapping}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test loading unavailable emotion
    try:
        paths, labels, mapping = load(['Anger', 'Happy', 'Surprise'])  # Surprise not in CREMA-D
    except Exception as e:
        print(f"\nExpected error for Surprise: {type(e).__name__}")

