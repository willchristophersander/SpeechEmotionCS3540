"""
SAVEE Dataset Loader
====================

Surrey Audio-Visual Expressed Emotion
~480 samples from 4 male British actors
British English

File format: DC_aXX.wav or similar
    Emotion prefix at start of filename:
    - a = anger
    - d = disgust
    - f = fear
    - h = happiness
    - n = neutral
    - sa = sadness
    - su = surprise
"""

MODULE_INFO = {
    'description': 'SAVEE dataset loader - 480 samples, 7 emotions, British English',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust'],
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple

from .base import BaseDatasetLoader


class SAVEELoader(BaseDatasetLoader):
    """Loader for SAVEE dataset."""
    
    DATASET_NAME = "SAVEE"
    AVAILABLE_EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust']
    
    # Mapping from filename prefixes to emotion names
    # Order matters - check longer prefixes first
    EMOTION_PREFIXES = [
        ('sa', 'Sad'),
        ('su', 'Surprise'),
        ('a', 'Anger'),
        ('d', 'Disgust'),
        ('f', 'Fear'),
        ('h', 'Happy'),
        ('n', 'Neutral'),
    ]
    
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """Load all SAVEE data."""
        # SAVEE is sometimes stored as "SURREY" directory
        audio_dir = self.data_root / 'SAVEE'
        if not audio_dir.exists():
            audio_dir = self.data_root / 'SURREY'
        
        # Check for Audio subdirectory
        if audio_dir.exists():
            audio_subdir = audio_dir / 'Audio'
            if audio_subdir.exists():
                audio_dir = audio_subdir
        
        if not audio_dir.exists():
            print(f"Warning: SAVEE/SURREY directory not found")
            if return_weights:
                return [], [], []
            return [], []
        
        file_paths = []
        emotions = []
        
        for wav_file in audio_dir.rglob('*.wav'):
            name = wav_file.stem.lower()
            
            # SAVEE format: DC_a01.wav or JK_sa15.wav
            # Extract emotion code (after underscore or at start)
            if '_' in name:
                emotion_code = name.split('_')[1]  # Part after underscore
            else:
                emotion_code = name
            
            # Find emotion from emotion code prefix
            for prefix, emotion in self.EMOTION_PREFIXES:
                # Check if emotion_code starts with prefix
                # For 'sa' vs 'su', check longer prefix first (already ordered)
                if emotion_code.startswith(prefix):
                    file_paths.append(str(wav_file))
                    emotions.append(emotion)
                    break
        
        print(f"  SAVEE: Found {len(file_paths)} samples")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load SAVEE data.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = SAVEELoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing SAVEE Loader...")
    print(f"Available emotions: {SAVEELoader.AVAILABLE_EMOTIONS}")
    
    paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
    print(f"\n4-class load: {len(paths)} samples")
    print(f"Label mapping: {mapping}")

