"""
RAVDESS Dataset Loader
======================

Ryerson Audio-Visual Database of Emotional Speech and Song
~1,440 samples from 24 actors (12 male, 12 female)
English, professional actors, very high quality

File format: XX-XX-XX-XX-XX-XX-XX.wav
    - Position 3 (index 2) is emotion code

Emotion codes:
    01 = neutral
    02 = calm (mapped to Neutral)
    03 = happy
    04 = sad
    05 = angry
    06 = fearful
    07 = disgust (mapped to Anger in 4-class)
    08 = surprised
"""

MODULE_INFO = {
    'description': 'RAVDESS speech dataset loader - ~2,880 speech samples, 8 emotions, English',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust'],
    'notes': 'RAVDESS speech only. For songs, use RAVDESSSongsLoader',
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple, Dict

from .base import BaseDatasetLoader


class RAVDESSLoader(BaseDatasetLoader):
    """Loader for RAVDESS dataset."""
    
    DATASET_NAME = "RAVDESS"
    AVAILABLE_EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust']
    
    # Mapping from emotion codes to emotion names
    EMOTION_MAP = {
        '01': 'Neutral',
        '02': 'Neutral',  # Calm → Neutral
        '03': 'Happy',
        '04': 'Sad',
        '05': 'Anger',
        '06': 'Fear',
        '07': 'Disgust',
        '08': 'Surprise',
    }
    
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """Load all RAVDESS speech data (speech only, no songs)."""
        audio_dir = self.data_root / 'RAVDESS'
        
        if not audio_dir.exists():
            print(f"Warning: RAVDESS directory not found at {audio_dir}")
            if return_weights:
                return [], [], []
            return [], []
        
        file_paths = []
        emotions = []
        
        # RAVDESS has nested actor folders
        # Filename format: XX-XX-XX-XX-XX-XX-XX.wav
        # Position 1 (index 1): vocal_channel (01=speech, 02=song)
        # Position 2 (index 2): emotion_code
        
        # Only load speech files (vocal_channel='01')
        for wav_file in audio_dir.rglob('*.wav'):
            # Parse filename: 03-01-05-01-01-01-12.wav
            parts = wav_file.stem.split('-')
            if len(parts) >= 3:
                vocal_channel = parts[1]  # 01=speech, 02=song
                emotion_code = parts[2]
                
                # Only speech (vocal_channel='01')
                if vocal_channel == '01' and emotion_code in self.EMOTION_MAP:
                    file_paths.append(str(wav_file))
                    emotions.append(self.EMOTION_MAP[emotion_code])
        
        print(f"  RAVDESS: Found {len(file_paths)} samples (speech)")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load RAVDESS speech data.
    
    Args:
        emotions: List of emotions to load
        data_root: Path to DataSets directory
        skip_missing: If True, skip unavailable emotions
    
    Returns:
        Tuple of (file_paths, labels, emotion_to_idx)
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = RAVDESSLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing RAVDESS Loader...")
    print(f"Available emotions: {RAVDESSLoader.AVAILABLE_EMOTIONS}")
    
    paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
    print(f"\n4-class load: {len(paths)} samples")
    print(f"Label mapping: {mapping}")

