"""
RAVDESS Songs Dataset Loader
============================

Ryerson Audio-Visual Database of Emotional Speech and Song - SONGS ONLY
~2,880 song samples from 24 actors (12 male, 12 female)
English, professional actors, very high quality

File format: XX-XX-XX-XX-XX-XX-XX.wav
    - Position 2 (index 1) should be '02' (song)
    - Position 3 (index 2) is emotion code

Location: DataSets/RAVDESS Emotional song audio/

Emotion codes:
    01 = neutral
    02 = calm (mapped to Neutral)
    03 = happy
    04 = sad
    05 = angry
    06 = fearful
    07 = disgust
    08 = surprised
"""

MODULE_INFO = {
    'description': 'RAVDESS songs dataset loader - ~2,880 song samples, 8 emotions, English',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust'],
    'location': 'DataSets/RAVDESS Emotional song audio/',
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple, Dict

from .base import BaseDatasetLoader


class RAVDESSSongsLoader(BaseDatasetLoader):
    """Loader for RAVDESS Emotional song audio dataset."""
    
    DATASET_NAME = "RAVDESS Songs"
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
        """Load all RAVDESS song data."""
        # Songs are in "RAVDESS Emotional song audio" directory
        song_dir = self.data_root / 'RAVDESS Emotional song audio'
        
        if not song_dir.exists():
            print(f"Warning: RAVDESS songs directory not found at {song_dir}")
            if return_weights:
                return [], [], []
            return [], []
        
        file_paths = []
        emotions = []
        
        # RAVDESS songs have nested actor folders
        # Filename format: XX-02-XX-XX-XX-XX-XX.wav
        # Position 1 (index 1): vocal_channel (should be '02' for songs)
        # Position 2 (index 2): emotion_code
        for wav_file in song_dir.rglob('*.wav'):
            # Parse filename: 03-02-05-01-01-01-12.wav
            parts = wav_file.stem.split('-')
            if len(parts) >= 3:
                vocal_channel = parts[1]  # Should be '02' for songs
                emotion_code = parts[2]
                
                # Only load song files (vocal_channel='02')
                if vocal_channel == '02' and emotion_code in self.EMOTION_MAP:
                    file_paths.append(str(wav_file))
                    emotions.append(self.EMOTION_MAP[emotion_code])
        
        print(f"  RAVDESS Songs: Found {len(file_paths)} samples")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load RAVDESS song data.
    
    Args:
        emotions: List of emotions to load
        data_root: Path to DataSets directory
        skip_missing: If True, skip unavailable emotions
    
    Returns:
        Tuple of (file_paths, labels, emotion_to_idx)
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = RAVDESSSongsLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing RAVDESS Songs Loader...")
    print(f"Available emotions: {RAVDESSSongsLoader.AVAILABLE_EMOTIONS}")
    
    paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
    print(f"\n4-class load: {len(paths)} samples")
    print(f"Label mapping: {mapping}")

