"""
TESS Dataset Loader
===================

Toronto Emotional Speech Set
~2,800 samples from 2 female actors (younger and older)
Canadian English

Folder structure: OAF_xxx or YAF_xxx folders
Emotion in folder/filename:
    - angry
    - disgust
    - fear
    - happy/happiness
    - neutral
    - ps/pleasant_surprise/surprise
    - sad
"""

MODULE_INFO = {
    'description': 'TESS dataset loader - 2,800 samples, 7 emotions, Canadian English',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust'],
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple

from .base import BaseDatasetLoader


class TESSLoader(BaseDatasetLoader):
    """Loader for TESS dataset."""
    
    DATASET_NAME = "TESS"
    AVAILABLE_EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise', 'Disgust']
    
    # Keywords to look for in filenames/paths
    EMOTION_KEYWORDS = {
        'angry': 'Anger',
        'anger': 'Anger',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'happiness': 'Happy',
        'neutral': 'Neutral',
        'ps': 'Surprise',
        'pleasant_surprise': 'Surprise',
        'pleasant surprise': 'Surprise',
        'surprise': 'Surprise',
        'sad': 'Sad',
        'sadness': 'Sad',
    }
    
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """Load all TESS data."""
        audio_dir = self.data_root / 'TESS'
        
        if not audio_dir.exists():
            print(f"Warning: TESS directory not found at {audio_dir}")
            if return_weights:
                return [], [], []
            return [], []
        
        file_paths = []
        emotions = []
        
        for wav_file in audio_dir.rglob('*.wav'):
            # Check filename and parent folder for emotion keywords
            search_str = (wav_file.stem + '_' + wav_file.parent.name).lower()
            
            for keyword, emotion in self.EMOTION_KEYWORDS.items():
                if keyword in search_str:
                    file_paths.append(str(wav_file))
                    emotions.append(emotion)
                    break
        
        print(f"  TESS: Found {len(file_paths)} samples")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load TESS data.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = TESSLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing TESS Loader...")
    print(f"Available emotions: {TESSLoader.AVAILABLE_EMOTIONS}")
    
    paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
    print(f"\n4-class load: {len(paths)} samples")
    print(f"Label mapping: {mapping}")

