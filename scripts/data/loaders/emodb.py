"""
EmoDB Dataset Loader
====================

Berlin Database of Emotional Speech (EmoDB)
~535 samples from 10 German actors (5 male, 5 female)
German language (useful for cross-lingual testing)
Requires: emodb_manifest.tsv file with pre-processed labels

Emotions available:
    - Anger (W)
    - Happy (F)
    - Neutral (N)
    - Sad (T)
    - Fear (A)
    
Note: EmoDB does NOT have Surprise emotion.
"""

MODULE_INFO = {
    'description': 'EmoDB dataset loader - ~535 samples, 5 emotions, German',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear'],
    'language': 'German',
    'notes': 'EmoDB does not have Surprise. Requires emodb_manifest.tsv with pre-processed labels',
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple

from .base import BaseDatasetLoader


class EmoDBLoader(BaseDatasetLoader):
    """Loader for EmoDB (German) dataset."""
    
    DATASET_NAME = "EmoDB"
    AVAILABLE_EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear']
    
    # Label index to emotion name (from manifest)
    # EmoDB mapping: 0=Anger(W), 1=Happy(F), 2=Neutral(N), 3=Sad(T), 4=Fear(A)
    LABEL_TO_EMOTION = {
        0: 'Anger',
        1: 'Happy',
        2: 'Neutral',
        3: 'Sad',
        4: 'Fear',
    }
    
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """Load all EmoDB data from manifest."""
        emodb_dir = self.data_root / 'EmoDB'
        manifest_path = emodb_dir / 'emodb_manifest.tsv'
        
        if not manifest_path.exists():
            print(f"Warning: EmoDB manifest not found at {manifest_path}")
            print("  Create emodb_manifest.tsv with format: path<tab>label_idx")
            if return_weights:
                return [], [], []
            return [], []
        
        file_paths = []
        emotions = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    wav_path = Path(parts[0])
                    label_idx = int(parts[1])
                    
                    if wav_path.exists() and label_idx in self.LABEL_TO_EMOTION:
                        file_paths.append(str(wav_path))
                        emotions.append(self.LABEL_TO_EMOTION[label_idx])
        
        print(f"  EmoDB: Found {len(file_paths)} samples")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load EmoDB data.
    
    Args:
        emotions: List of emotions to load
        data_root: Path to DataSets directory
        skip_missing: If True, skip unavailable emotions
    
    Returns:
        Tuple of (file_paths, labels, emotion_to_idx)
    """
    if data_root is None:
        # scripts/data/loaders/emodb.py -> scripts/data/loaders -> scripts/data -> scripts -> project_root
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = EmoDBLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing EmoDB Loader...")
    print(f"Available emotions: {EmoDBLoader.AVAILABLE_EMOTIONS}")
    
    try:
        paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
        print(f"\n4-class load: {len(paths)} samples")
        print(f"Label mapping: {mapping}")
    except Exception as e:
        print(f"Error: {e}")

