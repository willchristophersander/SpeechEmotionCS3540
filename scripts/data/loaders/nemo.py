"""
nEMO Dataset Loader
===================

Polish Emotional Speech Database
~4,500 samples from professional Polish actors
Polish language (useful for cross-lingual testing)
Requires: nemo_manifest.tsv file with pre-processed labels

Emotions available:
    - Anger
    - Happy
    - Neutral
    - Sad
    - Fear
    - Surprise
"""

MODULE_INFO = {
    'description': 'nEMO dataset loader - ~4,500 samples, 6 emotions, Polish',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise'],
    'language': 'Polish',
    'notes': 'Requires nemo_manifest.tsv with pre-processed labels',
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple

from .base import BaseDatasetLoader


class nEMOLoader(BaseDatasetLoader):
    """Loader for nEMO (Polish) dataset."""
    
    DATASET_NAME = "nEMO"
    AVAILABLE_EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise']
    
    # Label index to emotion name (from manifest)
    LABEL_TO_EMOTION = {
        0: 'Anger',
        1: 'Happy',
        2: 'Neutral',
        3: 'Sad',
        4: 'Fear',
        5: 'Surprise',
    }
    
    def _load_all(self, return_weights: bool = False) -> Tuple[List[str], List[str], ...]:
        """Load all nEMO data from manifest."""
        nemo_dir = self.data_root / 'nEMO'
        manifest_path = nemo_dir / 'nemo_manifest.tsv'
        
        if not manifest_path.exists():
            print(f"Warning: nEMO manifest not found at {manifest_path}")
            print("  Create nemo_manifest.tsv with format: path<tab>label_idx")
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
        
        print(f"  nEMO: Found {len(file_paths)} samples")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load nEMO data.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = nEMOLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing nEMO Loader...")
    print(f"Available emotions: {nEMOLoader.AVAILABLE_EMOTIONS}")
    
    try:
        paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
        print(f"\n4-class load: {len(paths)} samples")
        print(f"Label mapping: {mapping}")
    except Exception as e:
        print(f"Error: {e}")

