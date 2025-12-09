"""
IEMOCAP Dataset Loader
======================

Interactive Emotional Dyadic Motion Capture
~10,000 utterances from 10 actors (5 sessions, 2 actors each)
English, conversational and scripted
Requires: iemocap_manifest.tsv file with pre-processed labels

Emotions available (from annotations):
    - Anger (includes Frustrated)
    - Happy (includes Excited)
    - Neutral
    - Sad
    - Fear (limited)
    - Surprise (limited)
"""

MODULE_INFO = {
    'description': 'IEMOCAP dataset loader - ~10,000 utterances, 6 emotions, English conversational',
    'inputs': ['List of emotions to load'],
    'outputs': ['File paths and labels'],
    'emotions': ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise'],
    'notes': 'Requires iemocap_manifest.tsv with pre-processed labels',
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple

from .base import BaseDatasetLoader


class IEMOCAPLoader(BaseDatasetLoader):
    """Loader for IEMOCAP dataset."""
    
    DATASET_NAME = "IEMOCAP"
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
        """Load all IEMOCAP data from manifest or by scanning files."""
        iemocap_dir = self.data_root / 'IEMOCAP'
        manifest_path = iemocap_dir / 'iemocap_manifest.tsv'
        
        file_paths = []
        emotions = []
        
        # Try manifest first
        if manifest_path.exists():
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
        else:
            # Fallback: scan wav files directly from filename patterns
            # IEMOCAP files are named like: ANG_00012.wav, HAP_00027.wav, etc.
            emotion_map = {
                'ANG': 'Anger',
                'HAP': 'Happy',
                'NEU': 'Neutral',
                'SAD': 'Sad',
                'FEA': 'Fear',
                'SUR': 'Surprise',
                'DIS': 'Anger',  # Disgust → Anger
                'FRU': 'Anger',  # Frustrated → Anger
                'EXC': 'Happy',  # Excited → Happy
                'OTH': 'Neutral',  # Other → Neutral
            }
            
            for wav_file in iemocap_dir.glob('*.wav'):
                # Extract emotion code from filename (first 3 chars before underscore)
                name = wav_file.stem.upper()
                emotion_code = name.split('_')[0] if '_' in name else name[:3]
                
                if emotion_code in emotion_map:
                    file_paths.append(str(wav_file))
                    emotions.append(emotion_map[emotion_code])
        
        print(f"  IEMOCAP: Found {len(file_paths)} samples")
        if return_weights:
            return file_paths, emotions, [1.0] * len(file_paths)
        return file_paths, emotions


def load(emotions: List[str], data_root: Path = None, skip_missing: bool = False):
    """
    Convenience function to load IEMOCAP data.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    loader = IEMOCAPLoader(data_root)
    return loader.load(emotions, skip_missing=skip_missing)


if __name__ == '__main__':
    print("Testing IEMOCAP Loader...")
    print(f"Available emotions: {IEMOCAPLoader.AVAILABLE_EMOTIONS}")
    
    try:
        paths, labels, mapping = load(['Anger', 'Happy', 'Neutral', 'Sad'])
        print(f"\n4-class load: {len(paths)} samples")
        print(f"Label mapping: {mapping}")
    except Exception as e:
        print(f"Error: {e}")

