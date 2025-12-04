"""Dataset loading functions for various emotion recognition datasets."""

from pathlib import Path


# PROJECT_ROOT: ser/data/loaders.py -> ser/data -> ser -> fly-app -> SpeechEmotionCS3540
# So we need parents[3] to go from loaders.py to SpeechEmotionCS3540
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # ser/data/loaders.py -> ser/data -> ser -> fly-app -> SpeechEmotionCS3540


def load_cremad():
    """Load CREMA-D.
    6-class mapping: 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
    """
    print("Loading CREMA-D...")
    emotion_map = {'ANG': 0, 'HAP': 1, 'NEU': 2, 'SAD': 3, 'FEA': 4, 'DIS': 0}  # DIS -> Anger (similar)
    audio_dir = PROJECT_ROOT / 'DataSets' / 'CREMA-D' / 'AudioWAV'
    files, labels = [], []
    
    for f in audio_dir.glob('*.wav'):
        parts = f.stem.split('_')
        if len(parts) >= 3 and parts[2] in emotion_map:
            files.append(str(f))
            labels.append(emotion_map[parts[2]])
    
    print(f"  Found {len(files)} samples")
    return files, labels


def load_emodb():
    """Load EmoDB from manifest if available, else directly from filenames."""
    print("Loading EmoDB...")
    emodb_dir = PROJECT_ROOT / 'DataSets' / 'EmoDB'
    manifest_path = emodb_dir / 'emodb_manifest.tsv'
    files, labels = [], []

    if manifest_path.exists():
        with manifest_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                wav_path = Path(parts[0])
                if wav_path.exists():
                    files.append(str(wav_path))
                    labels.append(int(parts[1]))  # Already 6-class from manifest
    else:
        # Fallback: infer labels directly from EmoDB filenames
        wav_dir = emodb_dir / 'wav'
        if not wav_dir.exists():
            print(f"  EmoDB wav directory not found: {wav_dir}")
            return files, labels

        # 6-class mapping: W=Anger(0), F=Happy(1), N=Neutral(2), T=Sad(3), A=Fear(4)
        # Note: EmoDB doesn't have Surprise, so we only use 0-4
        emo_map = {
            'W': 0,  # anger
            'F': 1,  # happiness
            'N': 2,  # neutral
            'T': 3,  # sadness
            'L': 2,  # boredom -> neutral
            'E': 0,  # disgust -> anger (similar)
            'A': 4,  # fear/anxiety -> fear
        }

        for f in wav_dir.glob('*.wav'):
            stem = f.stem
            if len(stem) < 6:
                continue
            emo_code = stem[5].upper()
            if emo_code not in emo_map:
                continue
            files.append(str(f))
            labels.append(emo_map[emo_code])

    print(f"  Found {len(files)} samples")
    return files, labels


def load_ravdess():
    """Load RAVDESS."""
    print("Loading RAVDESS...")
    # RAVDESS: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    # 6-class mapping: 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
    emotion_map = {
        1: 2,   # neutral
        2: 2,   # calm -> neutral
        3: 1,   # happy
        4: 3,   # sad
        5: 0,   # angry
        6: 4,   # fearful
        7: 0,   # disgust -> anger (similar)
        8: 5,   # surprised
    }
    audio_dir = PROJECT_ROOT / 'DataSets' / 'RAVDESS' / 'RAVDESSData'
    files, labels = [], []
    
    for actor_dir in audio_dir.glob('Actor_*'):
        for f in actor_dir.glob('*.wav'):
            parts = f.stem.split('-')
            if len(parts) >= 3:
                emo = int(parts[2])
                if emo in emotion_map:
                    files.append(str(f))
                    labels.append(emotion_map[emo])
    
    print(f"  Found {len(files)} samples")
    return files, labels


def load_savee():
    """Load SAVEE/SURREY."""
    print("Loading SAVEE...")
    audio_dir = PROJECT_ROOT / 'DataSets' / 'SURREY' / 'Audio'
    files, labels = [], []
    
    if not audio_dir.exists():
        print(f"  Directory not found: {audio_dir}")
        return files, labels
    
    for f in audio_dir.glob('*.wav'):
        fname = f.stem.lower()
        # SAVEE naming: DC_a01 (anger), DC_d01 (disgust), DC_f01 (fear), 
        #               DC_h01 (happy), DC_n01 (neutral), DC_sa01 (sad), DC_su01 (surprise)
        label_6class = None
        if '_a' in fname and '_sa' not in fname:
            label_6class = 0  # Anger
        elif '_h' in fname:
            label_6class = 1  # Happy
        elif '_n' in fname:
            label_6class = 2  # Neutral
        elif '_sa' in fname:
            label_6class = 3  # Sad
        elif '_f' in fname:
            label_6class = 4  # Fear
        elif '_su' in fname:
            label_6class = 5  # Surprise
        elif '_d' in fname:
            label_6class = 0  # Disgust -> Anger (similar)
        
        if label_6class is not None:
            files.append(str(f))
            labels.append(label_6class)
    
    print(f"  Found {len(files)} samples")
    return files, labels


def load_tess():
    """Load TESS."""
    print("Loading TESS...")
    # TESS: angry, disgust, fear, happy, neutral, pleasant_surprise, sad
    # 6-class mapping: 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
    emotion_map = {
        'angry': 0,
        'disgust': 0,  # Disgust -> Anger (similar)
        'fear': 4,
        'happy': 1,
        'neutral': 2,
        'pleasant_surprise': 5,
        'surprise': 5,
        'sad': 3,
    }
    audio_dir = PROJECT_ROOT / 'DataSets' / 'TESS' / 'TESS Toronto emotional speech set data'
    files, labels = [], []
    
    if not audio_dir.exists():
        audio_dir = PROJECT_ROOT / 'DataSets' / 'TESS'
    
    for subdir in audio_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.glob('*.wav'):
                fname = f.stem.lower()
                for emo, label_6class in emotion_map.items():
                    if emo in fname:
                        files.append(str(f))
                        labels.append(label_6class)
                        break
    
    print(f"  Found {len(files)} samples")
    return files, labels


def load_iemocap():
    """Load IEMOCAP."""
    print("Loading IEMOCAP...")
    # IEMOCAP: ANG, HAP, NEU, SAD, FEA, SUR, DIS, FRU, EXC, OTH
    # 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
    emotion_map = {
        'ANG': 0,  # Anger
        'HAP': 1,  # Happy
        'NEU': 2,  # Neutral
        'SAD': 3,  # Sad
        'FEA': 4,  # Fear
        'SUR': 5,  # Surprise
        'DIS': 0,  # Disgust -> Anger
        'FRU': 0,  # Frustrated -> Anger
        'EXC': 1,  # Excited -> Happy
        'OTH': 2,  # Other -> Neutral
    }
    audio_dir = PROJECT_ROOT / 'DataSets' / 'IEMOCAP'
    files, labels = [], []
    
    for f in audio_dir.glob('*.wav'):
        emo = f.stem.split('_')[0]
        if emo in emotion_map:
            files.append(str(f))
            labels.append(emotion_map[emo])
    
    print(f"  Found {len(files)} samples")
    return files, labels


def load_nemo():
    """Load nEMO from manifest file."""
    print("Loading nEMO...")
    nemo_dir = PROJECT_ROOT / 'DataSets' / 'nEMO'
    manifest_path = nemo_dir / 'nemo_manifest.tsv'
    files, labels = [], []
    
    if not manifest_path.exists():
        print(f"  nEMO manifest not found (run process_nemo.py first)")
        return files, labels
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                wav_path = Path(parts[0])
                if wav_path.exists():
                    files.append(str(wav_path))
                    labels.append(int(parts[1]))  # Already 6-class from manifest
    
    print(f"  Found {len(files)} samples")
    return files, labels


def load_meld():
    """Load MELD from manifest files (train + dev splits only, test kept separate).
    MELD emotions: anger, disgust, joy, surprise, neutral, sadness, fear
    6-class mapping: 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
    Note: MELD manifest should already have 6-class labels after processing.
    """
    print("Loading MELD...")
    meld_dir = PROJECT_ROOT / 'DataSets' / 'MELD'
    files, labels = [], []
    
    # Load from train and dev manifests (skip test to keep it separate)
    for split in ['train', 'dev']:
        manifest_path = meld_dir / f'meld_{split}_manifest.tsv'
        if not manifest_path.exists():
            print(f"  {split} manifest not found (run process_meld.py first)")
            continue
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    wav_path = Path(parts[0])
                    if wav_path.exists():
                        files.append(str(wav_path))
                        labels.append(int(parts[1]))  # Should be 6-class after processing
    
    print(f"  Found {len(files)} samples")
    return files, labels

