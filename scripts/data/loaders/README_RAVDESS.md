# RAVDESS Loaders Guide

## Two Separate Loaders

### 1. RAVDESS Speech (`ravdess.py`)
- **Location**: `DataSets/RAVDESS/`
- **Files**: Speech files with `vocal_channel='01'`
- **Samples**: ~2,880 speech samples

### 2. RAVDESS Songs (`ravdess_songs.py`)
- **Location**: `DataSets/RAVDESS Emotional song audio/`
- **Files**: Song files with `vocal_channel='02'`
- **Samples**: ~2,024 song samples

## Usage Examples

### Load Speech Only
```python
from scripts.data.loaders.ravdess import RAVDESSLoader

loader = RAVDESSLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
# Returns ~1,728 samples (4-class filtered)
```

### Load Songs Only
```python
from scripts.data.loaders.ravdess_songs import RAVDESSSongsLoader

loader = RAVDESSSongsLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
# Returns ~1,656 samples (4-class filtered)
```

### Load Both (Unified Loader)
```python
from scripts.data import load_datasets

# Automatically includes both when include_songs=True
paths, labels, mapping, used = load_datasets(
    emotions=['Anger', 'Happy', 'Neutral', 'Sad'],
    datasets=['RAVDESS'],
    include_songs=True,  # Adds 'RAVDESS Songs' automatically
)
# Returns ~3,384 samples total (speech + songs)
```

### Load Both Explicitly
```python
from scripts.data import load_datasets

paths, labels, mapping, used = load_datasets(
    emotions=['Anger', 'Happy', 'Neutral', 'Sad'],
    datasets=['RAVDESS', 'RAVDESS Songs'],  # Explicitly list both
)
# Returns ~3,384 samples total
```

## In Training Scripts

To use songs in training, add `'RAVDESS Songs'` to your dataset list:

```python
TRAINING_DATASETS = [
    'CREMA-D', 
    'RAVDESS',           # Speech
    'RAVDESS Songs',     # Songs
    'SAVEE', 
    'TESS', 
    'IEMOCAP', 
    'nEMO'
]
```

Or use `include_songs=True` in unified loader (automatically adds songs).

## File Structure

```
DataSets/
├── RAVDESS/                          ← Speech loader reads here
│   └── RAVDESSData/
│       ├── Actor_01/
│       │   ├── 03-01-05-01-01-01-01.wav  (speech, angry)
│       │   └── ...
│       └── ...
│
└── RAVDESS Emotional song audio/     ← Songs loader reads here
    ├── Actor_01/
    │   ├── 03-02-05-01-01-01-01.wav  (song, angry)
    │   └── ...
    └── ...
```

## Benefits of Separate Loaders

1. **Modularity**: Can use speech or songs independently
2. **Clarity**: Explicit control over what data is loaded
3. **Flexibility**: Easy to test impact of adding songs
4. **Performance tracking**: Can track speech vs songs separately in model registry





