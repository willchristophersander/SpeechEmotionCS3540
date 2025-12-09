# loaders

<!-- STATIC_DESCRIPTION_START -->
## Overview

*Add a description of this folder's purpose here. This section is preserved when auto-updating.*

<!-- STATIC_DESCRIPTION_END -->

## Files

<!-- AUTO_GENERATED_START -->
### `__init__.py`

**Status:** ✅ production

Unified dataset loading system with emotion validation

**Inputs:**
- List of emotions
- List of datasets

**Outputs:**
- File paths, labels, and emotion mapping

======================================================================

### `base.py`

**Status:** ✅ production

Base class and utilities for dataset loaders

**Inputs:**
- Emotion classes to load

**Outputs:**
- File paths and labels

======================================================================

### `cremad.py`

**Status:** ✅ production

CREMA-D dataset loader - 7,400 samples, 6 emotions, English

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Disgust

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.cremad import CREMADLoader
from pathlib import Path

loader = CREMADLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `emodb.py`

**Status:** ✅ production

EmoDB dataset loader - ~535 samples, 5 emotions, German

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear

**Language:** German

**Note:** EmoDB does not have Surprise. Requires emodb_manifest.tsv with pre-processed labels

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.emodb import EmoDBLoader
from pathlib import Path

loader = EmoDBLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `iemocap.py`

**Status:** ✅ production

IEMOCAP dataset loader - ~10,000 utterances, 6 emotions, English conversational

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Surprise

**Note:** Requires iemocap_manifest.tsv with pre-processed labels

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.iemocap import IEMOCAPLoader
from pathlib import Path

loader = IEMOCAPLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `nemo.py`

**Status:** ✅ production

nEMO dataset loader - ~4,500 samples, 6 emotions, Polish

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Surprise

**Language:** Polish

**Note:** Requires nemo_manifest.tsv with pre-processed labels

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.nemo import nEMOLoader
from pathlib import Path

loader = nEMOLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `ravdess.py`

**Status:** ✅ production

RAVDESS speech dataset loader - ~2,880 speech samples, 8 emotions, English

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Surprise, Disgust

**Note:** RAVDESS speech only. For songs, use RAVDESSSongsLoader

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.ravdess import RAVDESSLoader
from pathlib import Path

loader = RAVDESSLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `ravdess_songs.py`

**Status:** ✅ production

RAVDESS songs dataset loader - ~2,880 song samples, 8 emotions, English

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Surprise, Disgust

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.ravdess_songs import RAVDESSSongsLoader
from pathlib import Path

loader = RAVDESSSongsLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `savee.py`

**Status:** ✅ production

SAVEE dataset loader - 480 samples, 7 emotions, British English

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Surprise, Disgust

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.savee import SAVEELoader
from pathlib import Path

loader = SAVEELoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `tess.py`

**Status:** ✅ production

TESS dataset loader - 2,800 samples, 7 emotions, Canadian English

**Available Emotions:**
- Anger, Happy, Neutral, Sad, Fear, Surprise, Disgust

**Inputs:**
- List of emotions to load

**Outputs:**
- File paths and labels

**Example Usage:**
```python
from scripts.data.loaders.tess import TESSLoader
from pathlib import Path

loader = TESSLoader(Path('DataSets'))
paths, labels, mapping = loader.load(['Anger', 'Happy', 'Neutral', 'Sad'])
print(f'Loaded {len(paths)} samples')
```

======================================================================

### `unified.py`

**Status:** ✅ production

Unified loader that combines multiple datasets with emotion validation

**Inputs:**
- List of emotions
- List of dataset names

**Outputs:**
- Combined file paths, labels, emotion mapping, datasets used

**Example Usage:**
```python
from scripts.data import load_datasets

# Load 4-class emotions from multiple datasets
paths, labels, mapping, datasets_used = load_datasets(
    emotions=['Anger', 'Happy', 'Neutral', 'Sad'],
    datasets=['CREMA-D', 'RAVDESS', 'TESS'],
    skip_unavailable_emotions=False,
)
print(f'Loaded {len(paths)} samples from {datasets_used}')
```
<!-- AUTO_GENERATED_END -->

## Dependencies & Relations

<!-- AUTO_RELATIONS_START -->
### Internal Dependencies

*No internal dependencies detected.*

### External Dependencies

<!-- AUTO_RELATIONS_END -->

---
*Last updated: 2025-12-04 12:20:49*
