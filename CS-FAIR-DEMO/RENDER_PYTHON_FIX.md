# Fixing Python 3.13 / Numba Compatibility Issue

## Problem
Render was using Python 3.13.4, which has compatibility issues with numba/librosa JIT compilation. This caused:
- `NotImplementedError` in numba when librosa tried to compile audio processing functions
- Worker timeout (30s default too short for model loading)
- Worker killed (possible memory issues)

## Solutions Applied

### 1. Python Version Fix
- Created `runtime.txt` with `python-3.10.19`
- Updated `render.yaml` to specify Python 3.10.19
- Python 3.10 is well-tested with numba/librosa

### 2. Gunicorn Timeout Increase
- Changed start command to: `gunicorn app:app --timeout 120 --workers 1 --threads 2`
- **120 second timeout** (enough for model loading + processing)
- **1 worker** (to avoid memory issues)
- **2 threads** (for concurrent requests)

### 3. Fallback for librosa.trim()
- Added try/except around `librosa.effects.trim()`
- Falls back to simple energy-based trimming if numba fails
- Prevents crashes from numba compatibility issues

## Next Steps

1. **Push these changes to GitHub**
2. **Render will auto-deploy** with Python 3.10
3. **Monitor logs** to confirm it's working

## If Issues Persist

### Option 1: Disable Numba JIT (slower but more compatible)
Add to `app.py` at the top:
```python
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
```

### Option 2: Use Python 3.11 instead
Change `runtime.txt` to:
```
python-3.11.9
```

### Option 3: Disable noise reduction temporarily
In `audio_preprocessor.py`, set `noise_reduce=False` to speed up processing.

