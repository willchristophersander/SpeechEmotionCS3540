# Debugging 500 Internal Server Error

## What This Means

A 500 error means the CGI script is being executed, but it's crashing. This is progress from the previous issue!

## Steps to Debug

### Step 1: Test Debug Script

**On Silk:**
```bash
cd ~/www-root

# Upload predict_debug.cgi
# Then:
chmod +x predict_debug.cgi
```

Then test: `https://wsander.w3.uvm.edu/predict_debug.cgi`

This will show you exactly which import is failing.

### Step 2: Check Server Error Logs

**On Silk:**
```bash
# Check if there are error logs
ls -la ~/logs/ 2>/dev/null
# Or check Apache error logs (ask SAA where they are)
```

### Step 3: Test Imports Manually

**On Silk:**
```bash
cd ~/www-root
~/www-root/venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
import numpy
import torch
import librosa
from ser.utils.audio_preprocessor import AudioPreprocessor
from ser.models import CRNN
from ser.data.dataset import N_MELS, HOP_LENGTH, N_FFT
print('All imports OK')
"
```

### Step 4: Check File Paths

**On Silk:**
```bash
cd ~/www-root
ls -la checkpoints/4class/crnn_emotion_model.pth
# Should exist
```

### Step 5: Test Script Directly

**On Silk:**
```bash
cd ~/www-root
echo '{"audio":"test"}' | ~/www-root/venv/bin/python3 predict.cgi 2>&1 | head -20
```

This will show the actual error.

## Common Causes

1. **Import Error** - Missing package or wrong path
2. **Model File Not Found** - Wrong path to model
3. **Permission Error** - Can't read model file
4. **Memory Error** - Model too large
5. **Syntax Error** - Python syntax issue

## Quick Fix: Use Debug Script

The `predict_debug.cgi` script will tell you exactly what's wrong. Upload it and test it.

