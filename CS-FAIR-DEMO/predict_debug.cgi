#!/users/w/s/wsander/www-root/venv/bin/python3
"""
Debug version of CGI script - shows errors in response for debugging
"""

import sys
import os
import json
import traceback

# Set content type FIRST
print("Content-Type: application/json\n")
sys.stdout.flush()

try:
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Test imports one by one
    try:
        import numpy as np
        print("numpy OK", file=sys.stderr)
    except Exception as e:
        raise Exception(f"numpy import failed: {e}")
    
    try:
        import torch
        print("torch OK", file=sys.stderr)
    except Exception as e:
        raise Exception(f"torch import failed: {e}")
    
    try:
        import librosa
        print("librosa OK", file=sys.stderr)
    except Exception as e:
        raise Exception(f"librosa import failed: {e}")
    
    try:
        from ser.utils.audio_preprocessor import AudioPreprocessor
        print("AudioPreprocessor OK", file=sys.stderr)
    except Exception as e:
        raise Exception(f"AudioPreprocessor import failed: {e}")
    
    try:
        from ser.models import CRNN
        print("CRNN OK", file=sys.stderr)
    except Exception as e:
        raise Exception(f"CRNN import failed: {e}")
    
    try:
        from ser.data.dataset import N_MELS, HOP_LENGTH, N_FFT, MAX_DURATION, MAX_FRAMES
        print("dataset constants OK", file=sys.stderr)
    except Exception as e:
        raise Exception(f"dataset constants import failed: {e}")
    
    # If we get here, all imports worked
    result = {
        'status': 'success',
        'message': 'All imports successful',
        'python_path': sys.executable,
        'current_dir': os.getcwd(),
        'script_dir': os.path.dirname(__file__)
    }
    print(json.dumps(result))
    sys.stdout.flush()
    
except Exception as e:
    error_trace = traceback.format_exc()
    error_response = {
        'error': str(e),
        'traceback': error_trace,
        'python_path': sys.executable,
        'current_dir': os.getcwd() if 'os' in dir() else 'unknown',
        'script_dir': os.path.dirname(__file__) if '__file__' in dir() else 'unknown'
    }
    print(json.dumps(error_response))
    sys.stdout.flush()

