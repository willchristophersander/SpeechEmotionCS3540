"""
Real-time Speech Emotion Recognition Demo
Using CRNN (CNN + BiLSTM) on Log-Mel Spectrograms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
import wave
import struct

app = Flask(__name__)
CORS(app)

# Model parameters (must match training)
SAMPLE_RATE = 22050

from ser.utils.audio_preprocessor import AudioPreprocessor
from ser.models import CRNN
from ser.data.dataset import N_MELS, HOP_LENGTH, N_FFT, MAX_DURATION, MAX_FRAMES

# Initialize preprocessor with EXACT same parameters as training
preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE, target_rms=0.1, noise_reduce=True)

EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad']
EMOTION_COLORS = {'Anger': '#FF4444', 'Happy': '#FFD700', 'Neutral': '#4A90D9', 'Sad': '#6B7B8C'}

# Global model
model = None


def load_model():
    global model
    
    from pathlib import Path
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / '4class'
    
    # Prefer regular model for accuracy (quantized may have slight accuracy loss)
    # Fall back to quantized if regular model not available
    regular_path = checkpoint_dir / 'crnn_emotion_model.pth'
    quantized_path = checkpoint_dir / 'crnn_emotion_model_quantized.pth'
    
    if regular_path.exists():
        print("Loading CRNN model (full precision for accuracy)...")
        model_path = regular_path
        is_quantized = False
    elif quantized_path.exists():
        print("Loading quantized CRNN model (optimized, may have slight accuracy loss)...")
        model_path = quantized_path
        is_quantized = True
    else:
        raise FileNotFoundError(f"Model not found. Checked: {regular_path} and {quantized_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = CRNN(n_mels=N_MELS, num_classes=len(EMOTIONS))
    
    # Load weights (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # If quantized, the model is already quantized in the checkpoint
    # For quantized models loaded from file, they're already in quantized state
    if is_quantized and checkpoint.get('quantized', False):
        print("  Using quantized model (4x smaller, 2x faster)")
    
    model.eval()
    
    # Get accuracy info
    val_acc = checkpoint.get('val_accuracy', checkpoint.get('accuracy', None))
    epoch = checkpoint.get('epoch', 'unknown')
    
    if val_acc is not None:
        print(f"Model loaded! Validation accuracy: {val_acc*100:.2f}% (epoch {epoch})")
    else:
        print(f"Model loaded! (epoch {epoch})")


def audio_to_spectrogram(audio, sr):
    """Convert audio to log-mel spectrogram."""
    # Note: Audio is already preprocessed (volume normalized, noise reduced, trimmed)
    # by AudioPreprocessor before this function is called
    
    # Extract log-mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [-1, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    mel_db = mel_db * 2 - 1
    
    # Pad or truncate to fixed length
    if mel_db.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    
    # Add batch dimension (channel dimension will be added by model or DataLoader)
    # Match training: mel_db[np.newaxis, :, :] → (1, n_mels, time)
    mel_db = mel_db[np.newaxis, :, :]  # (1, n_mels, time)
    
    # Add channel dimension for Conv2d: (batch, channels, height, width)
    mel_db = mel_db[:, np.newaxis, :, :]  # (1, 1, n_mels, time)
    
    return torch.FloatTensor(mel_db)


def decode_audio(audio_b64):
    """Decode base64 audio to numpy array.
    
    Returns raw audio - preprocessing will be done by AudioPreprocessor
    (same as training pipeline).
    """
    audio_bytes = base64.b64decode(audio_b64)
    audio_io = io.BytesIO(audio_bytes)
    
    with wave.open(audio_io, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_data = wav.readframes(n_frames)
    
    if sample_width == 2:
        audio = np.array(struct.unpack(f'{n_frames * n_channels}h', audio_data), dtype=np.float32)
    elif sample_width == 4:
        audio = np.array(struct.unpack(f'{n_frames * n_channels}i', audio_data), dtype=np.float32)
    else:
        audio = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) - 128
    
    # Convert to mono (if stereo)
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    
    # Normalize to [-1, 1] range (but don't peak normalize - let preprocessor handle RMS normalization)
    if np.abs(audio).max() > 0:
        audio = audio / (2 ** (sample_width * 8 - 1))  # Scale by bit depth
    
    # Resample to model's sample rate (preprocessor expects this)
    if framerate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=framerate, target_sr=SAMPLE_RATE)
    
    return audio


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion from audio."""
    import sys
    try:
        data = request.json
        audio_b64 = data.get('audio')
        
        if not audio_b64:
            return jsonify({'error': 'No audio data'}), 400
        
        # Decode audio (raw, unprocessed - same as librosa.load() output)
        audio = decode_audio(audio_b64)
        
        print(f"\n{'='*50}", flush=True)
        print(f"Prediction request:", flush=True)
        print(f"  Raw audio: {len(audio)/SAMPLE_RATE:.2f}s, RMS={np.sqrt(np.mean(audio**2)):.4f}", flush=True)
        
        # Apply unified preprocessing (EXACT same pipeline as training!)
        # preprocessor.process() handles: resampling, mono conversion, noise reduction, trimming, volume normalization
        audio = preprocessor.process(audio, sr=SAMPLE_RATE)
        print(f"  After preprocessing: {len(audio)/SAMPLE_RATE:.2f}s, RMS={np.sqrt(np.mean(audio**2)):.4f}", flush=True)
        
        # Save debug audio
        import scipy.io.wavfile as wav
        wav.write('debug_audio.wav', SAMPLE_RATE, (audio * 32767).astype(np.int16))
        print(f"  Saved to debug_audio.wav", flush=True)
        
        # Convert to spectrogram
        spectrogram = audio_to_spectrogram(audio, SAMPLE_RATE)
        print(f"  Spectrogram shape: {spectrogram.shape}", flush=True)
        print(f"  Spectrogram range: [{spectrogram.min():.3f}, {spectrogram.max():.3f}]", flush=True)
        
        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(spectrogram)
            probs = F.softmax(logits, dim=1).numpy()[0]
        
        predicted_idx = np.argmax(probs)
        predicted_emotion = EMOTIONS[predicted_idx]
        confidence = probs[predicted_idx]
        
        print(f"  Prediction: {predicted_emotion} ({confidence*100:.1f}%)", flush=True)
        print(f"  Probs: A:{probs[0]*100:.1f}% H:{probs[1]*100:.1f}% N:{probs[2]*100:.1f}% S:{probs[3]*100:.1f}%", flush=True)
        print(f"{'='*50}", flush=True)
        sys.stdout.flush()
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': float(confidence),
            'probabilities': {emo: float(probs[i]) for i, emo in enumerate(EMOTIONS)},
            'color': EMOTION_COLORS[predicted_emotion],
            'audio_length': len(audio) / SAMPLE_RATE
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    from pathlib import Path
    model_path = Path(__file__).parent / 'checkpoints' / '4class' / 'crnn_emotion_model.pth'
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_type': 'CRNN',
        'model_path': str(model_path),
        'model_exists': model_path.exists(),
        'emotions': EMOTIONS,
        'num_classes': len(EMOTIONS)
    })


# Load model when module is imported (for WSGI deployment)
# This ensures the model is loaded whether running directly or via WSGI
try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load model during import: {e}")
    print("Model will be loaded on first request if running via WSGI")

if __name__ == '__main__':
    # For local development
    load_model()
    print("\n" + "="*50)
    print("Speech Emotion Recognition Demo (CRNN)")
    print("="*50)
    print("Open http://localhost:5001 in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
