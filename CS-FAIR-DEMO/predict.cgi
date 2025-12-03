#!/users/w/s/wsander/www-root/venv/bin/python3
"""
CGI script for emotion prediction.
This runs without WSGI - just needs to be executable and in the web directory.
"""

# CRITICAL: Set content type FIRST, before any imports that might fail
import sys
import os

# Set content type immediately - nothing before this!
print("Content-Type: application/json\n")
sys.stdout.flush()

# Now wrap everything in try-except to ensure we always return JSON
try:
    import json
    import cgi
    import traceback
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Import necessary libraries
    import numpy as np
    import torch
    import torch.nn.functional as F
    import librosa
    from ser.utils.audio_preprocessor import AudioPreprocessor
    from ser.models import CRNN
    from ser.data.dataset import N_MELS, HOP_LENGTH, N_FFT, MAX_DURATION, MAX_FRAMES
    
    # Model parameters
    SAMPLE_RATE = 22050
    EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad']
    EMOTION_COLORS = {'Anger': '#FF4444', 'Happy': '#FFD700', 'Neutral': '#4A90D9', 'Sad': '#6B7B8C'}
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE, target_rms=0.1, noise_reduce=True)
    
    # Global model (loaded once)
    model = None
    
    def load_model():
        """Load the trained model."""
        global model
        if model is not None:
            return
        
        from pathlib import Path
        model_path = Path(__file__).parent / 'checkpoints' / '4class' / 'crnn_emotion_model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # All output to stderr
        print("Loading model...", file=sys.stderr)
        sys.stderr.flush()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = CRNN(n_mels=N_MELS, num_classes=len(EMOTIONS))
        # Try different possible checkpoint keys
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        val_acc = checkpoint.get('val_accuracy', checkpoint.get('accuracy', None))
        if val_acc is not None:
            print(f"Model loaded! Validation accuracy: {val_acc*100:.2f}%", file=sys.stderr)
        else:
            print("Model loaded!", file=sys.stderr)
        sys.stderr.flush()
    
    def audio_to_spectrogram(audio, sr):
        """Convert audio to mel spectrogram (same as training)."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            fmin=0,
            fmax=sr//2
        )
        
        # Convert to log scale
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to fixed length
        if mel_db.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_FRAMES]
        
        # Add dimensions for model
        mel_db = mel_db[np.newaxis, :, :]  # (1, n_mels, time)
        mel_db = mel_db[:, np.newaxis, :, :]  # (1, 1, n_mels, time)
        
        return torch.FloatTensor(mel_db)
    
    def decode_audio(audio_b64):
        """Decode base64 audio to numpy array."""
        import base64
        import io
        import wave
        import struct
        
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
        
        # Convert to mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / (2 ** (sample_width * 8 - 1))
        
        # Resample if needed
        if framerate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=framerate, target_sr=SAMPLE_RATE)
        
        return audio
    
    # Main processing
    try:
        # Load model (first time only)
        print("Step 1: Loading model...", file=sys.stderr)
        sys.stderr.flush()
        load_model()
        
        # Check if model loaded
        if model is None:
            raise Exception("Model failed to load")
        print("Step 2: Model loaded successfully", file=sys.stderr)
        sys.stderr.flush()
        
        # Get POST data
        print("Step 3: Reading POST data...", file=sys.stderr)
        sys.stderr.flush()
        form = cgi.FieldStorage()
        
        # Read JSON from stdin
        content_length = int(os.environ.get('CONTENT_LENGTH', 0))
        if content_length > 0:
            post_data = sys.stdin.read(content_length)
            try:
                data = json.loads(post_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {str(e)}")
        else:
            # Try form data
            if 'audio' in form:
                data = {'audio': form['audio'].value}
            else:
                raise ValueError("No audio data provided")
        
        audio_b64 = data.get('audio')
        if not audio_b64:
            raise ValueError("No audio data")
        
        print(f"Step 4: Got audio data (length: {len(audio_b64)})", file=sys.stderr)
        sys.stderr.flush()
        
        # Decode and process audio
        print("Step 5: Decoding audio...", file=sys.stderr)
        sys.stderr.flush()
        audio = decode_audio(audio_b64)
        
        print("Step 6: Processing audio...", file=sys.stderr)
        sys.stderr.flush()
        audio = preprocessor.process(audio, sr=SAMPLE_RATE)
        
        # Convert to spectrogram
        print("Step 7: Creating spectrogram...", file=sys.stderr)
        sys.stderr.flush()
        spectrogram = audio_to_spectrogram(audio, SAMPLE_RATE)
        
        # Predict
        print("Step 8: Running prediction...", file=sys.stderr)
        sys.stderr.flush()
        with torch.no_grad():
            logits = model(spectrogram)
            probs = F.softmax(logits, dim=1).numpy()[0]
        
        predicted_idx = np.argmax(probs)
        predicted_emotion = EMOTIONS[predicted_idx]
        confidence = probs[predicted_idx]
        
        print("Step 9: Prediction complete", file=sys.stderr)
        sys.stderr.flush()
        
        # Return JSON response
        result = {
            'emotion': predicted_emotion,
            'confidence': float(confidence),
            'probabilities': {emo: float(probs[i]) for i, emo in enumerate(EMOTIONS)},
            'color': EMOTION_COLORS[predicted_emotion],
            'audio_length': len(audio) / SAMPLE_RATE
        }
        
        print(json.dumps(result))
        sys.stdout.flush()
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        # Log to stderr
        print(f"Error: {error_msg}", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        sys.stderr.flush()
        
        # Return JSON error with more details for debugging
        error_response = {
            'error': error_msg,
            'error_type': type(e).__name__
        }
        print(json.dumps(error_response))
        sys.stdout.flush()

except Exception as e:
    # Catch ANY error, including import errors
    import json
    error_msg = str(e)
    error_response = {'error': f'Script error: {error_msg}'}
    print(json.dumps(error_response))
    sys.stdout.flush()
