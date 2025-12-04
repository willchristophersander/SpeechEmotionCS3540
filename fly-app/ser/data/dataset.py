"""Dataset class for speech emotion recognition."""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

from ..utils.audio_preprocessor import AudioPreprocessor
from ..augmentation.specaugment import SpecAugment


# Audio parameters
SAMPLE_RATE = 22050
N_MELS = 96  # Increased from 80 to 96 for richer features
HOP_LENGTH = 512
N_FFT = 2048
MAX_DURATION = 4.0
MAX_FRAMES = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH)

# Unified preprocessor (same as demo)
preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE, target_rms=0.1, noise_reduce=True)


class EmotionDataset(Dataset):
    """Dataset with unified preprocessing and SpecAugment support."""
    
    def __init__(self, file_paths, labels, augment=False, use_specaugment=False, n_mels=N_MELS):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.use_specaugment = use_specaugment
        self.specaugment = SpecAugment() if use_specaugment else None
        self.n_mels = n_mels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Use unified preprocessor (same as demo!)
            y = preprocessor.load_and_process(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            y = np.zeros(SAMPLE_RATE)
        
        if self.augment:
            y = self._augment(y)
        
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=self.n_mels, 
            hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_db = mel_db * 2 - 1
        
        # Apply SpecAugment if enabled (spectrogram augmentation)
        if self.use_specaugment and self.augment:
            mel_db = self.specaugment(mel_db)
        
        # Pad or truncate
        if mel_db.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_FRAMES]
        
        mel_db = mel_db[np.newaxis, :, :]
        
        return torch.FloatTensor(mel_db), torch.LongTensor([label])[0]
    
    def _augment(self, y):
        # Pitch shift (more range for diversity)
        if np.random.random() < 0.5:
            n_steps = np.random.uniform(-4, 4)
            y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
        
        # Time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.85, 1.15)
            y = librosa.effects.time_stretch(y, rate=rate)
        
        # Add noise (more aggressive to simulate real mics)
        if np.random.random() < 0.5:
            snr = np.random.uniform(15, 30)  # Signal-to-noise ratio in dB
            signal_power = np.mean(y ** 2)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = np.random.randn(len(y)) * np.sqrt(noise_power)
            y = y + noise
        
        # Random volume variation (simulate different mic distances)
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.5, 1.5)
            y = y * gain
            y = np.clip(y, -1.0, 1.0)
        
        # Random low-pass filter (simulate different mics - laptop mics often have less high freq)
        if np.random.random() < 0.3:
            cutoff = np.random.uniform(4000, 8000)  # Hz
            nyq = SAMPLE_RATE / 2
            b, a = butter(4, cutoff / nyq, btype='low')
            y = filtfilt(b, a, y)
        
        return y

