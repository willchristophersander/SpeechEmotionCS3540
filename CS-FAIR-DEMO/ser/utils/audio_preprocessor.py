"""
Unified Audio Preprocessing Pipeline
=====================================
Apply EXACTLY the same processing to training data and demo audio.
This ensures the model sees consistent input characteristics.
"""

import numpy as np
import librosa
import noisereduce as nr


class AudioPreprocessor:
    """Unified audio preprocessing for training and inference."""
    
    def __init__(self, sample_rate=22050, target_rms=0.1, noise_reduce=True):
        self.sample_rate = sample_rate
        self.target_rms = target_rms
        self.noise_reduce = noise_reduce
    
    def process(self, audio, sr=None):
        """
        Apply full preprocessing pipeline.
        
        Steps:
        1. Resample to target sample rate
        2. Convert to mono
        3. Noise reduction (voice isolation)
        4. Trim silence
        5. Volume normalization
        
        Args:
            audio: Audio signal (numpy array)
            sr: Source sample rate (if different from target)
            
        Returns:
            Processed audio at target sample rate
        """
        # 1. Resample if needed
        if sr is not None and sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # 2. Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 3. Noise reduction (voice isolation)
        if self.noise_reduce and len(audio) > self.sample_rate * 0.5:
            # Use stationary noise reduction - fast and effective
            audio = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate,
                prop_decrease=0.6,  # Less aggressive to preserve voice quality
                stationary=True,    # Faster
                n_fft=2048,
                hop_length=512
            )
        
        # 4. Trim silence from beginning and end
        # Workaround for numba/librosa compatibility issues on Python 3.13
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
            if len(audio_trimmed) > self.sample_rate * 0.3:  # At least 0.3s
                audio = audio_trimmed
        except (NotImplementedError, SystemError, Exception) as e:
            # Fallback: simple energy-based trimming if librosa.trim fails
            # This can happen with numba/librosa compatibility issues
            import warnings
            warnings.warn(f"librosa.trim failed ({type(e).__name__}), using fallback trimming")
            # Simple fallback: remove leading/trailing silence based on RMS
            rms = np.sqrt(np.mean(audio ** 2))
            threshold = rms * 0.1  # 10% of RMS as silence threshold
            non_silent = np.where(np.abs(audio) > threshold)[0]
            if len(non_silent) > self.sample_rate * 0.3:
                audio = audio[non_silent[0]:non_silent[-1]+1]
        
        # 5. Volume normalization to consistent RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0.001:
            audio = audio * (self.target_rms / rms)
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def load_and_process(self, filepath):
        """Load audio file and apply preprocessing."""
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        return self.process(audio)


# Global preprocessor instance
preprocessor = AudioPreprocessor()


def preprocess_audio(audio, sr=None):
    """Convenience function for preprocessing."""
    return preprocessor.process(audio, sr)


def load_and_preprocess(filepath):
    """Convenience function to load and preprocess audio file."""
    return preprocessor.load_and_process(filepath)


if __name__ == '__main__':
    # Test the preprocessor
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Processing: {filepath}")
        
        # Load raw
        raw, sr = librosa.load(filepath, sr=22050)
        print(f"  Raw: {len(raw)/sr:.2f}s, RMS={np.sqrt(np.mean(raw**2)):.4f}")
        
        # Process
        processed = load_and_preprocess(filepath)
        print(f"  Processed: {len(processed)/22050:.2f}s, RMS={np.sqrt(np.mean(processed**2)):.4f}")
        
        # Compare spectral characteristics
        cent_raw = np.mean(librosa.feature.spectral_centroid(y=raw, sr=sr))
        cent_proc = np.mean(librosa.feature.spectral_centroid(y=processed, sr=22050))
        print(f"  Spectral centroid: {cent_raw:.0f} Hz -> {cent_proc:.0f} Hz")
    else:
        print("Usage: python audio_preprocessor.py <audio_file>")

