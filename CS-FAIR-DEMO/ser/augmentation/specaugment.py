"""SpecAugment: Time and frequency masking for spectrogram augmentation."""

import numpy as np


class SpecAugment:
    """SpecAugment: Time and frequency masking for spectrogram augmentation.
    Proven effective for speech emotion recognition."""
    
    def __init__(self, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
    
    def __call__(self, mel_spec):
        """
        Args:
            mel_spec: [n_mels, n_frames] spectrogram
        Returns:
            Augmented spectrogram
        """
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, mel_spec.shape[0]))
            if f > 0 and mel_spec.shape[0] > f:
                f0 = np.random.randint(0, mel_spec.shape[0] - f)
                mel_spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, mel_spec.shape[1]))
            if t > 0 and mel_spec.shape[1] > t:
                t0 = np.random.randint(0, mel_spec.shape[1] - t)
                mel_spec[:, t0:t0+t] = 0
        
        return mel_spec

