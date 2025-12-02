"""
Configuration and Hyperparameters for 4-Class Emotion Recognition Training
============================================================================

This module contains all hyperparameters, training settings, and model configuration
for the 4-class CRNN emotion recognition model.
"""

import torch
from dataclasses import dataclass
from pathlib import Path

# Emotion classes
EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad']
NUM_CLASSES = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration."""
    
    # Model hyperparameters
    n_mels: int = 96
    num_classes: int = NUM_CLASSES
    dropout: float = 0.3
    
    # Optimizer hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Scheduler hyperparameters
    scheduler_t0: int = 10  # CosineAnnealingWarmRestarts T_0
    scheduler_t_mult: int = 2  # CosineAnnealingWarmRestarts T_mult
    scheduler_eta_min: float = 1e-6  # Minimum learning rate
    
    # Training hyperparameters
    max_epochs: int = 150
    warmup_epochs: int = 5
    early_stop_patience: int = 28
    checkpoint_interval: int = 10
    
    # Data loading hyperparameters
    batch_size: int = 40
    num_workers: int = 4  # Will be adjusted based on device availability
    
    # Loss function hyperparameters
    class_weights: list = None  # [Anger, Happy, Neutral, Sad]
    label_smoothing: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpoint paths
    checkpoint_dir: Path = None
    best_model_filename: str = 'crnn_emotion_model.pth'
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.class_weights is None:
            # Default class weights: Penalize over-prediction of anger
            self.class_weights = [0.7, 1.3, 1.0, 1.0]  # [Anger, Happy, Neutral, Sad]
        
        if self.checkpoint_dir is None:
            # Default checkpoint directory (relative to demo/scripts)
            self.checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints' / '4class'
    
    def get_class_weights_tensor(self, device):
        """Get class weights as a PyTorch tensor on the specified device."""
        return torch.tensor(self.class_weights, dtype=torch.float32).to(device)
    
    def get_best_model_path(self):
        """Get the full path to the best model checkpoint."""
        return self.checkpoint_dir / self.best_model_filename


# Default configuration instance
DEFAULT_CONFIG = TrainingConfig()

