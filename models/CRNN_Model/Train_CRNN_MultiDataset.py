"""
CRNN Training Script - Multi-Dataset 4-Class Emotion Recognition
=================================================================

This is a standalone training script based on the best-performing model (77.26% accuracy).

Architecture:
    - 3-layer CNN (32→64→128 channels)
    - 2-layer Bidirectional LSTM (128 hidden)
    - Attention mechanism
    - 96 mel bands

Datasets (6 total):
    - CREMA-D: ~7,400 samples (English, professional actors)
    - RAVDESS: ~1,400 samples (English, high quality)
    - SAVEE: ~480 samples (British English)
    - TESS: ~2,400 samples (English, older females)
    - IEMOCAP: ~4,900 samples (English, conversational)
    - nEMO: ~4,500 samples (Polish, acted)

Usage:
    python scripts/training/Train_CRNN_MultiDataset.py
"""

MODULE_INFO = {
    'description': 'Standalone CRNN training script for 4-class emotion recognition using 6 datasets.',
    'inputs': [
        'Audio files from CREMA-D, RAVDESS, SAVEE, TESS, IEMOCAP, nEMO datasets',
        'Hyperparameters defined in TrainingConfig class'
    ],
    'outputs': [
        'Trained model checkpoint: models/crnn_multi/crnn_multi_dataset.pth',
        'Periodic checkpoints every 10 epochs'
    ],
    'architecture': '3-layer CNN (32→64→128) + 2-layer BiLSTM (128) + Attention',
    'best_accuracy': '77.26%',
    'dependencies': ['torch', 'librosa', 'numpy', 'sklearn', 'scipy', 'tqdm', 'noisereduce'],
    'status': 'production'
}

import warnings
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / 'DataSets'

# Emotion classes (4-class)
EMOTIONS = ['Anger', 'Happy', 'Neutral', 'Sad']
NUM_CLASSES = 4

# Audio parameters
SAMPLE_RATE = 22050
N_MELS = 96
HOP_LENGTH = 512
N_FFT = 2048
MAX_DURATION = 4.0
MAX_FRAMES = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Model
    n_mels: int = N_MELS
    num_classes: int = NUM_CLASSES
    dropout: float = 0.3
    
    # Optimizer
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Scheduler (CosineAnnealingWarmRestarts)
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6
    
    # Training
    max_epochs: int = 150
    warmup_epochs: int = 5
    early_stop_patience: int = 28
    checkpoint_interval: int = 10
    
    # Data
    batch_size: int = 40
    num_workers: int = 4
    
    # Loss (class weights: [Anger, Happy, Neutral, Sad])
    class_weights: list = None
    label_smoothing: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpoint
    checkpoint_dir: Path = None
    best_model_filename: str = 'crnn_multi_dataset.pth'
    
    def __post_init__(self):
        if self.class_weights is None:
            self.class_weights = [0.7, 1.3, 1.0, 1.0]  # Penalize anger, boost happy
        if self.checkpoint_dir is None:
            self.checkpoint_dir = PROJECT_ROOT / 'models' / 'crnn_multi'
    
    def get_class_weights_tensor(self, device):
        return torch.tensor(self.class_weights, dtype=torch.float32).to(device)


# ============================================================================
# MODEL: CRNN (Convolutional Recurrent Neural Network)
# ============================================================================

class CRNN(nn.Module):
    """CRNN for Speech Emotion Recognition - Based on 77.26% model."""
    
    def __init__(self, n_mels=96, num_classes=4, dropout=0.3):
        super().__init__()
        
        # CNN Feature Extraction (3 blocks)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        cnn_out_freq = n_mels // 8  # After 3 pooling: 96 → 12
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128 * cnn_out_freq,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classification
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape for LSTM
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        out = self.fc(context)
        return out


# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

def load_and_preprocess_audio(audio_path, target_rms=0.1, noise_reduce=True):
    """Load and preprocess audio file."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Check for valid audio
        if len(y) == 0:
            return np.zeros(SAMPLE_RATE)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) == 0:
            return np.zeros(SAMPLE_RATE)
        
        # Noise reduction (simple spectral gating)
        if noise_reduce and len(y) > 0:
            try:
                import noisereduce as nr
                y = nr.reduce_noise(y=y, sr=SAMPLE_RATE, prop_decrease=0.5)
            except ImportError:
                pass  # Skip if noisereduce not available
        
        # Replace any NaN/Inf with zeros
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # RMS normalization
        if len(y) > 0:
            rms = np.sqrt(np.mean(y**2))
            if rms > 1e-8:  # Avoid division by very small numbers
                y = y * (target_rms / rms)
                y = np.clip(y, -1.0, 1.0)
        
        # Final check for finite values
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.zeros(SAMPLE_RATE)


# ============================================================================
# SPECAUGMENT
# ============================================================================

class SpecAugment:
    """SpecAugment for spectrogram augmentation."""
    
    def __init__(self, freq_mask_param=10, time_mask_param=20, n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, mel_spec):
        mel_spec = mel_spec.copy()
        n_mels, n_frames = mel_spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel_spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_frames - t))
            mel_spec[:, t0:t0+t] = 0
        
        return mel_spec


# ============================================================================
# DATASET
# ============================================================================

class EmotionDataset(Dataset):
    """Dataset with preprocessing and augmentation."""
    
    def __init__(self, file_paths, labels, augment=False, use_specaugment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.specaugment = SpecAugment() if use_specaugment else None
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess
        y = load_and_preprocess_audio(audio_path)
        
        # Audio augmentation
        if self.augment:
            y = self._augment(y)
        
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
            hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_db = mel_db * 2 - 1
        
        # SpecAugment
        if self.specaugment and self.augment:
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
        # Skip augmentation if audio is too short or invalid
        if len(y) < 2048 or not np.all(np.isfinite(y)):
            return y
        
        try:
            # Pitch shift
            if np.random.random() < 0.5:
                n_steps = np.random.uniform(-4, 4)
                y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
            
            # Time stretch
            if np.random.random() < 0.3:
                rate = np.random.uniform(0.85, 1.15)
                y = librosa.effects.time_stretch(y, rate=rate)
            
            # Add noise
            if np.random.random() < 0.5:
                snr = np.random.uniform(15, 30)
                signal_power = np.mean(y ** 2)
                if signal_power > 1e-10:
                    noise_power = signal_power / (10 ** (snr / 10))
                    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
                    y = y + noise
            
            # Volume variation
            if np.random.random() < 0.5:
                gain = np.random.uniform(0.5, 1.5)
                y = y * gain
                y = np.clip(y, -1.0, 1.0)
            
            # Low-pass filter
            if np.random.random() < 0.3:
                cutoff = np.random.uniform(4000, 8000)
                nyq = SAMPLE_RATE / 2
                b, a = butter(4, cutoff / nyq, btype='low')
                y = filtfilt(b, a, y)
            
            # Ensure output is finite
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
        except Exception:
            pass  # Return original if augmentation fails
        
        return y


# ============================================================================
# DATA LOADING (using modular loaders)
# ============================================================================

# Add project root to path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Import the modular data loader
from scripts.data import load_datasets as load_data_modular

# Datasets to use for training (can be modified without changing loader code)
TRAINING_DATASETS = ['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS', 'IEMOCAP', 'nEMO']


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class DistanceWeightedLoss(nn.Module):
    """Distance-weighted cross-entropy loss with emotion relationships."""
    
    def __init__(self, distance_matrix, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.distance_matrix = distance_matrix
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        
        # Get predicted class
        probs = F.softmax(inputs, dim=1)
        _, predicted = torch.max(probs, dim=1)
        
        # Apply distance penalty for incorrect predictions
        distance_penalty = torch.ones_like(ce)
        for i in range(len(targets)):
            if predicted[i] != targets[i]:
                distance_penalty[i] = self.distance_matrix[targets[i], predicted[i]]
        
        return (ce * distance_penalty).mean()


# Emotion distance matrix (based on arousal-valence space)
EMOTION_DISTANCE_MATRIX = torch.tensor([
    [1.0, 0.9, 0.7, 0.6],  # Anger
    [0.9, 1.0, 0.6, 0.8],  # Happy
    [0.7, 0.6, 1.0, 0.8],  # Neutral
    [0.6, 0.8, 0.8, 1.0],  # Sad
])


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return train_loss / len(train_loader)


def validate(model, val_loader, criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            for i in range(len(y_batch)):
                label = y_batch[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    class_acc = [class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
                 for i in range(num_classes)]
    
    return avg_loss, accuracy, class_acc


def main():
    """Main training function."""
    config = TrainingConfig()
    
    print("="*70)
    print("CRNN Training - Multi-Dataset 4-Class Emotion Recognition")
    print("="*70)
    print(f"Model: CRNN (3 CNN + 2 BiLSTM + Attention)")
    print(f"Requested datasets: {', '.join(TRAINING_DATASETS)}")
    print(f"Emotions: {', '.join(EMOTIONS)}")
    print(f"Batch size: {config.batch_size}, LR: {config.learning_rate}")
    print("="*70)
    print()
    
    # Load data using modular loader
    # This will raise EmotionNotAvailableError if any dataset lacks a requested emotion
    all_files, all_labels, emotion_mapping, datasets_used = load_data_modular(
        emotions=EMOTIONS,
        datasets=TRAINING_DATASETS,
        data_root=DATASETS_DIR,
        skip_unavailable_datasets=True,  # Skip if dataset folder doesn't exist
        skip_unavailable_emotions=False,  # Error if emotion not available
    )
    
    if len(all_files) == 0:
        print("\nERROR: No data found!")
        return
    
    # Show which datasets were actually loaded
    missing_datasets = set(TRAINING_DATASETS) - set(datasets_used)
    if missing_datasets:
        print(f"\n⚠️  Note: Some datasets were skipped (not found): {', '.join(missing_datasets)}")
    print(f"✅ Datasets actually loaded: {', '.join(datasets_used)}")
    print()
    
    # Split data (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create data loaders
    train_dataset = EmotionDataset(X_train, y_train, augment=True, use_specaugment=True)
    val_dataset = EmotionDataset(X_val, y_val, augment=False, use_specaugment=False)
    test_dataset = EmotionDataset(X_test, y_test, augment=False, use_specaugment=False)
    
    num_workers = config.num_workers if torch.backends.mps.is_available() else 2
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    
    # Initialize model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = CRNN(n_mels=config.n_mels, num_classes=config.num_classes, dropout=config.dropout).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler_t0, T_mult=config.scheduler_t_mult, eta_min=config.scheduler_eta_min
    )
    
    # Loss function
    class_weights = config.get_class_weights_tensor(device)
    distance_matrix = EMOTION_DISTANCE_MATRIX.to(device)
    criterion = DistanceWeightedLoss(distance_matrix, class_weights, config.label_smoothing)
    
    print(f"Class weights: A={config.class_weights[0]}, H={config.class_weights[1]}, "
          f"N={config.class_weights[2]}, S={config.class_weights[3]}")
    
    # Setup checkpoints
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = config.checkpoint_dir / config.best_model_filename
    
    # Training loop
    print(f"\nTraining for up to {config.max_epochs} epochs...")
    print(f"Early stopping patience: {config.early_stop_patience}")
    print()
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(config.max_epochs):
        # Learning rate warmup
        if epoch < config.warmup_epochs:
            warmup_lr = config.learning_rate * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        elif epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Train
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        avg_val_loss, val_acc, val_class_acc = validate(model, val_loader, criterion, device, config.num_classes)
        
        # Check improvement
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            improved = True
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': avg_val_loss,
                'emotions': EMOTIONS
            }, best_model_path)
        else:
            epochs_without_improvement += 1
        
        # Periodic checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = config.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'emotions': EMOTIONS
            }, checkpoint_path)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        status = "★" if improved else " "
        print(f"Epoch {epoch+1:3d}{status}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f} | "
              f"Val Acc={val_acc*100:.2f}%, Best={best_val_acc*100:.2f}% | "
              f"A:{val_class_acc[0]:.0f}% H:{val_class_acc[1]:.0f}% "
              f"N:{val_class_acc[2]:.0f}% S:{val_class_acc[3]:.0f}% | "
              f"LR={current_lr:.6f} | "
              f"No imp: {epochs_without_improvement}/{config.early_stop_patience}")
        
        # Early stopping
        if epochs_without_improvement >= config.early_stop_patience:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("Final evaluation on test set...")
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    test_loss, test_acc, test_class_acc = validate(model, test_loader, criterion, device, config.num_classes)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Per-class Test Accuracy:")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo}: {test_class_acc[i]:.1f}%")
    print(f"{'='*70}")
    print(f"Best model saved to: {best_model_path}")
    
    # Register model performance
    # IMPORTANT: datasets_used is captured at training time from the actual data loaded
    try:
        from scripts.performance_registry import register_model
        register_model(
            script_name='Train_CRNN_MultiDataset.py',
            model_path=str(best_model_path.relative_to(PROJECT_ROOT)),
            metrics={
                'val_accuracy': best_val_acc,
                'test_accuracy': test_acc,
            },
            config={
                'epochs_trained': epoch + 1,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
            },
            notes=f'4-class CRNN. Test: {test_acc*100:.1f}%',
            datasets=datasets_used  # Actual datasets that were loaded
        )
    except Exception as e:
        print(f"Note: Could not register model ({e})")


if __name__ == '__main__':
    main()

