"""
CRNN Training with Unified Preprocessing - 4 CLASS VERSION
==========================================================
Main training script for 4-class emotion recognition.

This script orchestrates the training process:
1. Loads and prepares data
2. Creates data loaders
3. Initializes model, optimizer, and loss
4. Uses Trainer4Class for the actual training loop

All PyTorch training logic and hyperparameters are in:
- trainer_4class.py: Training loop and logic
- config_4class.py: Hyperparameters and configuration
"""

import warnings
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # For ser.* imports

from ser.models import CRNN
from ser.losses import DistanceWeightedLoss, EMOTION_DISTANCE_MATRIX
from ser.data import (
    EmotionDataset, 
    load_cremad, load_ravdess, load_savee, load_tess, 
    load_iemocap, load_emodb, load_nemo
)
from ser.data.dataset import N_MELS

# Import training modules (same directory)
# Handle both script execution and module import
try:
    from .config_4class import TrainingConfig, EMOTIONS, DEFAULT_CONFIG
    from .trainer_4class import Trainer4Class
except ImportError:
    # When running as script directly
    from config_4class import TrainingConfig, EMOTIONS, DEFAULT_CONFIG
    from trainer_4class import Trainer4Class

warnings.filterwarnings('ignore')


def map_to_4_classes(label_6class):
    """Map 6-class labels to 4-class labels.
    
    Args:
        label_6class: Label from 6-class system (0-5)
    
    Returns:
        Mapped label for 4-class system (0-3)
    
    Mapping:
        6-class: 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
        4-class: 0=Anger, 1=Happy, 2=Neutral, 3=Sad
        Fear -> Anger (similar negative emotion)
        Surprise -> Happy (similar positive emotion)
    """
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1}  # Fear->Anger, Surprise->Happy
    return mapping.get(label_6class, label_6class)


def load_datasets():
    """Load and combine all emotion datasets, mapping to 4-class labels.
    
    Returns:
        tuple: (all_files, all_labels) - Lists of file paths and 4-class labels
    """
    all_files, all_labels = [], []
    dataset_loaders = [load_cremad, load_ravdess, load_savee, load_tess, 
                       load_iemocap, load_emodb, load_nemo]
    
    for loader in dataset_loaders:
        try:
            files, labels = loader()
            labels_4class = [map_to_4_classes(l) for l in labels]
            all_files.extend(files)
            all_labels.extend(labels_4class)
        except Exception as e:
            print(f"  Error loading {loader.__name__}: {e}")
    
    return all_files, all_labels


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, config: TrainingConfig):
    """Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        X_test, y_test: Test data and labels
        config: Training configuration
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_dataset = EmotionDataset(X_train, y_train, augment=True, use_specaugment=True)
    val_dataset = EmotionDataset(X_val, y_val, augment=False, use_specaugment=False)
    test_dataset = EmotionDataset(X_test, y_test, augment=False, use_specaugment=False)
    
    # Adjust num_workers based on device availability
    num_workers = config.num_workers if torch.backends.mps.is_available() else 2
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function for 4-class emotion recognition model."""
    # Load configuration
    config = DEFAULT_CONFIG
    
    print("="*60)
    print("CRNN Training with Unified Preprocessing - 4 CLASS VERSION")
    print("="*60)
    print("IMPROVEMENTS:")
    print("  - SpecAugment: Spectrogram augmentation for better generalization")
    print("  - 96 mel bands: Richer features (vs 80)")
    print(f"  - Batch size: {config.batch_size} for stable gradients")
    print("  - Cosine annealing LR: Better convergence")
    print("  - Better checkpoint resume: Tries best model, then latest checkpoint")
    print("="*60)
    print(f"Preprocessing: noise_reduce=True, target_rms=0.1")
    print(f"Emotion classes: {', '.join(EMOTIONS)}")
    print()
    
    # ============================================================
    # 1. Load and Prepare Data
    # ============================================================
    all_files, all_labels = load_datasets()
    
    print(f"\nTotal: {len(all_files)} samples")
    
    # Class distribution
    for i, emo in enumerate(EMOTIONS):
        count = sum(1 for l in all_labels if l == i)
        print(f"  {emo}: {count}")
    
    # Split into train/val/test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 of 0.8 = 0.2 total
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # ============================================================
    # 2. Create Data Loaders
    # ============================================================
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )
    
    # ============================================================
    # 3. Initialize Model, Optimizer, and Loss
    # ============================================================
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = CRNN(n_mels=config.n_mels, num_classes=config.num_classes, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.scheduler_t0, 
        T_mult=config.scheduler_t_mult, 
        eta_min=config.scheduler_eta_min
    )
    
    # Distance-weighted loss with emotion relationships
    class_weights = config.get_class_weights_tensor(device)
    distance_matrix = EMOTION_DISTANCE_MATRIX.to(device)
    criterion = DistanceWeightedLoss(
        distance_matrix=distance_matrix,
        class_weights=class_weights,
        label_smoothing=config.label_smoothing
    )
    print(f"Class weights: Anger={config.class_weights[0]}, "
          f"Happy={config.class_weights[1]}, "
          f"Neutral={config.class_weights[2]}, "
          f"Sad={config.class_weights[3]}")
    print(f"Using distance-weighted loss with emotion relationship matrix")
    
    # ============================================================
    # 4. Initialize Trainer and Resume Training
    # ============================================================
    trainer = Trainer4Class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    resume_success = trainer.resume_from_checkpoint()
    if not resume_success:
        print("Starting training from scratch")
    
    # ============================================================
    # 5. Run Training
    # ============================================================
    trainer.train()
    
    # ============================================================
    # 6. Final Evaluation on Test Set
    # ============================================================
    print(f"\n{'='*60}")
    print("Final evaluation on test set...")
    
    test_loss, test_acc, test_class_acc = trainer.evaluate_on_test(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Per-class Test Accuracy:")
    for i, emo in enumerate(EMOTIONS):
        print(f"  {emo}: {test_class_acc[i]:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
