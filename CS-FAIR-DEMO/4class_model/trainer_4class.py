"""
PyTorch Training Module for 4-Class Emotion Recognition
========================================================

This module contains the core PyTorch training logic, including:
- Training loop
- Validation loop
- Checkpoint management
- Early stopping
- Learning rate scheduling
"""

import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional

from .config_4class import TrainingConfig, EMOTIONS


class Trainer4Class:
    """Trainer class for 4-class emotion recognition model."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: TrainingConfig
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Setup checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.config.get_best_model_path()
    
    def resume_from_checkpoint(self) -> bool:
        """Attempt to resume training from existing checkpoint.
        
        Returns:
            True if successfully resumed, False otherwise
        """
        # Try loading best model first
        if self.best_model_path.exists():
            try:
                checkpoint = torch.load(self.best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                if 'scheduler_state' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                self.best_val_acc = checkpoint.get('val_accuracy', 0)
                self.best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"✓ Resuming from best model (epoch {checkpoint.get('epoch', 0)}), "
                      f"best val acc: {self.best_val_acc*100:.2f}%")
                return True
            except Exception as e:
                print(f"Could not resume from best model: {e}")
        
        # Try loading latest periodic checkpoint
        checkpoint_files = sorted(
            self.config.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda x: int(x.stem.split('_')[-1]),
            reverse=True
        )
        if checkpoint_files:
            try:
                latest_checkpoint = checkpoint_files[0]
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                if 'scheduler_state' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                self.best_val_acc = checkpoint.get('val_accuracy', 0)
                self.best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"✓ Resuming from checkpoint {latest_checkpoint.name} "
                      f"(epoch {checkpoint.get('epoch', 0)}), val acc: {self.best_val_acc*100:.2f}%")
                return True
            except Exception as e:
                print(f"Could not resume from checkpoint: {e}")
        
        return False
    
    def save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, 
                       train_loss: float, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_acc: Validation accuracy
            val_loss: Validation loss
            train_loss: Training loss
            is_best: Whether this is the best model so far
        """
        checkpoint_data = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'emotions': EMOTIONS
        }
        
        if is_best:
            torch.save(checkpoint_data, self.best_model_path)
        else:
            # Periodic checkpoint
            checkpoint_path = self.config.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint_data, checkpoint_path)
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        train_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.start_epoch+1:2d}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            train_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'lr': f'{current_lr:.6f}'
            })
        
        return train_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float, list]:
        """Validate the model.
        
        Returns:
            Tuple of (avg_loss, accuracy, class_accuracies)
        """
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        class_correct = [0] * self.config.num_classes
        class_total = [0] * self.config.num_classes
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                for i in range(len(y_batch)):
                    label = y_batch[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        class_accuracies = [
            class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
            for i in range(self.config.num_classes)
        ]
        
        return avg_loss, accuracy, class_accuracies
    
    def update_learning_rate(self, epoch: int):
        """Update learning rate based on warmup and scheduler.
        
        Args:
            epoch: Current epoch number
        """
        if epoch < self.config.warmup_epochs and self.start_epoch == 0:
            # Learning rate warmup
            warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        elif epoch >= self.config.warmup_epochs:
            # Use scheduler after warmup
            self.scheduler.step()
    
    def train(self):
        """Main training loop."""
        print(f"\nTraining for up to {self.config.max_epochs} epochs "
              f"(early stop patience: {self.config.early_stop_patience})")
        print(f"Learning rate warmup: {self.config.warmup_epochs} epochs")
        print(f"Checkpoint interval: every {self.config.checkpoint_interval} epochs")
        print()
        
        for epoch in range(self.start_epoch, self.config.max_epochs):
            # Update learning rate
            self.update_learning_rate(epoch)
            
            # Train for one epoch
            avg_train_loss = self.train_epoch()
            
            # Validate
            avg_val_loss, val_acc, val_class_acc = self.validate()
            
            # Check for improvement
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                improved = True
                
                # Save best model
                self.save_checkpoint(
                    epoch, val_acc, avg_val_loss, avg_train_loss, is_best=True
                )
            else:
                self.epochs_without_improvement += 1
            
            # Periodic checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(
                    epoch, val_acc, avg_val_loss, avg_train_loss, is_best=False
                )
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            status = "★" if improved else " "
            print(f"Epoch {epoch+1:3d}{status}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f} | "
                  f"Val Acc={val_acc*100:.2f}%, Best={self.best_val_acc*100:.2f}% | "
                  f"A:{val_class_acc[0]:.0f}% H:{val_class_acc[1]:.0f}% "
                  f"N:{val_class_acc[2]:.0f}% S:{val_class_acc[3]:.0f}% | "
                  f"LR={current_lr:.6f} | "
                  f"No improvement: {self.epochs_without_improvement}/{self.config.early_stop_patience}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs "
                      f"(no improvement for {self.config.early_stop_patience} epochs)")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Best model saved to {self.best_model_path}")
    
    def evaluate_on_test(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float, list]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Tuple of (avg_loss, accuracy, class_accuracies)
        """
        # Load best model
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        class_correct = [0] * self.config.num_classes
        class_total = [0] * self.config.num_classes
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                for i in range(len(y_batch)):
                    label = y_batch[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        class_accuracies = [
            class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
            for i in range(self.config.num_classes)
        ]
        
        return avg_loss, accuracy, class_accuracies

