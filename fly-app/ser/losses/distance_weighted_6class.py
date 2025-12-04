"""Distance-weighted loss function for 6-class emotion recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Emotion relationship matrix (for distance-weighted loss) - 6 classes
# Values represent penalty multipliers: lower = more similar (less penalty), higher = more different (full penalty)
# Based on arousal-valence space and acoustic similarity
# When model predicts wrong emotion, penalty is reduced if predicted emotion is similar to true emotion
# Classes: 0=Anger, 1=Happy, 2=Neutral, 3=Sad, 4=Fear, 5=Surprise
EMOTION_DISTANCE_MATRIX_6CLASS = torch.tensor([
    # Anger, Happy, Neutral, Sad, Fear, Surprise
    [1.0, 0.9, 0.7, 0.6, 0.7, 0.8],  # Anger: similar to Sad/Fear, very different from Happy
    [0.9, 1.0, 0.6, 0.8, 0.8, 0.7],  # Happy: very different from Sad/Anger/Fear, similar to Surprise
    [0.7, 0.6, 1.0, 0.8, 0.7, 0.6],  # Neutral: somewhat similar to all
    [0.6, 0.8, 0.8, 1.0, 0.7, 0.8],  # Sad: similar to Anger/Fear (negative), different from Happy/Surprise
    [0.7, 0.8, 0.7, 0.7, 1.0, 0.8],  # Fear: similar to Anger/Sad (negative), different from Happy
    [0.8, 0.7, 0.6, 0.8, 0.8, 1.0],  # Surprise: similar to Happy/Neutral, different from negative emotions
])


class DistanceWeightedLoss(nn.Module):
    """Cross-entropy loss weighted by emotion relationships."""
    
    def __init__(self, distance_matrix, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.distance_matrix = distance_matrix  # [num_classes, num_classes]
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.num_classes = distance_matrix.shape[0]
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        Returns:
            Weighted loss
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Standard cross-entropy with class weights
        base_loss = F.cross_entropy(
            predictions, targets, 
            weight=self.class_weights, 
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # Apply distance weighting: if prediction is wrong, reduce penalty for similar emotions
        probs = F.softmax(predictions, dim=1)
        weighted_loss = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            true_label = targets[i].item()
            pred_label = predictions[i].argmax().item()
            
            if pred_label != true_label:
                # Wrong prediction: apply distance-based penalty reduction
                distance_penalty = self.distance_matrix[true_label, pred_label]
                weighted_loss[i] = base_loss[i] * distance_penalty
            else:
                # Correct prediction: full loss
                weighted_loss[i] = base_loss[i]
        
        return weighted_loss.mean()

