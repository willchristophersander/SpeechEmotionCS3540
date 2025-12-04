"""
Focal Loss for Speech Emotion Recognition.

Focal loss addresses class imbalance by down-weighting easy examples
and focusing on hard examples. This is particularly useful when some
emotions are easier to classify than others.

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
        p_t = predicted probability for true class
        alpha_t = class weight for true class
        gamma = focusing parameter (gamma=0 is standard cross-entropy)
    
    Args:
        alpha (list or tensor): Class weights [num_classes]
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Class indices [batch_size]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

