"""
CRNN (Convolutional Recurrent Neural Network) model for Speech Emotion Recognition.

Architecture Overview:
    Input: Mel spectrogram [batch, 1, n_mels, time_frames]
    ↓
    CNN Layers (3 conv blocks): Extract local spectral-temporal features
    ↓
    Reshape: Convert 2D feature maps to sequence for LSTM
    ↓
    Bidirectional LSTM (2 layers): Model temporal dynamics in both directions
    ↓
    Attention Mechanism: Weight important time steps
    ↓
    Fully Connected: Final classification
    ↓
    Output: Emotion logits [batch, num_classes]

The model combines:
    - CNN: Learns local patterns (spectral shapes, formants, energy contours)
    - LSTM: Captures temporal dependencies (pitch trajectories, energy dynamics)
    - Attention: Focuses on emotion-relevant segments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    CRNN for Speech Emotion Recognition.
    
    Args:
        n_mels (int): Number of mel frequency bands (default: 80, current: 96)
        num_classes (int): Number of emotion classes (default: 4)
        dropout (float): Dropout rate for regularization (default: 0.3)
    
    Input Shape:
        [batch_size, 1, n_mels, time_frames]
        Example: [40, 1, 96, 173] for batch_size=40, 96 mels, 4 seconds @ 22050Hz
    
    Output Shape:
        [batch_size, num_classes]
        Example: [40, 4] for 4 emotions
    """
    
    def __init__(self, n_mels=80, num_classes=4, dropout=0.3):
        super().__init__()
        
        # ============================================================
        # CNN Feature Extraction Layers
        # ============================================================
        # Purpose: Extract local spectral-temporal patterns from mel spectrogram
        # Each conv block: Conv → BatchNorm → ReLU → MaxPool
        
        # Conv Block 1: Extract basic spectral features
        # Input: [batch, 1, n_mels, time] → Output: [batch, 32, n_mels/2, time/2]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1→32 channels, 3×3 conv
            nn.BatchNorm2d(32),                          # Normalize activations
            nn.ReLU(),                                    # Non-linearity
            nn.MaxPool2d(kernel_size=(2, 2))             # Downsample by 2×2
        )
        
        # Conv Block 2: Extract mid-level patterns
        # Input: [batch, 32, n_mels/2, time/2] → Output: [batch, 64, n_mels/4, time/4]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32→64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Conv Block 3: Extract high-level features
        # Input: [batch, 64, n_mels/4, time/4] → Output: [batch, 128, n_mels/8, time/8]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64→128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Calculate CNN output frequency dimension after 3 pooling operations
        # Each MaxPool reduces by factor of 2, so: n_mels → n_mels/2 → n_mels/4 → n_mels/8
        # For 96 mels: 96 → 48 → 24 → 12
        cnn_out_freq = n_mels // 8
        
        # ============================================================
        # LSTM Temporal Modeling
        # ============================================================
        # Purpose: Model temporal dynamics and dependencies across time
        # Bidirectional: Processes sequence forward and backward
        # Input size: 128 channels × cnn_out_freq frequency bins = feature vector per time step
        # Output: [batch, time, 256] (128 forward + 128 backward)
        self.lstm = nn.LSTM(
            input_size=128 * cnn_out_freq,  # Feature vector size per time step
            hidden_size=128,                  # Hidden state dimension
            num_layers=2,                     # Stacked LSTM layers for deeper modeling
            batch_first=True,                 # Input format: [batch, time, features]
            bidirectional=True,               # Process both forward and backward
            dropout=dropout                   # Dropout between LSTM layers
        )
        
        # ============================================================
        # Attention Mechanism
        # ============================================================
        # Purpose: Learn which time steps are most important for emotion classification
        # Computes attention weights for each time step, then weighted sum of LSTM outputs
        # Input: [batch, time, 256] → Output: [batch, time, 1] (attention weights)
        self.attention = nn.Sequential(
            nn.Linear(256, 64),   # Project LSTM output to attention space
            nn.Tanh(),            # Non-linearity (bounded activation)
            nn.Linear(64, 1)      # Single attention score per time step
        )
        
        # ============================================================
        # Final Classification Layer
        # ============================================================
        # Purpose: Map attention-weighted context to emotion classes
        # Input: [batch, 256] (weighted sum of LSTM outputs) → Output: [batch, num_classes]
        self.fc = nn.Sequential(
            nn.Linear(256, 64),      # Reduce dimensionality
            nn.ReLU(),               # Non-linearity
            nn.Dropout(dropout),     # Regularization
            nn.Linear(64, num_classes)  # Final classification layer
        )
    
    def forward(self, x):
        """
        Forward pass through the CRNN model.
        
        Args:
            x: Input mel spectrogram [batch_size, 1, n_mels, time_frames]
        
        Returns:
            Logits for each emotion class [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # ============================================================
        # Stage 1: CNN Feature Extraction
        # ============================================================
        # Extract hierarchical features from spectrogram
        # Shape progression: [B, 1, 96, 173] → [B, 32, 48, 86] → [B, 64, 24, 43] → [B, 128, 12, 21]
        x = self.conv1(x)  # Extract low-level features (spectral shapes)
        x = self.conv2(x)  # Extract mid-level features (formants, energy patterns)
        x = self.conv3(x)  # Extract high-level features (emotion-specific patterns)
        
        # ============================================================
        # Stage 2: Reshape for LSTM
        # ============================================================
        # Convert 2D feature maps to sequence format for LSTM
        # Current shape: [batch, 128, 12, 21] (channels, freq, time)
        # Target shape: [batch, 21, 128*12] (time steps, feature vector)
        
        # Permute: [batch, channels, freq, time] → [batch, time, channels, freq]
        x = x.permute(0, 3, 1, 2)
        # Reshape: [batch, time, channels, freq] → [batch, time, channels*freq]
        # This creates a sequence where each time step has a feature vector
        x = x.contiguous().view(batch_size, x.size(1), -1)
        # Now: [batch, 21, 1536] where 1536 = 128 channels × 12 frequency bins
        
        # ============================================================
        # Stage 3: LSTM Temporal Modeling
        # ============================================================
        # Process sequence to capture temporal dynamics
        # Input: [batch, time, 1536] → Output: [batch, time, 256]
        # 256 = 128 (forward) + 128 (backward) from bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch, 21, 256]
        
        # ============================================================
        # Stage 4: Attention Mechanism
        # ============================================================
        # Compute attention weights to focus on important time steps
        # Input: [batch, time, 256] → Output: [batch, time, 1]
        attn_weights = self.attention(lstm_out)
        # Normalize to probability distribution over time steps
        attn_weights = F.softmax(attn_weights, dim=1)
        # Weighted sum: Combine LSTM outputs based on attention weights
        # [batch, time, 256] × [batch, time, 1] → [batch, 256]
        context = torch.sum(lstm_out * attn_weights, dim=1)
        # Context shape: [batch, 256] - single vector representing entire sequence
        
        # ============================================================
        # Stage 5: Final Classification
        # ============================================================
        # Map attention-weighted context to emotion probabilities
        # Input: [batch, 256] → Output: [batch, num_classes]
        out = self.fc(context)
        
        return out

