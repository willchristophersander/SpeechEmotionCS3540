"""
Improved CRNN model for Speech Emotion Recognition.

Enhancements over base CRNN:
1. Deeper CNN (4 layers instead of 3) with optional residual connections
2. Multi-head attention mechanism
3. Larger LSTM capacity
4. Better feature extraction

Architecture:
    Input: Mel spectrogram [batch, 1, n_mels, time_frames]
    ↓
    CNN Layers (4 conv blocks with residual): Extract hierarchical features
    ↓
    Reshape: Convert 2D feature maps to sequence for LSTM
    ↓
    Bidirectional LSTM (2 layers, 256 hidden): Model temporal dynamics
    ↓
    Multi-Head Attention: Focus on important time steps with multiple perspectives
    ↓
    Fully Connected: Final classification
    ↓
    Output: Emotion logits [batch, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock2D(nn.Module):
    """2D Residual block for CNN."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better temporal modeling."""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, time, embed_dim]
        Returns:
            context: [batch, embed_dim]
            attn_weights: [batch, time]
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        context = self.out_proj(context)
        
        # Global average pooling with attention weights
        # Use attention weights from first head for pooling
        attn_weights_global = attn_weights[:, 0, :, :].mean(dim=1)  # [batch, time]
        context = torch.sum(context * attn_weights_global.unsqueeze(-1), dim=1)  # [batch, embed_dim]
        
        return context, attn_weights_global


class CRNNImproved(nn.Module):
    """
    Improved CRNN for Speech Emotion Recognition.
    
    Args:
        n_mels (int): Number of mel frequency bands (default: 96)
        num_classes (int): Number of emotion classes (default: 4)
        dropout (float): Dropout rate (default: 0.4)
        cnn_channels (list): CNN channel progression (default: [32, 64, 128, 256])
        use_residual (bool): Use residual connections in CNN (default: True)
        lstm_hidden_size (int): LSTM hidden size (default: 256)
        lstm_num_layers (int): Number of LSTM layers (default: 2)
        attention_type (str): "single" or "multi_head" (default: "multi_head")
        attention_heads (int): Number of attention heads (default: 4)
    """
    
    def __init__(
        self, 
        n_mels=96, 
        num_classes=4, 
        dropout=0.4,
        cnn_channels=[32, 64, 128, 256],
        use_residual=True,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        attention_type="multi_head",
        attention_heads=4
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.attention_type = attention_type
        
        # ============================================================
        # CNN Feature Extraction (Deeper with Residual)
        # ============================================================
        if use_residual:
            # Use residual blocks
            self.conv1 = ResidualBlock2D(1, cnn_channels[0])
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.conv2 = ResidualBlock2D(cnn_channels[0], cnn_channels[1])
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.conv3 = ResidualBlock2D(cnn_channels[1], cnn_channels[2])
            self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.conv4 = ResidualBlock2D(cnn_channels[2], cnn_channels[3])
            self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
            
            cnn_out_freq = n_mels // 16  # 4 pooling operations
            cnn_out_channels = cnn_channels[3]
        else:
            # Standard CNN blocks (4 layers)
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, cnn_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_channels[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_channels[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(cnn_channels[2], cnn_channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_channels[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            
            cnn_out_freq = n_mels // 16
            cnn_out_channels = cnn_channels[3]
        
        # ============================================================
        # LSTM Temporal Modeling (Larger capacity)
        # ============================================================
        lstm_input_size = cnn_out_channels * cnn_out_freq
        lstm_output_size = lstm_hidden_size * 2 if lstm_num_layers > 0 else lstm_hidden_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )
        
        # ============================================================
        # Attention Mechanism
        # ============================================================
        if attention_type == "multi_head":
            self.attention = MultiHeadAttention(
                embed_dim=lstm_output_size,
                num_heads=attention_heads,
                dropout=dropout
            )
        else:
            # Single-head attention (original)
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, 128),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
        
        # ============================================================
        # Final Classification
        # ============================================================
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the improved CRNN model.
        
        Args:
            x: Input mel spectrogram [batch_size, 1, n_mels, time_frames]
        
        Returns:
            Logits for each emotion class [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        if self.use_residual:
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
        
        # Reshape for LSTM
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        x = x.contiguous().view(batch_size, x.size(1), -1)  # [batch, time, features]
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # [batch, time, lstm_output_size]
        
        # Attention mechanism
        if self.attention_type == "multi_head":
            context, _ = self.attention(lstm_out)  # [batch, lstm_output_size]
        else:
            # Single-head attention
            attn_weights = self.attention(lstm_out)  # [batch, time, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, lstm_output_size]
        
        # Final classification
        out = self.fc(context)  # [batch, num_classes]
        
        return out

