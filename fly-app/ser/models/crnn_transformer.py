"""
CRNN-Transformer Model for Speech Emotion Recognition
=====================================================

Architecture:
1. Deep CNN Encoder (ResNet-style blocks) for spectral feature extraction.
2. Transformer Encoder for capturing long-range temporal dependencies.
3. Multi-Head Attention Pooling for final classification.

This architecture is designed to outperform standard CRNNs by:
- Using residual connections to allow deeper CNNs without vanishing gradients.
- Using Self-Attention (Transformers) which is superior to LSTMs for sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """
    Residual Block with 2 Convolutional layers.
    Input: [batch, channels, freq, time]
    Output: [batch, channels, freq, time]
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CRNN_Transformer(nn.Module):
    def __init__(self, n_mels=128, num_classes=4, dropout=0.3):
        super().__init__()
        
        # ============================================================
        # 1. Deep CNN Encoder (ResNet-style)
        # ============================================================
        # Input: [batch, 1, n_mels, time]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [B, 32, n_mels/2, time/2]
        )
        
        self.layer1 = ResidualBlock(32, 64, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2) # [B, 64, n_mels/4, time/4]
        
        self.layer2 = ResidualBlock(64, 128, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2) # [B, 128, n_mels/8, time/8]
        
        self.layer3 = ResidualBlock(128, 256, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2) # [B, 256, n_mels/16, time/16]
        
        # Calculate feature dimension for Transformer
        # For n_mels=128: 128 -> 64 -> 32 -> 16 -> 8
        cnn_out_freq = n_mels // 16
        self.transformer_dim = 256 * cnn_out_freq
        
        # Projection to standard transformer dimension (e.g., 512)
        self.project = nn.Linear(self.transformer_dim, 512)
        
        # ============================================================
        # 2. Transformer Encoder
        # ============================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(512, dropout)
        
        # ============================================================
        # 3. Classification Head (Attention Pooling)
        # ============================================================
        self.attention_pool = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [batch, 1, n_mels, time]
        
        # CNN
        x = self.conv1(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        # x: [batch, 256, freq, time]
        
        # Prepare for Transformer
        # [batch, channels, freq, time] -> [batch, time, channels, freq]
        x = x.permute(0, 3, 1, 2) 
        batch, time, channels, freq = x.size()
        
        # Flatten channels and freq: [batch, time, channels*freq]
        x = x.reshape(batch, time, channels * freq)
        
        # Project to 512 dim
        x = self.project(x) # [batch, time, 512]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        # Masking is usually not needed for fixed-length segments, 
        # but could be added if variable lengths are used.
        x = self.transformer(x) # [batch, time, 512]
        
        # Attention Pooling
        # Weights: [batch, time, 1]
        attn_weights = F.softmax(self.attention_pool(x), dim=1)
        
        # Context: [batch, 512]
        context = torch.sum(x * attn_weights, dim=1)
        
        # Classification
        out = self.fc(context)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
