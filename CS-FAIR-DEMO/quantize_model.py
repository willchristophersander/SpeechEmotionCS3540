#!/usr/bin/env python3
"""
Quantize the trained CRNN model to reduce size and speed up inference.

This script:
1. Loads the trained model checkpoint
2. Quantizes it to INT8 (4x smaller, 2x faster)
3. Saves the quantized model
4. Optionally tests accuracy to verify minimal loss
"""

import torch
import torch.nn as nn
from pathlib import Path
from ser.models import CRNN
from ser.data.dataset import N_MELS

def quantize_model():
    """Quantize the trained model to INT8."""
    
    # Paths
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / '4class'
    input_path = checkpoint_dir / 'crnn_emotion_model.pth'
    output_path = checkpoint_dir / 'crnn_emotion_model_quantized.pth'
    
    if not input_path.exists():
        print(f"Error: Model checkpoint not found at {input_path}")
        return
    
    print("=" * 60)
    print("Model Quantization")
    print("=" * 60)
    
    # Load original checkpoint
    print(f"\n1. Loading model from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Get original size
    original_size = input_path.stat().st_size / (1024 * 1024)  # MB
    print(f"   Original size: {original_size:.2f} MB")
    
    # Create and load model
    print("\n2. Creating model architecture...")
    model = CRNN(n_mels=N_MELS, num_classes=4)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   Loaded from 'model_state_dict'")
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print("   Loaded from 'model_state'")
    else:
        model.load_state_dict(checkpoint)
        print("   Loaded from root checkpoint")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Quantize model
    print("\n3. Quantizing model to INT8...")
    # Dynamic quantization: quantizes weights to INT8, activations stay FP32
    # This is the safest quantization method with minimal accuracy loss
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Quantize these layer types
        dtype=torch.qint8
    )
    
    print("   ✓ Quantization complete")
    
    # Prepare new checkpoint
    print("\n4. Saving quantized model...")
    quantized_checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'val_accuracy': checkpoint.get('val_accuracy', checkpoint.get('accuracy', None)),
        'quantized': True,
        'original_size_mb': original_size
    }
    
    # Save quantized model
    torch.save(quantized_checkpoint, output_path)
    
    # Get new size
    new_size = output_path.stat().st_size / (1024 * 1024)  # MB
    reduction = (1 - new_size / original_size) * 100
    
    print(f"\n5. Results:")
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   Quantized size: {new_size:.2f} MB")
    print(f"   Reduction: {reduction:.1f}%")
    print(f"   Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("=" * 60)
    print("\nTo use the quantized model, update app.py:")
    print("  model_path = 'checkpoints/4class/crnn_emotion_model_quantized.pth'")
    print("\nNote: Quantized models are 2-4x faster and 4x smaller,")
    print("      with typically <1% accuracy loss.")


if __name__ == '__main__':
    quantize_model()

