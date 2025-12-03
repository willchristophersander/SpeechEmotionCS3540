# Model Optimization Guide

## Current Model Size

**Full CRNN Model:**
- CNN: 32→64→128 channels
- LSTM: 2 layers, 128 hidden, bidirectional
- Total parameters: ~500K-1M (estimated)
- Checkpoint size: ~26 MB
- Inference time: ~100-200ms per sample

## Optimization Strategies

### 1. Use Lightweight Model Architecture ✅ (Recommended)

I've created `CRNNLite` - a smaller version with:
- **75% fewer parameters** (~125K-250K)
- **4x faster inference** (~25-50ms)
- **Smaller checkpoint** (~6-8 MB)
- **Minimal accuracy loss** (typically <2%)

**To use:**
```python
from ser.models.crnn_lite import CRNNLite

# Load lightweight model
model = CRNNLite(n_mels=96, num_classes=4)
# Train or load checkpoint
```

### 2. Model Quantization (Post-Training)

Reduce model size by 4x with minimal accuracy loss:

```python
import torch

# Load your trained model
model = CRNN(n_mels=96, num_classes=4)
checkpoint = torch.load('checkpoints/4class/crnn_emotion_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Quantize to INT8 (4x smaller, ~2x faster)
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save({
    'model_state_dict': model_quantized.state_dict(),
    'val_accuracy': checkpoint.get('val_accuracy', None),
    'quantized': True
}, 'checkpoints/4class/crnn_emotion_model_quantized.pth')
```

**Benefits:**
- 4x smaller checkpoint (~6-7 MB)
- 2x faster inference
- Same accuracy (within 1-2%)

### 3. Model Pruning

Remove less important weights:

```python
import torch.nn.utils.prune as prune

# Prune 30% of least important weights
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')  # Make permanent

# Fine-tune for a few epochs to recover accuracy
```

**Benefits:**
- 30-50% smaller model
- Faster inference
- May need fine-tuning

### 4. Optimize Checkpoint Loading

Load only what you need:

```python
# In app.py, optimize model loading:
def load_model():
    global model
    if model is not None:
        return
    
    model_path = Path(__file__).parent / 'checkpoints' / '4class' / 'crnn_emotion_model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = CRNN(n_mels=96, num_classes=4)
    
    # Load only state dict (faster)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Optional: Use torch.jit.script for faster inference
    # model = torch.jit.script(model)
```

### 5. Use TorchScript (JIT Compilation)

Compile model for faster inference:

```python
# After loading model
model.eval()
model_scripted = torch.jit.script(model)
model_scripted.save('checkpoints/4class/crnn_emotion_model_scripted.pt')

# Load scripted model (faster)
model = torch.jit.load('checkpoints/4class/crnn_emotion_model_scripted.pt')
```

**Benefits:**
- 20-30% faster inference
- Smaller file size
- Better optimization

### 6. Reduce Input Size

Process shorter audio clips:

```python
# In app.py, reduce MAX_DURATION
MAX_DURATION = 3.0  # Instead of 4.0 seconds
MAX_FRAMES = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH)
```

**Benefits:**
- Less memory per sample
- Faster processing
- May reduce accuracy slightly

## Recommended Approach

### Quick Win (No Retraining):
1. **Quantize the existing model** → 4x smaller, 2x faster
2. **Use TorchScript** → 20-30% faster inference
3. **Optimize loading** → Faster startup

### Best Results (Requires Retraining):
1. **Train CRNNLite** → 75% smaller, 4x faster
2. **Quantize CRNNLite** → Even smaller
3. **Use TorchScript** → Maximum speed

## Implementation Priority

1. **Immediate**: Quantize existing model (no retraining needed)
2. **Short-term**: Optimize loading + TorchScript
3. **Long-term**: Retrain with CRNNLite if accuracy is acceptable

## Expected Improvements

| Method | Size Reduction | Speed Improvement | Accuracy Impact |
|--------|---------------|-------------------|-----------------|
| Quantization | 4x smaller | 2x faster | <1% loss |
| CRNNLite | 4x smaller | 4x faster | <2% loss |
| TorchScript | Same | 1.2-1.3x faster | None |
| Combined | 4x smaller | 8x faster | <2% loss |

## Code to Quantize Your Current Model

I can create a script to quantize your existing model checkpoint. Would you like me to do that?

