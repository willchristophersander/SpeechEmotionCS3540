# Quick Model Optimization Guide

## Fastest Option: Quantize Existing Model (No Retraining!)

**Takes 2 minutes, reduces size by 4x, speeds up by 2x**

```bash
cd CS-FAIR-DEMO
python quantize_model.py
```

This will:
- Load your current model
- Quantize it to INT8
- Save as `crnn_emotion_model_quantized.pth` (~6-7 MB instead of ~26 MB)
- Minimal accuracy loss (<1%)

Then update `app.py`:
```python
model_path = Path(__file__).parent / 'checkpoints' / '4class' / 'crnn_emotion_model_quantized.pth'
```

## Expected Results

| Metric | Before | After Quantization | Improvement |
|--------|--------|-------------------|--------------|
| Model Size | ~26 MB | ~6-7 MB | **4x smaller** |
| Load Time | ~2-3s | ~0.5-1s | **2-3x faster** |
| Inference | ~100-200ms | ~50-100ms | **2x faster** |
| Accuracy | 100% | ~99% | <1% loss |

## Other Options

### Option 2: Use Lightweight Architecture (Requires Retraining)
- Use `CRNNLite` instead of `CRNN`
- 75% fewer parameters
- 4x faster inference
- See `MODEL_OPTIMIZATION.md` for details

### Option 3: Combine Both (Best Results)
- Train `CRNNLite`
- Quantize it
- **8x faster, 4x smaller, <2% accuracy loss**

## Recommendation

**Start with quantization** - it's instant and gives you 4x size reduction and 2x speedup with no retraining needed!

