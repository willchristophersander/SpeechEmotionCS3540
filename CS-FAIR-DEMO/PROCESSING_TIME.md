# Audio Processing Time Estimates

## Typical Processing Times

### First Request (Cold Start)
- **Model Loading**: 10-30 seconds (one-time, only on first request or after spin-down)
- **Audio Decoding**: 0.1-0.5 seconds
- **Preprocessing** (resample, noise reduction, normalize): 0.5-2 seconds
- **Spectrogram Conversion**: 0.1-0.3 seconds
- **Model Inference**: 0.1-0.5 seconds
- **Total**: ~15-35 seconds (mostly model loading)

### Subsequent Requests (Warm - Model Already Loaded)
- **Audio Decoding**: 0.1-0.5 seconds
- **Preprocessing**: 0.5-2 seconds (noise reduction is the slowest step)
- **Spectrogram Conversion**: 0.1-0.3 seconds
- **Model Inference**: 0.1-0.5 seconds
- **Total**: **0.8-3.3 seconds** (typically 1-2 seconds for 3-5 second audio clips)

## Processing Pipeline Steps

1. **Base64 Decode** → WAV bytes (fast, ~0.1s)
2. **WAV Decode** → NumPy array (fast, ~0.1s)
3. **Resample** (if needed) → 22050 Hz (fast, ~0.1s)
4. **Noise Reduction** (noisereduce) → **SLOWEST STEP** (~0.5-1.5s)
5. **Trim Silence** (fast, ~0.05s)
6. **Volume Normalization** (fast, ~0.05s)
7. **Mel Spectrogram** conversion (fast, ~0.1-0.3s)
8. **Model Inference** (CRNN forward pass) (fast, ~0.1-0.5s)

## Factors Affecting Speed

### Audio Length
- **1-2 seconds**: ~0.5-1 second processing
- **3-5 seconds**: ~1-2 seconds processing
- **10+ seconds**: ~2-4 seconds processing

### Render Free Tier
- **After spin-down**: +30-50 seconds (waking up)
- **Warm instance**: Normal processing times above

### Noise Reduction
- The `noisereduce` library is the bottleneck
- Uses spectral gating which is computationally expensive
- Can be disabled for faster processing (but may reduce accuracy)

## Recommendations

1. **Add timeout to frontend**: 60 seconds for first request, 10 seconds for subsequent
2. **Show loading progress**: "Loading model..." vs "Processing audio..."
3. **Consider disabling noise reduction** if speed is critical (trade-off: slightly lower accuracy)
4. **Upgrade to Render Starter** ($7/month) for always-on (no spin-down delay)

## Current Status

Your frontend has **NO timeout** set, which means:
- Users might wait indefinitely if something goes wrong
- No feedback during long waits
- Should add timeout handling

