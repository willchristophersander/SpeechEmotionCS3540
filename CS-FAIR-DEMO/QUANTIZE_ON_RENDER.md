# Quantizing Model on Render

Since quantization requires PyTorch, you can run it on Render after deployment.

## Option 1: Run Quantization Script on Render (Recommended)

1. **SSH into your Render service** (if available) or use Render Shell
2. **Run the quantization script:**
   ```bash
   cd /opt/render/project/src/CS-FAIR-DEMO
   python quantize_model.py
   ```
3. **The quantized model will be saved** and app.py will automatically use it on next restart

## Option 2: Quantize Locally (If You Have PyTorch)

If you have PyTorch installed locally:
```bash
cd CS-FAIR-DEMO
python quantize_model.py
```

Then commit and push the quantized model.

## Option 3: Add to Build Process

Add to `render.yaml` build command:
```yaml
buildCommand: pip install -r requirements.txt && python quantize_model.py || echo "Quantization skipped"
```

This will quantize during build if possible, but won't fail if it can't.

## Current Status

The app is now configured to:
- ✅ **Try quantized model first** (if it exists)
- ✅ **Fall back to regular model** (if quantized doesn't exist)
- ✅ **Work with both** seamlessly

## Next Steps

1. Deploy current code (supports both models)
2. Run quantization on Render or locally
3. Restart service to use quantized model
4. Enjoy 4x smaller, 2x faster model!

