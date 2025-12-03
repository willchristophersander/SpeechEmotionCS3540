# Quick Deployment Guide for Silk (wsander)

## Your Silk Account Info
- **Username**: wsander
- **Host**: w3.uvm.edu
- **Directory**: ~/www-root/

## Quick Deploy Steps

### 1. Upload Files to Silk

From the CS-FAIR-DEMO directory:
```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp -r * wsander@w3.uvm.edu:~/www-root/
```

Or use the deployment script:
```bash
./deploy_to_silk.sh wsander
```

### 2. SSH into Silk

```bash
ssh wsander@w3.uvm.edu
```

### 3. Navigate to Your Site

```bash
cd ~/www-root
ls -la  # Verify files are there
```

### 4. Set Up Python Environment

Check Python version:
```bash
python3 --version
```

Create virtual environment (if needed):
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: PyTorch is large (~500MB). If installation fails due to space, you may need to:
- Use CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Or contact SAA about disk space limits

### 6. Test Locally on Silk

```bash
python3 app.py
```

Or test WSGI:
```bash
pip install gunicorn
gunicorn -w 1 -b 127.0.0.1:8000 --timeout 120 wsgi:app
```

### 7. Configure Silk Web Server

Contact SAA or check Silk documentation for:
- How to configure WSGI entry point (likely `wsgi.py`)
- Port configuration
- Process management

## Files Ready for Deployment

✅ `app.py` - Updated for WSGI compatibility
✅ `wsgi.py` - WSGI entry point created
✅ `templates/index.html` - Professional design, no emojis, citations tab
✅ `checkpoints/4class/crnn_emotion_model.pth` - Model file (26MB)
✅ `requirements.txt` - All dependencies listed
✅ `ser/` - Complete package
✅ Footer with William Sander and Seth Shienbrood

## Troubleshooting

- **SSH issues**: Make sure you have SSH keys set up or use password authentication
- **Permission denied**: Check file permissions on Silk: `chmod 755 ~/www-root`
- **Import errors**: Make sure you're in the correct directory and Python path is set
- **Model not loading**: Verify `checkpoints/4class/crnn_emotion_model.pth` exists and is readable

## Contact

If you need help with Silk configuration, contact SAA (Systems Architecture & Administration) at UVM.

