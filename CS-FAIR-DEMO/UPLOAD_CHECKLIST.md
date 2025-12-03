# Files Ready for Upload to Silk

## Quick Upload Checklist

When you get connected to Silk (via Cursor Remote SSH), here's what to upload:

### Core Files (Required)
- ✅ `app.py` - Main Flask application
- ✅ `wsgi.py` - WSGI entry point for production
- ✅ `requirements.txt` - Python dependencies
- ✅ `templates/index.html` - Web interface (updated, no emojis, citations tab)
- ✅ `checkpoints/4class/crnn_emotion_model.pth` - Model file (26MB)

### Directories (Required)
- ✅ `ser/` - Complete SER package (models, utils, data, losses, augmentation)
- ✅ `templates/` - HTML templates
- ✅ `4class_model/` - Training code (for reference)

### Optional Files
- `README.md` - Documentation
- `SETUP.md` - Setup guide
- `.htaccess` - Apache configuration (if needed)

## Upload Methods

### Method 1: Cursor Remote SSH File Explorer (Easiest)
1. Connect via Cursor Remote SSH to `uvm-silk`
2. Open remote folder: `~/www-root`
3. Drag and drop files from local `CS-FAIR-DEMO/` folder
4. Drop into remote `www-root` folder

### Method 2: SCP from Local Terminal
```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp -r app.py wsgi.py requirements.txt templates/ checkpoints/ ser/ 4class_model/ wsander@w3.uvm.edu:~/www-root/
```

### Method 3: Tar Archive (If connection is slow)
```bash
# Already created at: /tmp/silk_deploy.tar.gz (24MB)
scp /tmp/silk_deploy.tar.gz wsander@w3.uvm.edu:~/
# Then on Silk:
cd ~/www-root
tar -xzf ~/silk_deploy.tar.gz
```

## After Upload - Verify Files

On Silk, check:
```bash
cd ~/www-root
ls -la
# Should see: app.py, wsgi.py, templates/, checkpoints/, ser/, etc.

# Verify model file
ls -lh checkpoints/4class/crnn_emotion_model.pth
# Should show ~26MB
```

## Next Steps After Upload

1. Set up Python environment
2. Install dependencies
3. Test the application
4. Configure for production

