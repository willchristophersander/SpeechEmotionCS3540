# Final Setup for https://www.uvm.edu/~wsander/

## ✅ Current Configuration

Your app is already configured for Silk deployment:
- ✅ `wsgi.py` exports `application` (required by WSGI)
- ✅ `app.py` loads model on import (works with WSGI)
- ✅ Routes are configured (`/` and `/predict`)
- ✅ `.htaccess` is configured
- ✅ All dependencies should be installed

## Steps to Complete Setup on Silk

### Step 1: Upload Updated Files (if needed)

From your **local machine**:

```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp app.py wsgi.py .htaccess VERIFY_SILK_SETUP.sh wsander@w3.uvm.edu:~/www-root/
```

### Step 2: Verify Setup on Silk

**On Silk**, run the verification script:

```bash
cd ~/www-root
chmod +x VERIFY_SILK_SETUP.sh
./VERIFY_SILK_SETUP.sh
```

This will check:
- All required files are present
- Model file exists
- Dependencies are installed
- WSGI can import the app

### Step 3: Test the App Locally (on Silk)

```bash
cd ~/www-root
~/www-root/venv/bin/python3 wsgi.py
```

If it starts without errors, press `Ctrl+C` and proceed.

### Step 4: Set File Permissions

```bash
cd ~/www-root
chmod 644 *.py *.txt
chmod 755 templates/
chmod 644 templates/*
chmod 644 checkpoints/4class/*.pth
```

### Step 5: Access Your App

**Try accessing:**
```
https://www.uvm.edu/~wsander/
```

## Troubleshooting

### If you get "404 Not Found"

**Most likely cause:** WSGI is not enabled for your account.

**Solution:**
1. Contact SAA (Systems Architecture & Administration)
2. Ask: "Can you enable Python/WSGI for my Silk account (wsander)?"
3. Mention: "I have a Flask app with wsgi.py ready to deploy"

### If you get "500 Internal Server Error"

**Check:**
1. Run the verification script: `./VERIFY_SILK_SETUP.sh`
2. Check if model file exists: `ls -lh checkpoints/4class/crnn_emotion_model.pth`
3. Test app import: `~/www-root/venv/bin/python3 -c "from app import app; print('OK')"`
4. Check server logs (ask SAA where they are)

### If you see a directory listing

**Cause:** WSGI is not configured to run automatically.

**Solution:**
1. Contact SAA
2. Ask them to configure WSGI for your account
3. They may need to point to your `wsgi.py` file

### If the URL works but shows errors

**Check:**
1. All dependencies installed: `~/www-root/venv/bin/pip list`
2. Model file is present: `ls -lh checkpoints/4class/crnn_emotion_model.pth`
3. Test the `/health` endpoint: `curl https://www.uvm.edu/~wsander/health`

## Expected Result

When you visit `https://www.uvm.edu/~wsander/`, you should see:
- The Speech Emotion Recognition demo page
- A button to record audio
- A button to upload audio
- Citations tab
- Footer with your names

## Quick Test Commands

```bash
# On Silk, test WSGI import
cd ~/www-root
~/www-root/venv/bin/python3 -c "from wsgi import application; print('SUCCESS')"

# Test app import
~/www-root/venv/bin/python3 -c "from app import app; print('SUCCESS')"

# Check if model loads
~/www-root/venv/bin/python3 -c "from app import load_model; load_model(); print('Model loaded!')"
```

## Next Steps

1. **Upload files** (if you made changes)
2. **Run verification script** on Silk
3. **Test locally** on Silk
4. **Try the URL**: `https://www.uvm.edu/~wsander/`
5. **Contact SAA** if you get 404 or need WSGI enabled

## Contact Information

**UVM SAA (Systems Architecture & Administration):**
- Email: Check UVM IT website
- Mention: "Need WSGI enabled for Flask app on Silk"
- Your NetID: `wsander`
- Your directory: `~/www-root/`

Your app is ready - it just needs WSGI enabled on your Silk account!

