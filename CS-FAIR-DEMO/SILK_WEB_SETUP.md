# Setting Up App for https://www.uvm.edu/~wsander/

## Current Configuration Status

✅ **WSGI Entry Point**: `wsgi.py` is configured correctly
✅ **Model Loading**: App loads model on import (works with WSGI)
✅ **Flask App**: Configured with CORS and proper routes
✅ **Static Files**: Templates directory is in place

## Steps to Deploy on Silk

### 1. Verify File Structure on Silk

On Silk, make sure your files are in `~/www-root/`:

```bash
cd ~/www-root
ls -la
# Should see: app.py, wsgi.py, templates/, checkpoints/, ser/, etc.
```

### 2. Verify WSGI Configuration

The `wsgi.py` file exports `application` which is what Silk's WSGI server needs:

```python
# wsgi.py exports:
application = app
```

### 3. Test WSGI Locally (on Silk)

```bash
cd ~/www-root
~/www-root/venv/bin/python3 wsgi.py
# Should start the app (for testing)
# Press Ctrl+C to stop
```

### 4. Check File Permissions

```bash
cd ~/www-root
chmod 644 *.py
chmod 644 *.txt
chmod 755 templates/
chmod 644 templates/*
chmod 644 checkpoints/4class/*.pth
```

### 5. Verify Dependencies Are Installed

```bash
cd ~/www-root
~/www-root/venv/bin/pip list | grep -E "flask|torch|librosa|numpy"
```

### 6. Test the App

**Option A: Test via WSGI directly:**
```bash
cd ~/www-root
~/www-root/venv/bin/python3 wsgi.py
# Then use port forwarding from local machine:
# ssh -L 5001:localhost:5001 wsander@w3.uvm.edu
# Access at http://localhost:5001
```

**Option B: Access via Web URL:**
```
https://www.uvm.edu/~wsander/
```

## Troubleshooting

### If URL shows "404 Not Found"

1. **Check directory name:**
   - Silk might use `public_html` instead of `www-root`
   - Try: `ls ~/public_html/` or check with SAA

2. **Check if Python/WSGI is enabled:**
   - Contact SAA to enable Python/WSGI for your account
   - Some accounts need explicit activation

3. **Verify file location:**
   ```bash
   pwd  # Should be in ~/www-root or ~/public_html
   ls -la wsgi.py  # Should exist
   ```

### If URL shows "500 Internal Server Error"

1. **Check server logs:**
   - Ask SAA where error logs are located
   - Usually in `~/logs/` or similar

2. **Test app locally first:**
   ```bash
   cd ~/www-root
   ~/www-root/venv/bin/python3 app.py
   # Should start without errors
   ```

3. **Check model file:**
   ```bash
   ls -lh ~/www-root/checkpoints/4class/crnn_emotion_model.pth
   # Should show ~26MB file
   ```

4. **Verify all dependencies:**
   ```bash
   ~/www-root/venv/bin/python3 -c "import flask, torch, librosa; print('OK')"
   ```

### If URL shows directory listing

- Silk might not be configured to use WSGI automatically
- Contact SAA to configure WSGI for your account
- May need to create `application.py` instead of `wsgi.py`

## Next Steps

1. **Upload updated files** (if you made changes):
   ```bash
   # From local machine:
   scp app.py wsgi.py .htaccess wsander@w3.uvm.edu:~/www-root/
   ```

2. **Test locally on Silk:**
   ```bash
   # On Silk:
   cd ~/www-root
   ~/www-root/venv/bin/python3 wsgi.py
   ```

3. **Access via web:**
   - Try: `https://www.uvm.edu/~wsander/`
   - If it doesn't work, contact SAA

4. **Contact SAA if needed:**
   - Ask: "Is Python/WSGI enabled for my account?"
   - Ask: "What is the correct directory for web files?"
   - Ask: "How do I configure WSGI for my Flask app?"

## Expected Behavior

When you visit `https://www.uvm.edu/~wsander/`:
- Should see the Speech Emotion Recognition demo page
- Should be able to record/upload audio
- Should get emotion predictions

If you see a directory listing or 404, the WSGI isn't configured yet - contact SAA.

