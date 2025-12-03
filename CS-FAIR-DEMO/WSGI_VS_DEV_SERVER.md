# Development Server vs Production WSGI

## What You're Seeing

When you run `~/www-root/venv/bin/python3 wsgi.py`, you're starting Flask's **development server**. This is for testing only.

The console shows:
```
Running on http://127.0.0.1:5001
Running on http://132.198.178.238:5001
```

This is the development server running locally on the Silk server - NOT the production web server.

## How Production Works

For production web access at `https://wsander.w3.uvm.edu/SpeechEmotionProject/`:

1. **Web server (Apache/Nginx)** receives the request
2. **WSGI module** loads your `wsgi.py` file
3. **Flask app** handles the request
4. **Response** is sent back through the web server

You DON'T need to run `python3 wsgi.py` manually for production.

## Steps for Production

### 1. Stop the Development Server

If you have `wsgi.py` running, press `Ctrl+C` to stop it.

### 2. Verify WSGI Configuration

**On Silk**, check if WSGI is configured:

```bash
cd ~/www-root/SpeechEmotionProject
ls -la wsgi.py
# Should exist
```

### 3. Test if Web Server Can Access It

Try accessing: `https://wsander.w3.uvm.edu/SpeechEmotionProject/`

**If you get 404:**
- WSGI may not be enabled
- Contact SAA to enable WSGI for your account

**If you get 500:**
- Check server error logs
- Verify all dependencies are installed
- Test app import: `~/www-root/venv/bin/python3 -c "from wsgi import application; print('OK')"`

**If you see directory listing:**
- WSGI isn't configured to run automatically
- Contact SAA to configure WSGI

### 4. Contact SAA

Tell them:
- "I have a Flask app with `wsgi.py` at `~/www-root/SpeechEmotionProject/`"
- "Can you enable WSGI so it's accessible at `https://wsander.w3.uvm.edu/SpeechEmotionProject/`?"
- "The app runs fine when I test it with `python3 wsgi.py`"

## Summary

- **Development server** (`python3 wsgi.py`) = For testing only, runs on port 5001
- **Production WSGI** = Web server automatically serves your app at the URL
- **You don't need to run the dev server** for production access
- **Contact SAA** to enable WSGI if the URL doesn't work

