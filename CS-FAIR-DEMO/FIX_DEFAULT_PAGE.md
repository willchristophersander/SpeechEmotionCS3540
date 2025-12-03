# Fixing Default Page Issue

## What I See

Your files are all in place:
- ✅ `app.py` exists
- ✅ `wsgi.py` exists  
- ✅ `templates/` exists
- ✅ `ser/` exists
- ✅ `checkpoints/` exists

But the site still shows the default page.

## Possible Issues

1. **No index.html** - Web server might be looking for one
2. **WSGI not enabled** - Most likely cause
3. **Default page being served** - Could be `test_cases.html` or a server default

## Steps to Fix

### Step 1: Check for index.html

**On Silk:**
```bash
cd ~/www-root
ls -la index.*
```

If there's no `index.html`, the web server might be serving a default or directory listing.

### Step 2: Check what's being served

Try accessing:
- `https://wsander.w3.uvm.edu/test.html` (you created this)
- `https://wsander.w3.uvm.edu/test_cases.html` (this exists)

If those work, the web server is working but WSGI isn't configured.

### Step 3: Create a simple index.html (temporary test)

**On Silk:**
```bash
cd ~/www-root
echo "<h1>Flask App Test</h1><p>If you see this, web server is working but WSGI needs configuration.</p>" > index.html
```

Then try: `https://wsander.w3.uvm.edu/`

If you see your test message, the web server works but WSGI isn't enabled.

### Step 4: Contact SAA (Required)

You need to contact SAA to enable WSGI. Tell them:

- "I have a Flask app with `wsgi.py` in `~/www-root/`"
- "The URL `https://wsander.w3.uvm.edu` shows the default page"
- "Can you enable WSGI and configure it to serve my Flask app?"
- "My `wsgi.py` exports an `application` object as required"

### Step 5: Alternative - Check if Silk needs different setup

Some Silk setups require:
- File named `application.py` instead of `wsgi.py`
- Specific directory structure
- Different configuration

Ask SAA about the correct setup for Flask apps.

## Quick Test Commands

**On Silk:**
```bash
cd ~/www-root

# Test if wsgi.py can be imported
~/www-root/venv/bin/python3 -c "from wsgi import application; print('WSGI import: OK')"

# Check file permissions
ls -la wsgi.py app.py

# Check if .htaccess exists
ls -la .htaccess
```

## Most Likely Solution

**Contact SAA** - WSGI needs to be enabled at the server level. Your files are correct, but the web server doesn't know to use WSGI yet.

