# Troubleshooting Default Page Issue

## Problem

You moved files to root, but `https://wsander.w3.uvm.edu` still shows the default page instead of your Flask app.

## Possible Causes

1. **WSGI not enabled** - Most likely
2. **Default index.html taking precedence**
3. **WSGI file needs specific name/location**
4. **Web server not configured to use Python**

## Steps to Fix

### Step 1: Check What Files Are in Root

**On Silk:**
```bash
cd ~/www-root
ls -la
```

Check if there's a default `index.html` that's being served instead of your Flask app.

### Step 2: Check WSGI File

```bash
cd ~/www-root
ls -la wsgi.py
cat wsgi.py  # Verify it exists and is correct
```

### Step 3: Remove Default index.html (if exists)

If there's a default `index.html`, either:
- Remove it: `rm index.html`
- Or rename it: `mv index.html index.html.bak`

### Step 4: Check if Silk Needs Different WSGI Setup

Some Silk setups require:
- File named `application.py` instead of `wsgi.py`
- File in a specific location
- Different configuration

### Step 5: Contact SAA

You'll likely need to contact SAA to:
1. Enable WSGI/Python for your account
2. Configure it to use your `wsgi.py` file
3. Verify the correct setup for your account

Tell them:
- "I have a Flask app with `wsgi.py` in `~/www-root/`"
- "The URL `https://wsander.w3.uvm.edu` shows the default page instead of my app"
- "Can you enable WSGI and configure it to serve my Flask app?"

## Quick Test

**On Silk:**
```bash
cd ~/www-root
# Create a test file
echo "<?php phpinfo(); ?>" > test.php
# Or
echo "<h1>Test</h1>" > test.html
```

Then try: `https://wsander.w3.uvm.edu/test.html` or `https://wsander.w3.uvm.edu/test.php`

If those work, the web server is working but WSGI isn't configured yet.

