# Setting Up CGI Version (No WSGI Required)

## Files Created

1. **`index.html`** - Standalone HTML file (no Flask templates needed)
2. **`predict.cgi`** - Python CGI script that handles predictions

## Setup on Silk

### Step 1: Upload Files

**From local machine:**
```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp index.html predict.cgi wsander@w3.uvm.edu:~/www-root/
```

### Step 2: Make CGI Script Executable

**On Silk:**
```bash
cd ~/www-root
chmod +x predict.cgi
```

### Step 3: Verify Files

**On Silk:**
```bash
cd ~/www-root
ls -la index.html predict.cgi
# Both should exist
```

### Step 4: Test

Access: `https://wsander.w3.uvm.edu/`

The `index.html` should load, and when you record audio, it will call `predict.cgi`.

## How It Works

1. **`index.html`** is served as a static file (no WSGI needed)
2. When you record audio, JavaScript sends it to **`predict.cgi`**
3. **`predict.cgi`** processes the audio and returns JSON
4. JavaScript displays the results

## Troubleshooting

### If `index.html` doesn't load:

Check if there's a default `index.html`:
```bash
cd ~/www-root
ls -la index.*
```

If there's a default one, you may need to remove it or rename it.

### If predictions don't work:

**Check CGI script:**
```bash
cd ~/www-root
~/www-root/venv/bin/python3 predict.cgi
```

This should show an error (since it needs POST data), but it should at least import correctly.

**Check file permissions:**
```bash
chmod 755 predict.cgi
chmod 644 index.html
```

**Check if CGI is enabled:**
- Contact SAA if CGI scripts don't execute
- Some servers need `.htaccess` to enable CGI

### Enable CGI in .htaccess

If needed, create/update `.htaccess`:
```apache
Options +ExecCGI
AddHandler cgi-script .cgi .py
```

## Advantages of CGI Approach

- ✅ No WSGI configuration needed
- ✅ Works with static HTML hosting
- ✅ Simpler setup
- ✅ Easier to debug

## Disadvantages

- ⚠️ Slower than WSGI (each request spawns new process)
- ⚠️ Model loads on each request (can be slow)
- ⚠️ May hit timeout limits for long processing

## Next Steps

1. Upload `index.html` and `predict.cgi` to Silk
2. Make `predict.cgi` executable
3. Test at `https://wsander.w3.uvm.edu/`
4. If it doesn't work, check CGI is enabled (contact SAA if needed)

