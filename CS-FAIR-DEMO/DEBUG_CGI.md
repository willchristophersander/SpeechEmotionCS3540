# Debugging CGI Script

## Issue: Stuck at "Analyzing"

If the website is stuck at "Analyzing", the CGI script might be:
1. Not executable
2. Using wrong Python interpreter
3. Hitting an error
4. Taking too long to load model

## Quick Fixes

### 1. Check CGI Script is Executable

**On Silk:**
```bash
cd ~/www-root
ls -la predict.cgi
# Should show: -rwxr-xr-x (executable)
chmod +x predict.cgi
```

### 2. Test CGI Script Directly

**On Silk:**
```bash
cd ~/www-root
~/www-root/venv/bin/python3 predict.cgi
```

This should show an error (since it needs POST data), but it should at least import correctly.

### 3. Check Server Error Logs

**On Silk:**
```bash
# Check if there are error logs
ls -la ~/logs/ 2>/dev/null
# Or check Apache error logs location (ask SAA)
```

### 4. Update Shebang in predict.cgi

The shebang should point to your venv Python. Check:

**On Silk:**
```bash
cd ~/www-root
head -1 predict.cgi
# Should show: #!/home/w/s/wsander/www-root/venv/bin/python3
# Or: #!/users/w/s/wsander/www-root/venv/bin/python3
```

If it's wrong, update it:
```bash
which python3
# Use the venv Python path
```

### 5. Test with Simple CGI Script

Create a test script to verify CGI works:

**On Silk:**
```bash
cd ~/www-root
cat > test.cgi << 'EOF'
#!/home/w/s/wsander/www-root/venv/bin/python3
print("Content-Type: application/json\n")
print('{"status": "CGI works!"}')
EOF
chmod +x test.cgi
```

Then test: `https://wsander.w3.uvm.edu/test.cgi`

### 6. Check Browser Console

In your browser, open Developer Tools (F12) and check:
- **Console tab** - for JavaScript errors
- **Network tab** - see if the request to `predict.cgi` is pending or failed

### 7. Check Model Loading

The model might be taking too long to load. Check:

**On Silk:**
```bash
cd ~/www-root
time ~/www-root/venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from predict import load_model
load_model()
print('Model loaded successfully')
"
```

## Common Issues

### Issue: "500 Internal Server Error"
- CGI script has syntax error
- Missing dependencies
- Wrong Python path in shebang

### Issue: Request Pending Forever
- Model loading is too slow
- Script is waiting for input
- Server timeout

### Issue: "Module not found"
- Using system Python instead of venv
- Missing packages in venv
- Wrong Python path

## Solution: Use Absolute Path in Shebang

Update the first line of `predict.cgi` to use the absolute path to venv Python:

**On Silk, find the correct path:**
```bash
cd ~/www-root
readlink -f venv/bin/python3
# Or
which ~/www-root/venv/bin/python3
```

Then update `predict.cgi` line 1 with that path.

