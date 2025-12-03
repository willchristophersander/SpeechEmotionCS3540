# Fix: CGI Script Not Executing

## Problem
The browser is receiving the shebang line `#!/users/w/s/wsander/www-root/venv/bin/python3` as part of the response. This means the CGI script isn't being executed - it's being served as a plain text file.

## Solutions

### 1. Check File Permissions

**On Silk:**
```bash
cd ~/www-root
ls -la predict.cgi
# Should show: -rwxr-xr-x (executable)

# If not executable:
chmod 755 predict.cgi
```

### 2. Check if CGI is Enabled

**On Silk:**
```bash
cd ~/www-root
ls -la .htaccess
cat .htaccess
```

If `.htaccess` doesn't exist or doesn't enable CGI, create/update it:

```bash
cat > .htaccess << 'EOF'
Options +ExecCGI
AddHandler cgi-script .cgi .py
EOF
```

### 3. Try Different File Extension

Some servers need `.py` instead of `.cgi`:

```bash
cd ~/www-root
cp predict.cgi predict.py
chmod +x predict.py
```

Then update `index.html` to call `predict.py` instead of `predict.cgi`.

### 4. Check if Server Supports CGI

**On Silk:**
```bash
cd ~/www-root
# Create a simple test
cat > test.cgi << 'EOF'
#!/users/w/s/wsander/www-root/venv/bin/python3
print("Content-Type: text/plain\n")
print("CGI is working!")
EOF
chmod +x test.cgi
```

Then test: `https://wsander.w3.uvm.edu/test.cgi`

If this also shows the shebang, CGI isn't enabled.

### 5. Contact SAA

If none of the above works, contact SAA and ask:
- "Is CGI enabled for my account?"
- "How do I enable CGI scripts on Silk?"
- "My CGI script isn't executing - it's being served as text"

## Quick Fix to Try First

**On Silk:**
```bash
cd ~/www-root
chmod 755 predict.cgi
chmod 644 .htaccess  # If it exists
# Or create .htaccess:
echo "Options +ExecCGI" > .htaccess
echo "AddHandler cgi-script .cgi .py" >> .htaccess
```

Then test again.

