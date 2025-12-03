# How to Access Your App on Silk

## Option 1: Test Locally on Silk (Development)

If you're running the app directly for testing:

```bash
# On Silk, run:
cd ~/www-root
~/www-root/venv/bin/python3 app.py
```

The app will say it's running on `http://localhost:5001`, but you can't access this from your browser directly.

**To test from your browser**, you need to set up port forwarding:

```bash
# On your LOCAL machine, run:
ssh -L 5001:localhost:5001 wsander@w3.uvm.edu
```

Then in your browser, go to: `http://localhost:5001`

## Option 2: Access via Silk Web Server (Production)

For production access, Silk needs to serve your app via WSGI. Your app should be accessible at:

```
https://silk.uvm.edu/~wsander/
```

or

```
https://www.uvm.edu/~wsander/
```

### Steps to Set Up Web Access:

1. **Make sure your files are in the right place:**
   ```bash
   cd ~/www-root
   ls -la  # Should see app.py, wsgi.py, etc.
   ```

2. **Check if Silk needs a specific directory structure:**
   - Some Silk setups require files in `~/public_html/` instead of `~/www-root/`
   - Check with SAA or Silk documentation

3. **Configure WSGI:**
   - Silk should automatically detect `wsgi.py` in your directory
   - Or you may need to configure it via `.htaccess` (already created)

4. **Test the WSGI entry point:**
   ```bash
   cd ~/www-root
   ~/www-root/venv/bin/python3 wsgi.py
   ```

5. **Contact SAA if needed:**
   - Ask them to enable Python/WSGI for your account
   - Verify the URL structure for your account
   - Ask about any special configuration needed

## Quick Test Commands

```bash
# Check if app runs
cd ~/www-root
~/www-root/venv/bin/python3 app.py

# Test WSGI
~/www-root/venv/bin/python3 wsgi.py

# Check health endpoint (if running)
curl http://localhost:5001/health
```

## Troubleshooting

**"Can't access the URL"**
- Contact SAA to verify your account is set up for web hosting
- Check if you need to use `public_html` instead of `www-root`
- Verify Python/WSGI is enabled for your account

**"404 Not Found"**
- Make sure files are in the correct directory
- Check file permissions: `chmod 644 ~/www-root/*.py`
- Verify `.htaccess` is configured correctly

**"500 Internal Server Error"**
- Check server logs (ask SAA where logs are)
- Test the app locally first: `~/www-root/venv/bin/python3 app.py`
- Verify all dependencies are installed

## Next Steps

1. **Test locally first** using port forwarding (Option 1)
2. **Contact SAA** to verify web hosting setup
3. **Access via web URL** once configured (Option 2)

