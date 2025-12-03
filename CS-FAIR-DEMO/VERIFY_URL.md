# Verifying Your Silk URL

## Possible URLs to Try

Based on your setup, try these URLs:

1. **Primary (most likely):**
   ```
   https://www.uvm.edu/~wsander/SpeechEmotionProject/
   ```

2. **Alternative domain:**
   ```
   https://silk.uvm.edu/~wsander/SpeechEmotionProject/
   ```

3. **Root directory (if subdirectory doesn't work):**
   ```
   https://www.uvm.edu/~wsander/
   ```

## How to Verify on Silk

**On Silk, check:**

```bash
# Check what directory Silk serves from
ls -la ~/public_html/ 2>/dev/null && echo "Found public_html" || echo "No public_html"
ls -la ~/www-root/ 2>/dev/null && echo "Found www-root" || echo "No www-root"

# Check if files are in the right place
cd ~/www-root/SpeechEmotionProject
ls -la
```

## Important: Check Which Directory Silk Serves

Some Silk setups use `public_html` instead of `www-root`. Check:

```bash
# On Silk:
ls -la ~/ | grep -E "www-root|public_html"
```

If you have `public_html`, you may need to:
1. Copy files there, OR
2. Create a symlink, OR  
3. Check with SAA which directory is served

## Quick Test

**On Silk**, create a test file:

```bash
cd ~/www-root/SpeechEmotionProject
echo "Test" > test.html
```

Then try accessing:
```
https://www.uvm.edu/~wsander/SpeechEmotionProject/test.html
```

If that works, the URL structure is correct. If not, you may need to:
- Use `public_html` instead
- Check with SAA about the correct directory
- Verify WSGI is enabled

## Contact SAA

If none of the URLs work, ask SAA:
- "What is the correct URL for my web files?"
- "Which directory does Silk serve from? (www-root or public_html?)"
- "Is WSGI enabled for my account?"

