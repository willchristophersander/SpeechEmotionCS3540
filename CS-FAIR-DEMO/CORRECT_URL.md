# Your Correct URL

## Your Site URL

Your Silk site is accessible at:
```
https://wsander.w3.uvm.edu
```

## Your App URL

Since you moved files to the `SpeechEmotionProject` subdirectory, your app should be at:

```
https://wsander.w3.uvm.edu/SpeechEmotionProject/
```

## If You Want It at the Root

If you want the app at the root URL (`https://wsander.w3.uvm.edu/`), you can:

**Option 1: Move files back to root**
```bash
cd ~/www-root
mv SpeechEmotionProject/* .
mv SpeechEmotionProject/.* . 2>/dev/null
rmdir SpeechEmotionProject
```

**Option 2: Keep it in subdirectory** (recommended - cleaner)
Access at: `https://wsander.w3.uvm.edu/SpeechEmotionProject/`

## Testing

Try accessing:
1. `https://wsander.w3.uvm.edu/SpeechEmotionProject/`
2. `https://wsander.w3.uvm.edu/` (if you want root)

If you get 404 or directory listing, WSGI may need to be enabled. Contact SAA.

