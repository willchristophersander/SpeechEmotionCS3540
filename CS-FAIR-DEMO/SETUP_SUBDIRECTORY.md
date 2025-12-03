# Setting Up App at https://www.uvm.edu/~wsander/SpeechEmotionProject/

## Option 1: Create Subdirectory on Silk (Recommended)

**On Silk**, create a subdirectory and move files:

```bash
cd ~/www-root
mkdir -p SpeechEmotionProject
# Move all files to the subdirectory
mv app.py wsgi.py requirements.txt templates/ checkpoints/ ser/ 4class_model/ *.md *.sh .htaccess SpeechEmotionProject/ 2>/dev/null
# Or copy if you want to keep originals
# cp -r app.py wsgi.py requirements.txt templates/ checkpoints/ ser/ 4class_model/ *.md *.sh .htaccess SpeechEmotionProject/
```

Then access at: `https://www.uvm.edu/~wsander/SpeechEmotionProject/`

## Option 2: Use Symbolic Link

**On Silk**:

```bash
cd ~/www-root
ln -s . SpeechEmotionProject
```

Then access at: `https://www.uvm.edu/~wsander/SpeechEmotionProject/`

## Option 3: Configure Flask for Subdirectory

If you want the app to work at a subdirectory path, we can configure Flask with a `SCRIPT_NAME`. But for Silk, Option 1 is simplest.

## Recommended: Use Subdirectory

The cleanest approach is to create a `SpeechEmotionProject` subdirectory in `www-root` and put all your files there.

