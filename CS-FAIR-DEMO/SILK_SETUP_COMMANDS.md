# Silk Setup Commands - Run These Now

## Step 1: Navigate to Web Directory

```bash
cd ~
mkdir -p www-root
cd www-root
pwd
```

## Step 2: Upload Files

**Option A: If you can drag & drop in Cursor Remote SSH:**
- Drag all files from local `CS-FAIR-DEMO/` folder
- Drop into remote `~/www-root/` folder

**Option B: From local terminal (new window), run:**
```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp -r app.py wsgi.py requirements.txt templates/ checkpoints/ ser/ 4class_model/ wsander@w3.uvm.edu:~/www-root/
```

**Option C: Using the tar file:**
```bash
# From local terminal:
scp /tmp/silk_deploy.tar.gz wsander@w3.uvm.edu:~/

# Then on Silk:
cd ~/www-root
tar -xzf ~/silk_deploy.tar.gz
```

## Step 3: Verify Files Are Uploaded

```bash
cd ~/www-root
ls -la
ls -lh checkpoints/4class/crnn_emotion_model.pth
```

## Step 4: Check Python Version

```bash
python3 --version
which python3
```

## Step 5: Create Virtual Environment

```bash
cd ~/www-root
python3 -m venv venv
source venv/bin/activate
```

## Step 6: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If PyTorch is too large, try CPU-only version:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Step 7: Test the Application

```bash
cd ~/www-root
source venv/bin/activate  # If not already activated
python3 app.py
```

Or test with Gunicorn (production WSGI server):
```bash
pip install gunicorn
gunicorn -w 1 -b 127.0.0.1:8000 --timeout 120 wsgi:app
```

## Step 8: Check File Permissions

```bash
chmod 755 ~/www-root
chmod 644 ~/www-root/*.py
chmod 644 ~/www-root/*.txt
chmod -R 755 ~/www-root/templates/
chmod -R 644 ~/www-root/checkpoints/
```

## Quick Reference - All Commands in Order

```bash
# 1. Navigate
cd ~/www-root

# 2. Verify files
ls -la

# 3. Python environment
python3 -m venv venv
source venv/bin/activate

# 4. Install
pip install --upgrade pip
pip install -r requirements.txt

# 5. Test
python3 app.py
```

