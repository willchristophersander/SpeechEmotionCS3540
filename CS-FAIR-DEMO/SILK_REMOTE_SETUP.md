# Setting Up on Silk via Cursor Remote SSH

Since you're connected to Silk via Cursor Remote SSH, follow these steps:

## Step 1: Navigate to Your Web Directory

In the **remote terminal** (in Cursor, connected to Silk):

```bash
# Check where you are
pwd

# Navigate to your home directory
cd ~

# Check if www-root exists
ls -la | grep www-root

# If it doesn't exist, create it
mkdir -p www-root

# Navigate into it
cd www-root
```

## Step 2: Upload Files from Local Machine

You have two options:

### Option A: Using Cursor's File Explorer (Easiest)

1. In Cursor, you should see the remote file system in the left sidebar
2. Navigate to `~/www-root` in the remote file explorer
3. From your **local** file explorer (separate window), drag and drop files from:
   - `/Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO/`
4. Drop them into the remote `www-root` folder

### Option B: Using SCP from Local Terminal

Open a **new local terminal** (not the remote one) and run:

```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO
scp -r * wsander@w3.uvm.edu:~/www-root/
```

Or use the tar file I created:
```bash
scp /tmp/silk_deploy.tar.gz wsander@w3.uvm.edu:~/
# Then on Silk, extract it:
# tar -xzf silk_deploy.tar.gz -C www-root/
```

## Step 3: Verify Files Are Uploaded

In the **remote terminal** (in Cursor):

```bash
cd ~/www-root
ls -la
# You should see: app.py, templates/, checkpoints/, ser/, etc.
```

## Step 4: Set Up Python Environment

In the **remote terminal**:

```bash
cd ~/www-root

# Check Python version
python3 --version

# Create virtual environment (recommended)
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: PyTorch is large. If you get space errors:
```bash
# Try CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Step 5: Test the Application

In the **remote terminal**:

```bash
cd ~/www-root
source venv/bin/activate  # If using venv
python3 app.py
```

Or test WSGI:
```bash
pip install gunicorn
gunicorn -w 1 -b 127.0.0.1:8000 --timeout 120 wsgi:app
```

## Step 6: Configure Silk Web Server

Contact SAA or check Silk documentation for:
- How to configure the WSGI entry point
- What port/host to use
- Process management (may need to set up a service)

## Quick Commands Reference

```bash
# Navigate to web directory
cd ~/www-root

# Check files
ls -la

# Check Python
python3 --version
which python3

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Test app
python3 app.py
```

## File Structure on Silk

After upload, your `~/www-root/` should contain:
```
www-root/
├── app.py
├── wsgi.py
├── requirements.txt
├── templates/
│   └── index.html
├── checkpoints/
│   └── 4class/
│       └── crnn_emotion_model.pth
├── ser/
│   ├── models/
│   ├── utils/
│   ├── data/
│   └── ...
└── 4class_model/
```

## Troubleshooting

**"Permission denied" on www-root**
```bash
chmod 755 ~/www-root
chmod -R 644 ~/www-root/*
chmod -R 755 ~/www-root/*/
```

**"Module not found" errors**
- Make sure you're in the `www-root` directory
- Check that `ser/` directory is present
- Verify virtual environment is activated

**"Model file not found"**
```bash
ls -lh ~/www-root/checkpoints/4class/crnn_emotion_model.pth
# Should show ~26MB file
```

