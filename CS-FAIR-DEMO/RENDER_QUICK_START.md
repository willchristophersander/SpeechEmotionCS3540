# Quick Start: Deploy to Render.com

## ✅ Yes, Render Supports Private GitHub Repos!

You can use a private GitHub repository - Render will ask for permission when you connect it.

## Step 1: Add Gunicorn to Requirements

I've already added `gunicorn>=20.0.0` to your `requirements.txt`.

## Step 2: Push to GitHub

Make sure your `CS-FAIR-DEMO` folder is in a GitHub repository (private is fine):

```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO

# If not already a git repo:
git init
git add .
git commit -m "Prepare for Render deployment"

# If you have a remote:
git push
```

## Step 3: Deploy on Render

1. **Go to:** https://render.com
2. **Sign up/Login** (can use GitHub account)
3. **Click:** "New +" → "Web Service"
4. **Connect GitHub** (authorize Render to access your repos)
5. **Select your repository** (private repos work!)
6. **Configure:**
   - **Name:** `speech-emotion-recognition` (or your choice)
   - **Environment:** `Python 3`
   - **Region:** Choose closest to you
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** Leave empty (or `CS-FAIR-DEMO` if repo is in subfolder)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free (or Starter for $7/month - always on)

7. **Click:** "Create Web Service"

## Step 4: Wait for Deployment

Render will:
- Clone your repo
- Install dependencies (this takes a few minutes - PyTorch is large!)
- Start your app
- Give you a URL

## Step 5: Access Your App

Your app will be at:
```
https://speech-emotion-recognition.onrender.com
```
(Or whatever name you chose)

## Important Notes

### Free Tier Behavior
- **Spins down** after 15 minutes of inactivity
- **First request** after spin-down takes ~30 seconds (waking up)
- **Subsequent requests** are fast
- **512MB RAM** limit

### Model Loading
- Model loads when app starts (or on first request)
- Takes ~10-30 seconds first time
- Then stays in memory

### If Build Fails
- Check build logs in Render dashboard
- Common issues:
  - PyTorch installation (large, may timeout - but usually works)
  - Missing dependencies
  - Python version mismatch

## Your App Structure

Render expects:
```
CS-FAIR-DEMO/
├── app.py              ✅ (Flask app)
├── requirements.txt    ✅ (dependencies)
├── templates/         ✅ (HTML)
├── checkpoints/       ✅ (model file)
└── ser/              ✅ (your package)
```

## Troubleshooting

**Build takes too long:**
- Normal - PyTorch is ~900MB
- Usually completes in 5-10 minutes

**App crashes:**
- Check logs in Render dashboard
- Verify model file path is correct
- Check memory usage (free tier has limits)

**First request is slow:**
- Normal on free tier (spins down)
- Consider Starter plan ($7/month) for always-on

## Next Steps

1. Push your code to GitHub (private repo is fine)
2. Go to Render.com and create web service
3. Connect your GitHub repo
4. Deploy!

Your Flask app is already set up correctly - Render will handle the rest!

