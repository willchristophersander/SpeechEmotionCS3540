# Deploying to Render.com

## Prerequisites

1. **GitHub Repository** (private is fine - Render supports it)
2. **Render.com Account** (free tier available)
3. **Your code pushed to GitHub**

## Step 1: Prepare Your Code

Your Flask app is already set up! Just make sure:

- ✅ `app.py` exists (Flask app)
- ✅ `requirements.txt` exists (dependencies)
- ✅ `templates/index.html` exists
- ✅ `checkpoints/4class/crnn_emotion_model.pth` exists (model file)
- ✅ `ser/` directory exists (your package)

## Step 2: Create render.yaml (Optional but Recommended)

I've created `render.yaml` for you. This tells Render how to deploy your app.

## Step 3: Push to GitHub

```bash
cd /Users/will/Projects/SpeechEmotionCS3540/CS-FAIR-DEMO

# Make sure everything is committed
git add .
git commit -m "Prepare for Render deployment"
git push
```

## Step 4: Deploy on Render

1. **Go to Render.com** and sign up/login
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository** (private repos work fine)
4. **Select your repository** and branch
5. **Configure:**
   - **Name:** `speech-emotion-recognition` (or whatever you want)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free (or paid if you want)

6. **Click "Create Web Service"**

Render will:
- Clone your repo
- Install dependencies
- Start your app
- Give you a URL like: `https://speech-emotion-recognition.onrender.com`

## Step 5: Environment Variables (if needed)

If you need any environment variables, add them in Render dashboard:
- Settings → Environment Variables

## Important Notes

### Model File Size
- Your model file is ~26MB
- Render free tier has limits, but this should be fine
- If you hit limits, consider using a smaller model or paid tier

### First Request
- Free tier spins down after inactivity
- First request after spin-down takes ~30 seconds
- Subsequent requests are fast

### Gunicorn
Render uses Gunicorn (production WSGI server). Make sure it's in requirements.txt:
```
gunicorn>=20.0.0
```

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Make sure all dependencies are in `requirements.txt`
- Check Python version compatibility

### App Crashes
- Check runtime logs in Render dashboard
- Verify model file path is correct
- Check memory limits (free tier has limits)

### Slow First Request
- Normal on free tier (spins down after inactivity)
- Consider paid tier for always-on

## Cost

- **Free Tier:** Works, but spins down after inactivity
- **Starter ($7/month):** Always on, better performance

## Your App Will Be At

`https://your-app-name.onrender.com`

You can customize the name when creating the service.

