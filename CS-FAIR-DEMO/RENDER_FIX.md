# Fix for Render Deployment

## Problem
Render was using the root `requirements.txt` (missing Flask, gunicorn, etc.) instead of `CS-FAIR-DEMO/requirements.txt`.

## Solution
I've updated `render.yaml` to specify `rootDir: CS-FAIR-DEMO`.

## Next Steps

### Option 1: Update in Render Dashboard (Recommended)
1. Go to your Render service: https://dashboard.render.com/web/srv-d4nqc9i4d50c739qmuag
2. Go to **Settings**
3. Find **Root Directory** field
4. Set it to: `CS-FAIR-DEMO`
5. Click **Save Changes**
6. Render will automatically redeploy

### Option 2: Use render.yaml (After pushing)
1. Push the updated `render.yaml` to GitHub
2. Render should pick it up automatically

## Verify
After redeploy, check the build logs to confirm it's installing:
- ✅ `flask>=2.0.0`
- ✅ `gunicorn>=20.0.0`
- ✅ `torch>=1.12.0`
- ✅ `noisereduce>=2.0.0`

## Your Render API Key
I've noted your API key. You can use it for CLI deployments if needed, but the dashboard method above is easier.

