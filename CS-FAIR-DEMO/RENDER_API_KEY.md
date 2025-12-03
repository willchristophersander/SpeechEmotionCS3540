# Using Render API Key

## Your API Key
```
rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni
```

## Option 1: Set as Environment Variable (Recommended)

### On macOS/Linux:
```bash
export RENDER_API_KEY="rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni"
```

### Make it permanent (add to ~/.zshrc or ~/.bashrc):
```bash
echo 'export RENDER_API_KEY="rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni"' >> ~/.zshrc
source ~/.zshrc
```

## Option 2: Use with Render CLI

### Install Render CLI:
```bash
# macOS
brew install render

# Or via npm
npm install -g render-cli
```

### Authenticate:
```bash
render auth login
# Or set the key directly:
export RENDER_API_KEY="rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni"
```

### Use CLI commands:
```bash
# List services
render services list

# Get service info
render services get srv-d4nqc9i4d50c739qmuag

# Trigger manual deploy
render deploys create --service-id srv-d4nqc9i4d50c739qmuag

# Update service settings
render services update srv-d4nqc9i4d50c739qmuag --root-dir CS-FAIR-DEMO
```

## Option 3: Use with Render REST API

### Example: Update Root Directory via API
```bash
curl -X PATCH \
  "https://api.render.com/v1/services/srv-d4nqc9i4d50c739qmuag" \
  -H "Authorization: Bearer rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni" \
  -H "Content-Type: application/json" \
  -d '{
    "rootDir": "CS-FAIR-DEMO"
  }'
```

### Example: Trigger Manual Deploy
```bash
curl -X POST \
  "https://api.render.com/v1/services/srv-d4nqc9i4d50c739qmuag/deploys" \
  -H "Authorization: Bearer rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni" \
  -H "Content-Type: application/json"
```

## Option 4: Quick Fix - Update Root Directory Now

Run this command to fix the deployment issue:

```bash
curl -X PATCH \
  "https://api.render.com/v1/services/srv-d4nqc9i4d50c739qmuag" \
  -H "Authorization: Bearer rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni" \
  -H "Content-Type: application/json" \
  -d '{"rootDir": "CS-FAIR-DEMO"}'
```

This will:
1. Set the root directory to `CS-FAIR-DEMO`
2. Automatically trigger a new deployment
3. Use the correct `requirements.txt` with Flask and gunicorn

## Security Note

⚠️ **Never commit API keys to Git!**

If you need to use the key in scripts, use environment variables:
```bash
# In your script
API_KEY="${RENDER_API_KEY:-rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni}"
```

Or use a `.env` file (and add it to `.gitignore`):
```
RENDER_API_KEY=rnd_CDeFJvIjtCuwVzX9ELKCbGlUeyni
```

