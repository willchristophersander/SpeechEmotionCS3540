# Deploying to UVM Silk

## Overview

UVM Silk supports Python and Flask applications, making it suitable for deploying the Speech Emotion Recognition demo. However, some modifications are needed for production deployment.

## Steps to Deploy

### 1. Request Silk Account

Email Systems Architecture & Administration (SAA) with:
- Your UVM NetID
- Request for Silk web hosting account
- Mention you're deploying a Python Flask application

### 2. Access Your Account

SSH into Silk:
```bash
ssh your_netid@w3.uvm.edu
# or
ssh your_netid@silk.uvm.edu
```

### 3. Upload Files

Upload your CS-FAIR-DEMO files to your `www-root` directory:
```bash
# From your local machine
scp -r CS-FAIR-DEMO/* your_netid@w3.uvm.edu:~/www-root/
```

### 4. Install Dependencies

On Silk, you may need to:
- Use a virtual environment
- Install Python packages (PyTorch, librosa, Flask, etc.)
- Note: PyTorch is large (~500MB), ensure you have sufficient disk space

### 5. Configure for Production

The app needs modifications for Silk deployment (see below).

## Required Modifications

### A. Create WSGI Entry Point

Silk typically uses WSGI for Python apps. Create `wsgi.py`:

```python
#!/usr/bin/env python3
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import app

if __name__ == "__main__":
    app.run()
```

### B. Modify app.py for Production

Update the `if __name__ == '__main__'` block:

```python
if __name__ == '__main__':
    # For local development
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
else:
    # For production (WSGI)
    load_model()
```

### C. Configure Host and Port

Silk may have specific requirements. Check with SAA about:
- Port configuration
- Host binding
- Process management (may use gunicorn or similar)

### D. Environment Considerations

- **Model Loading**: Ensure the model path works in production
- **File Permissions**: Make sure model file is readable
- **Memory**: PyTorch models can be memory-intensive
- **Timeout**: Long-running requests may need special configuration

## Potential Challenges

1. **PyTorch Installation**: PyTorch is large and may require specific installation
2. **Memory Limits**: Loading the model requires significant RAM
3. **Processing Time**: Audio processing may hit timeout limits
4. **Dependencies**: librosa and scipy may have system dependencies

## Alternative: Use Gunicorn

If Silk supports it, use Gunicorn for production:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app
```

## Testing Locally First

Before deploying, test the WSGI configuration locally:
```bash
gunicorn wsgi:app
```

## Contact SAA

For specific questions about:
- Python version available
- Package installation process
- Resource limits (memory, disk, CPU)
- Port and host configuration
- Process management options

Email: SAA (check UVM IT website for current contact)

## Notes

- The model file (26MB) should fit within typical disk quotas
- Consider using a smaller model or model quantization if memory is limited
- May need to configure CORS if accessing from different domains
- Ensure all file paths are relative or use environment variables

