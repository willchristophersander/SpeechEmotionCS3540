#!/bin/bash
# Script to verify Silk setup on the server

echo "=== Verifying Silk Setup ==="
echo ""

echo "1. Checking current directory..."
pwd
echo ""

echo "2. Checking required files..."
ls -la app.py wsgi.py requirements.txt 2>/dev/null || echo "Missing files!"
echo ""

echo "3. Checking directory structure..."
ls -la templates/ checkpoints/ ser/ 2>/dev/null || echo "Missing directories!"
echo ""

echo "4. Checking model file..."
ls -lh checkpoints/4class/crnn_emotion_model.pth 2>/dev/null || echo "Model file not found!"
echo ""

echo "5. Checking Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment found"
    echo "Python version:"
    venv/bin/python3 --version
    echo ""
    echo "Checking key packages:"
    venv/bin/pip list | grep -E "flask|torch|librosa|numpy|noisereduce"
else
    echo "Virtual environment not found!"
fi
echo ""

echo "6. Testing WSGI import..."
venv/bin/python3 -c "import sys; sys.path.insert(0, '.'); from wsgi import application; print('WSGI import: SUCCESS')" 2>&1
echo ""

echo "7. Testing app import..."
venv/bin/python3 -c "import sys; sys.path.insert(0, '.'); from app import app; print('App import: SUCCESS')" 2>&1
echo ""

echo "=== Setup Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. If all checks pass, try accessing: https://www.uvm.edu/~wsander/"
echo "2. If you get 404, contact SAA to enable WSGI"
echo "3. If you get 500, check server logs"

