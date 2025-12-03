#!/bin/bash
# Test the CGI script

cd ~/www-root

echo "=== Testing predict.cgi ==="
echo ""

echo "1. Check if file exists and is executable:"
ls -la predict.cgi
echo ""

echo "2. Check shebang:"
head -1 predict.cgi
echo ""

echo "3. Test if Python can parse the script (syntax check):"
~/www-root/venv/bin/python3 -m py_compile predict.cgi 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Syntax OK"
else
    echo "✗ Syntax error!"
fi
echo ""

echo "4. Test if imports work (simulate what the script does):"
~/www-root/venv/bin/python3 -c "
import sys
import os
sys.path.insert(0, '.')
try:
    import numpy as np
    import torch
    import librosa
    from ser.utils.audio_preprocessor import AudioPreprocessor
    from ser.models import CRNN
    print('✓ All imports OK')
except ImportError as e:
    print(f'✗ Import error: {e}')
"
echo ""

echo "5. Test script execution (will fail without POST data, but should show JSON error):"
echo '{"audio":"test"}' | ~/www-root/venv/bin/python3 predict.cgi 2>&1 | head -5
echo ""

echo "=== Test Complete ==="

