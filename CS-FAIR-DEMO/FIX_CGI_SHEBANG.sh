#!/bin/bash
# Script to fix the shebang in predict.cgi on Silk

echo "=== Fixing predict.cgi Shebang ==="
echo ""

cd ~/www-root

# Find the correct Python path
PYTHON_PATH=$(readlink -f venv/bin/python3 2>/dev/null || which ~/www-root/venv/bin/python3 2>/dev/null || echo "/users/w/s/wsander/www-root/venv/bin/python3")

echo "Found Python at: $PYTHON_PATH"
echo ""

# Update the shebang
if [ -f "predict.cgi" ]; then
    # Create backup
    cp predict.cgi predict.cgi.bak
    
    # Update first line
    sed -i "1s|.*|#!$PYTHON_PATH|" predict.cgi
    
    echo "✓ Updated predict.cgi shebang to: $PYTHON_PATH"
    echo ""
    echo "First line of predict.cgi:"
    head -1 predict.cgi
    echo ""
    echo "=== Done ==="
    echo ""
    echo "Test the script:"
    echo "  ~/www-root/venv/bin/python3 predict.cgi"
    echo ""
    echo "If it imports without errors, it should work!"
else
    echo "✗ predict.cgi not found!"
fi

