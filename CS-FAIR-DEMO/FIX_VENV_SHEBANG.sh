#!/bin/bash
# Fix shebang to use venv Python (not system Python)

cd ~/www-root

# Use the venv Python path (not the resolved system path)
VENV_PYTHON="$HOME/www-root/venv/bin/python3"

echo "=== Fixing predict.cgi to use venv Python ==="
echo ""

if [ -f "predict.cgi" ]; then
    # Create backup
    cp predict.cgi predict.cgi.bak
    
    # Update first line to use venv Python
    sed -i "1s|.*|#!$VENV_PYTHON|" predict.cgi
    
    echo "✓ Updated predict.cgi shebang to: $VENV_PYTHON"
    echo ""
    echo "First line of predict.cgi:"
    head -1 predict.cgi
    echo ""
    echo "=== Done ==="
    echo ""
    echo "The venv Python will use the venv's packages."
else
    echo "✗ predict.cgi not found!"
fi

