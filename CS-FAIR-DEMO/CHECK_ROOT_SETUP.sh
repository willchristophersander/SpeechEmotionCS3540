#!/bin/bash
# Script to check root directory setup on Silk

echo "=== Checking Root Directory Setup ==="
echo ""

echo "1. Current directory:"
pwd
echo ""

echo "2. Files in root:"
ls -la | head -20
echo ""

echo "3. Checking for default index files:"
ls -la index.* 2>/dev/null || echo "No index files found"
echo ""

echo "4. Checking for wsgi.py:"
if [ -f "wsgi.py" ]; then
    echo "✓ wsgi.py exists"
    ls -lh wsgi.py
else
    echo "✗ wsgi.py NOT FOUND"
fi
echo ""

echo "5. Checking for app.py:"
if [ -f "app.py" ]; then
    echo "✓ app.py exists"
    ls -lh app.py
else
    echo "✗ app.py NOT FOUND"
fi
echo ""

echo "6. Checking for .htaccess:"
if [ -f ".htaccess" ]; then
    echo "✓ .htaccess exists"
    cat .htaccess
else
    echo "✗ .htaccess NOT FOUND"
fi
echo ""

echo "7. Checking for templates:"
if [ -d "templates" ]; then
    echo "✓ templates/ exists"
    ls templates/
else
    echo "✗ templates/ NOT FOUND"
fi
echo ""

echo "=== Summary ==="
echo "If wsgi.py exists but site shows default page:"
echo "  → WSGI is likely not enabled"
echo "  → Contact SAA to enable WSGI"
echo ""
echo "If index.html exists:"
echo "  → It may be taking precedence"
echo "  → Consider: rm index.html (if it's the default one)"

