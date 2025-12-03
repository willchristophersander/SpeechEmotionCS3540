#!/bin/bash
# Quick fix for CGI execution

cd ~/www-root

echo "=== Fixing CGI Execution ==="
echo ""

echo "1. Making predict.cgi executable..."
chmod 755 predict.cgi
echo "✓ Done"
echo ""

echo "2. Updating .htaccess to enable .cgi files..."
if [ -f ".htaccess" ]; then
    # Check if .cgi is already in AddHandler
    if grep -q "AddHandler.*\.cgi" .htaccess; then
        echo "✓ .cgi already enabled in .htaccess"
    else
        # Add .cgi to AddHandler line
        sed -i 's/AddHandler cgi-script .py/AddHandler cgi-script .py .cgi/' .htaccess
        echo "✓ Updated .htaccess"
    fi
else
    # Create .htaccess
    cat > .htaccess << 'EOF'
Options +ExecCGI
AddHandler cgi-script .py .cgi
EOF
    echo "✓ Created .htaccess"
fi
echo ""

echo "3. Verifying .htaccess:"
cat .htaccess
echo ""

echo "4. Verifying predict.cgi permissions:"
ls -la predict.cgi
echo ""

echo "=== Done ==="
echo ""
echo "Test: https://wsander.w3.uvm.edu/test.cgi (if you create a test file)"
echo "Or try your app again!"

