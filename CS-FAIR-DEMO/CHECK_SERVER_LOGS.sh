#!/bin/bash
# Check server error logs

echo "=== Checking for Server Error Logs ==="
echo ""

echo "1. Check common log locations:"
find ~ -name "*error*log*" -o -name "*apache*log*" -o -name "*httpd*log*" 2>/dev/null | head -10
echo ""

echo "2. Check if there's a logs directory:"
ls -la ~/logs/ 2>/dev/null || echo "No ~/logs/ directory"
echo ""

echo "3. Check Apache error log (common location):"
ls -la /var/log/httpd/error_log 2>/dev/null || echo "Cannot access /var/log/httpd/error_log"
echo ""

echo "4. Check for .error_log in www-root:"
ls -la ~/www-root/.error_log 2>/dev/null || echo "No .error_log in www-root"
echo ""

echo "=== To view recent errors (if logs exist): ==="
echo "tail -50 <log_file>"

