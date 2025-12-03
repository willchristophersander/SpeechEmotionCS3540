#!/bin/bash
# Script to move app files to SpeechEmotionProject subdirectory on Silk

echo "=== Moving files to SpeechEmotionProject subdirectory ==="
echo ""

# Create subdirectory
mkdir -p SpeechEmotionProject

# Move all app files
echo "Moving files..."
mv app.py SpeechEmotionProject/ 2>/dev/null && echo "✓ app.py"
mv wsgi.py SpeechEmotionProject/ 2>/dev/null && echo "✓ wsgi.py"
mv requirements.txt SpeechEmotionProject/ 2>/dev/null && echo "✓ requirements.txt"
mv .htaccess SpeechEmotionProject/ 2>/dev/null && echo "✓ .htaccess"

# Move directories
mv templates/ SpeechEmotionProject/ 2>/dev/null && echo "✓ templates/"
mv checkpoints/ SpeechEmotionProject/ 2>/dev/null && echo "✓ checkpoints/"
mv ser/ SpeechEmotionProject/ 2>/dev/null && echo "✓ ser/"
mv 4class_model/ SpeechEmotionProject/ 2>/dev/null && echo "✓ 4class_model/"

# Move documentation (optional - you can keep these in root if you want)
# mv *.md SpeechEmotionProject/ 2>/dev/null
# mv *.sh SpeechEmotionProject/ 2>/dev/null

echo ""
echo "=== Files moved successfully ==="
echo ""
echo "Your app is now at: https://www.uvm.edu/~wsander/SpeechEmotionProject/"
echo ""
echo "Next steps:"
echo "1. cd SpeechEmotionProject"
echo "2. Test: ~/www-root/venv/bin/python3 wsgi.py"
echo "3. Access: https://www.uvm.edu/~wsander/SpeechEmotionProject/"

