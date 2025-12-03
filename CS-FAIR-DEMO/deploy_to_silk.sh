#!/bin/bash
# Deployment script for UVM Silk
# Usage: ./deploy_to_silk.sh [your_netid]

NETID=${1:-"wsander"}
SILK_HOST="w3.uvm.edu"

echo "=========================================="
echo "Deploying Speech Emotion Recognition Demo"
echo "to UVM Silk"
echo "=========================================="
echo ""

# Check if NetID was provided
if [ "$NETID" == "your_netid" ]; then
    echo "Error: Please provide your UVM NetID"
    echo "Usage: ./deploy_to_silk.sh your_netid"
    exit 1
fi

echo "NetID: $NETID"
echo "Silk Host: $SILK_HOST"
echo ""

# Create a temporary directory for files to upload (excluding unnecessary files)
echo "Preparing files for deployment..."
TEMP_DIR=$(mktemp -d)
DEPLOY_DIR="$TEMP_DIR/CS-FAIR-DEMO"

# Copy files, excluding cache and unnecessary files
mkdir -p "$DEPLOY_DIR"
rsync -av --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.git' \
          --exclude='*.log' \
          --exclude='debug_*.wav' \
          --exclude='.DS_Store' \
          CS-FAIR-DEMO/ "$DEPLOY_DIR/"

echo "Files prepared in: $DEPLOY_DIR"
echo ""

# Ask for confirmation
read -p "Ready to upload to Silk? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Upload files
echo "Uploading files to Silk..."
scp -r "$DEPLOY_DIR"/* "${NETID}@${SILK_HOST}:~/www-root/"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Upload successful!"
    echo ""
    echo "Next steps:"
    echo "1. SSH into Silk: ssh ${NETID}@${SILK_HOST}"
    echo "2. Navigate to: cd ~/www-root"
    echo "3. Install dependencies (may need virtual environment):"
    echo "   pip install -r requirements.txt"
    echo "4. Test the WSGI application"
    echo "5. Configure Silk to use wsgi.py as entry point"
    echo ""
    echo "See SILK_DEPLOYMENT.md for detailed instructions"
    echo "=========================================="
else
    echo ""
    echo "Upload failed. Please check:"
    echo "- Your NetID is correct"
    echo "- You have SSH access to Silk"
    echo "- The www-root directory exists"
fi

# Cleanup
rm -rf "$TEMP_DIR"

