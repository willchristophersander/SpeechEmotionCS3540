#!/usr/bin/env python3
"""
WSGI entry point for Silk deployment.

This file allows the Flask application to be served by WSGI-compatible
web servers (used by UVM Silk).
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from app import app

# For WSGI servers, the app object is what they need
application = app

# If running directly (for testing), load model and run
if __name__ == '__main__':
    # This will only run if executed directly, not when imported by WSGI server
    from app import load_model
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=False)

