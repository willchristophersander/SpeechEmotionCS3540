#!/users/w/s/wsander/www-root/venv/bin/python3
"""
Test CGI script to see if POST requests work
"""

import sys
import os
import json
import cgi

print("Content-Type: application/json\n")
sys.stdout.flush()

try:
    # Get POST data
    form = cgi.FieldStorage()
    content_length = int(os.environ.get('CONTENT_LENGTH', 0))
    
    result = {
        'status': 'success',
        'content_length': content_length,
        'has_form': 'audio' in form,
        'method': os.environ.get('REQUEST_METHOD', 'unknown')
    }
    
    if content_length > 0:
        post_data = sys.stdin.read(content_length)
        try:
            data = json.loads(post_data)
            result['json_data'] = 'valid'
            result['has_audio'] = 'audio' in data
            if 'audio' in data:
                result['audio_length'] = len(data['audio'])
        except:
            result['json_data'] = 'invalid'
            result['raw_data'] = post_data[:100]  # First 100 chars
    
    print(json.dumps(result))
    sys.stdout.flush()
    
except Exception as e:
    error_response = {
        'error': str(e),
        'error_type': type(e).__name__
    }
    print(json.dumps(error_response))
    sys.stdout.flush()

