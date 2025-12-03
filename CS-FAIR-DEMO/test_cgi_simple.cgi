#!/home/w/s/wsander/www-root/venv/bin/python3
import sys
import json

print("Content-Type: application/json\n")
sys.stdout.flush()

try:
    result = {"status": "test", "message": "CGI is working"}
    print(json.dumps(result))
    sys.stdout.flush()
except Exception as e:
    error = {"error": str(e)}
    print(json.dumps(error))
    sys.stdout.flush()

