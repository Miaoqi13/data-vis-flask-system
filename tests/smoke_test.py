import time
import requests
import sys

for i in range(20):
    try:
        r = requests.get('http://127.0.0.1:5000', timeout=2)
        print('GET / ->', r.status_code)
        r2 = requests.get('http://127.0.0.1:5000/history', timeout=2)
        print('GET /history ->', r2.status_code)
        sys.exit(0)
    except Exception as e:
        time.sleep(0.5)
print('FAILED: could not reach server')
sys.exit(1)
