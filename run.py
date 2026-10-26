"""
CustomerIQ Application Launcher
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"==================================================")
    print(f" Starting CustomerIQ Autonomous Analytics Platform")
    print(f" URL: http://127.0.0.1:{port}")
    print(f"==================================================")
    app.run(host='127.0.0.1', port=port, debug=False)

