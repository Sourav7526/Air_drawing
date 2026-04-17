"""
serve.py – Minimal HTTP server for the Air Drawing web app.
Run this and open the printed link in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIR)

Handler = http.server.SimpleHTTPRequestHandler

print(f"\n  ✏️  Air Drawing – Web Edition")
print(f"  ────────────────────────────")
print(f"  🌐  Open in browser: http://localhost:{PORT}")
print(f"  📁  Serving from:    {DIR}")
print(f"  ⛔  Press Ctrl+C to stop\n")

try:
    webbrowser.open(f"http://localhost:{PORT}")
except Exception:
    pass

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n  Server stopped.")
    sys.exit(0)
except OSError as e:
    if "Address already in use" in str(e):
        PORT = 8001
        print(f"  Port 8000 in use, trying {PORT}...")
        print(f"  🌐  Open: http://localhost:{PORT}\n")
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    else:
        raise
