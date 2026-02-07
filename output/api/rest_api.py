"""REST API module."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    """HTTP request handler"""
    
    def do_GET(self):
        """Handle GET requests"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok', 'api': 'vibe-guard-advanced'}).encode())
    
    def log_message(self, format, *args):
        """Suppress logging"""
        pass


def start_api(host='localhost', port=8000):
    """Start REST API server"""
    server = HTTPServer((host, port), Handler)
    print(f"API server running on {host}:{port}")
    server.serve_forever()
