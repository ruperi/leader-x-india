"""Loopback-only local UI and read-only scan API. Run: python server.py."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import screener

PUBLIC = Path(__file__).resolve().parent / 'public'
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/index.html': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}
MAX_BODY = 32768


class Handler(BaseHTTPRequestHandler):
    server_version = 'LocalScreener'

    def log_message(self, format, *args):
        pass  # Neither request bodies nor arbitrary URL strings enter logs.

    def send_data(self, code, body, content_type='application/json; charset=utf-8'):
        if isinstance(body, dict):
            body = json.dumps(body, allow_nan=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def host_allowed(self):
        hosts = self.headers.get_all('Host', [])
        port = self.server.server_port
        return len(hosts) == 1 and hosts[0] in (f'127.0.0.1:{port}', f'localhost:{port}')

    def do_GET(self):
        if not self.host_allowed():
            return self.send_data(403, {'error': 'Host is not allowed.'})
        if self.path == '/api/status':
            return self.send_data(200, screener.status())
        entry = STATIC.get(self.path)
        if not entry:
            return self.send_data(404, {'error': 'Not found.'})
        path = PUBLIC / entry[0]
        try:
            if path.resolve().parent != PUBLIC.resolve():
                raise OSError()
            body = path.read_bytes()
        except OSError:
            return self.send_data(404, {'error': 'Static file not found.'})
        self.send_data(200, body, entry[1])

    def do_POST(self):
        if not self.host_allowed():
            return self.send_data(403, {'error': 'Host is not allowed.'})
        origins = self.headers.get_all('Origin', [])
        if origins != ['http://' + self.headers['Host']]:
            return self.send_data(403, {'error': 'Same-origin request required.'})
        if self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin'):
            return self.send_data(403, {'error': 'Same-origin request required.'})
        if self.path != '/api/scan':
            return self.send_data(404, {'error': 'Not found.'})
        if self.headers.get_content_type() != 'application/json':
            return self.send_data(415, {'error': 'Content-Type must be application/json.'})
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
            return self.send_data(400, {'error': 'A single Content-Length is required.'})
        try:
            size = int(lengths[0])
        except ValueError:
            return self.send_data(400, {'error': 'Invalid Content-Length.'})
        if not 0 < size <= MAX_BODY:
            return self.send_data(413, {'error': 'Request body exceeds limit or is empty.'})
        try:
            self.connection.settimeout(10)
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError()
            payload = json.loads(body)
        except (ValueError, OSError):
            return self.send_data(400, {'error': 'Invalid JSON request body.'})
        try:
            result = screener.scan(payload)
        except screener.ScreenerError as exc:
            code = 409 if 'already running' in str(exc) else 400
            mode = payload.get('mode') if isinstance(payload, dict) else None
            return self.send_data(code, {'mode': mode, 'as_of': screener.now_ist().isoformat(),
                                        'error': str(exc), 'rows': [],
                                        'errors': [{'message': str(exc)}], 'warnings': []})
        except Exception:
            return self.send_data(500, {'error': 'Scan failed unexpectedly. Check server setup.'})
        self.send_data(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'Read-only screener: http://127.0.0.1:{args.port}')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
