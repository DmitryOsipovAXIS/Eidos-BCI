import asyncio
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

VALID_TYPES = {"WIDGET", "FUNCTION"}


class EventHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default access log

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        event_type = params.get("type", [None])[0]

        if event_type not in VALID_TYPES:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "type must be WIDGET or FUNCTION"}')
            return

        ts = str(datetime.now().timestamp())
        message = '{"type": "' + event_type + '", "timestamp": ' + ts + '}'

        asyncio.run_coroutine_threadsafe(
            self.server.broadcast(message), self.server.loop)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')


async def start_http(broadcast, host="0.0.0.0", port=8080):
    server = HTTPServer((host, port), EventHandler)
    server.loop = asyncio.get_running_loop()
    server.broadcast = broadcast
    print(f"HTTP server listening on {host}:{port}")

    await asyncio.get_running_loop().run_in_executor(None, server.serve_forever)
