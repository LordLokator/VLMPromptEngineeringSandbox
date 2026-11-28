"""A loopback HTTP server that records requests and answers like a chat API. Used by the provider wire checks.

It shows what the real SDKs send. It is not a provider run.
"""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHAT_REPLY = {
    "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "stub-model-2026",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "a street"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127},
}
MESSAGES_REPLY = {
    "id": "msg_stub", "type": "message", "role": "assistant", "model": "stub-claude",
    "content": [{"type": "text", "text": "a street"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 20,
              "cache_creation_input_tokens": 0},
}


class Recorder:
    def __init__(self):
        self.requests: list[dict] = []
        self.routes: dict[tuple[str, str], tuple[int, dict, dict]] = {}

    def route(self, method: str, path: str, body: dict, status: int = 200, headers: dict | None = None) -> None:
        self.routes[(method, path)] = (status, body, headers or {})


@contextmanager
def stub_server():
    """Yields (base URL, recorder). Unknown routes answer 404."""
    recorder = Recorder()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            recorder.requests.append({
                "method": method, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(raw) if raw else None,
            })
            status, body, headers = recorder.routes.get((method, self.path), (404, {"error": "no route"}, {}))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._answer("GET")

        def do_POST(self):
            self._answer("POST")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", recorder
    finally:
        server.shutdown()
        server.server_close()
