import asyncio
from contextlib import asynccontextmanager
import json
import os
from urllib.parse import urlparse
from fastapi import Request
from utils.button_presets import load_presets, load_user_presets, save_user_preset
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger
import threading
from typing import Any
import uvicorn

from utils import telemetry
from utils.errors import log_failure
from utils.session import CLIENT_TYPES, message
from utils.hosts import LOOPBACK_HOSTS, parse_allowed_hosts, require_extra_hosts
from utils.config import DEFAULT_PORT, ConfigError, parse_port
from utils.file_management import BASE_DIR
from video_handling.streaming import mjpeg_stream

# Read at import: main.py loads .env before importing this module. An empty variable counts as unset.
HOST: str = os.environ.get("VLM_HOST", "").strip() or "127.0.0.1"
try:
    PORT: int = parse_port(os.environ.get("VLM_PORT", ""))
except ConfigError:
    PORT = DEFAULT_PORT  # never served: main.py's startup check reports the bad value with the others and stops
IS_LOOPBACK: bool = HOST in ("127.0.0.1", "localhost", "::1")
# Names served besides the loopback ones. A bad value stops the app at import, before anything starts.
EXTRA_HOSTS: list[str] = parse_allowed_hosts(os.environ.get("VLM_ALLOWED_HOSTS", ""))

active_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Worker threads hand broadcasts to this loop (see broadcast_sync).
    app.state.loop = asyncio.get_running_loop()
    yield
    app.state.loop = None


# No auth: expose the interactive API docs only on loopback.
app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs" if IS_LOOPBACK else None,
    redoc_url="/redoc" if IS_LOOPBACK else None,
    openapi_url="/openapi.json" if IS_LOOPBACK else None,
)

class SafeErrors:
    """Stops exceptions inside the app with a fixed reply: HTTP 500, or WebSocket close 1011.

    With OTEL_ENABLED=1, the instrumentation would otherwise copy an exception's message and stack onto the server span,
    and those can quote paths or keys. User middlewares run inside its exception recorder, so this one catches first.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(event: dict) -> None:
            nonlocal started
            if event["type"] == "http.response.start":
                started = True
            await send(event)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as e:  # noqa: BLE001 - the boundary: nothing below may reach the server span
            log_failure(f"Unhandled error in {scope['type']} {scope.get('path', '')}", e)
            try:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1011})
                elif not started:
                    await send({"type": "http.response.start", "status": 500,
                                "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
                    await send({"type": "http.response.body", "body": b"Internal Server Error"})
            except Exception:  # noqa: BLE001 - the connection is already gone
                pass


app.add_middleware(SafeErrors)

# DNS rebinding: a page on another domain can resolve its name to this machine, and its Origin then matches
# its Host. Only the names below are served, for HTTP and the WebSocket alike.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*LOOPBACK_HOSTS, *EXTRA_HOSTS], www_redirect=False)

# Serve static frontend
# Only the frontend is static. presets/ is not mounted: it holds the prompts users saved; /preset_buttons serves them.
app.mount("/src", StaticFiles(directory=BASE_DIR / "src"), name="src")


@app.get("/")
async def get():
    return FileResponse(BASE_DIR / "src" / "page.html", media_type="text/html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(BASE_DIR / "src" / "favicon.ico", media_type="image/x-icon")


MAX_TEXT_CHARS = 2000
# Raw frame cap in bytes. uvicorn closes the socket (code 1009) on larger frames before they are
# parsed; its default is 16 MiB. 64 KiB leaves room for the 2000-char text plus JSON overhead.
WS_MAX_BYTES = 64 * 1024


def emit(kind: str, text: str, **extra: Any) -> dict:
    """Build a `system` or `error` frame: {"type": kind, "text": str, ...}."""
    return message(kind, text, **extra)


def parse_client_message(raw: str) -> tuple[dict | None, str | None]:
    """Return (message, None) for a valid client frame, else (None, error text). Fields are checked by the session."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Invalid JSON"
    if not isinstance(msg, dict):
        return None, "Message must be a JSON object"
    if msg.get("type") not in CLIENT_TYPES:
        return None, f"type must be one of: {', '.join(CLIENT_TYPES)}"
    text = msg.get("text")
    if isinstance(text, str) and len(text) > MAX_TEXT_CHARS:
        return None, f"text too long (max {MAX_TEXT_CHARS} characters)"
    return msg, None


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    session = ws.app.state.session

    # Browsers always send Origin; reject cross-site pages. Non-browser clients send none.
    origin = ws.headers.get("origin")
    if origin is not None and urlparse(origin).netloc != ws.headers.get("host"):
        logger.warning(f"Rejected WebSocket from origin {origin!r}")
        await ws.close(code=1008)
        return

    await ws.accept()
    active_connections.append(ws)
    telemetry.ws_connection(1)

    try:
        await ws.send_text(json.dumps(await asyncio.to_thread(session.state)))
        while True:
            raw = await ws.receive_text()
            msg, err = parse_client_message(raw)
            if msg is None:
                telemetry.record_ws_message(None)
                logger.warning(f"Rejected client frame: {err}")
                await ws.send_text(json.dumps(emit("error", err or "invalid message")))
                continue
            telemetry.record_ws_message(msg["type"])
            logger.debug(f"Received {msg['type']} frame")
            # Off the event loop: the session broadcasts while holding its lock (see utils/session.py).
            for reply in await asyncio.to_thread(session.handle, msg):
                await ws.send_text(json.dumps(reply))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        telemetry.ws_connection(-1)
        if ws in active_connections:
            active_connections.remove(ws)
        session.disconnected(len(active_connections))  # synchronous: an await here is skipped on cancellation


@app.get("/video_feed")
def video_feed(request: Request):
    return StreamingResponse(
        mjpeg_stream(request.app.state.hub),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/preset_buttons")
def get_presets():
    sys_presets = load_presets()
    user_presets = load_user_presets()
    logger.info(f"{len(sys_presets)} system presets and {len(user_presets)} user presets were loaded.")
    logger.debug(f"System presets: [ {sys_presets} ].")
    logger.debug(f"User presets: [ {user_presets} ].")

    return sys_presets + user_presets


@app.post("/save_preset")
async def post_preset(request: Request):
    if not request.headers.get("content-type", "").startswith("application/json"):
        return JSONResponse({"status": "error", "reason": "Content-Type must be application/json"}, status_code=415)

    def bad(reason: str) -> JSONResponse:
        return JSONResponse({"status": "error", "reason": reason}, status_code=400)

    try:
        data = await request.json()
    except ValueError:
        return bad("Body must be valid JSON")
    if not isinstance(data, dict):
        return bad("Body must be a JSON object")

    label, text = data.get("label"), data.get("text")
    if not isinstance(label, str) or not isinstance(text, str):
        return bad("label and text must be strings")
    label, text = label.strip()[:25], text.strip()
    if not label or not text:
        return bad("label and text must not be empty")
    if len(text) > MAX_TEXT_CHARS:
        return bad(f"text too long (max {MAX_TEXT_CHARS} characters)")

    save_user_preset(label, text)
    return {"status": "ok", "label": label}


def make_server() -> uvicorn.Server:
    if not IS_LOOPBACK:
        require_extra_hosts(HOST, EXTRA_HOSTS)
        logger.warning(f"Binding to {HOST}: the server has no authentication, anyone on the network can use it.")
    # log_config=None: uvicorn adds no handlers of its own, so its records reach the redacting sink.
    config = uvicorn.Config(
        "web_engine:app", host=HOST, port=PORT, reload=False, ws_max_size=WS_MAX_BYTES, log_config=None
    )
    return uvicorn.Server(config)


def broadcast_sync(msg: dict, timeout: float = 5.0) -> None:
    """Called from worker threads: run the broadcast on the server's own event loop."""
    loop = getattr(app.state, "loop", None)
    if loop is None:
        logger.warning("Server event loop not running yet, dropping broadcast.")
        return
    future = asyncio.run_coroutine_threadsafe(broadcast(msg), loop)
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        logger.warning("Broadcast timed out.")


async def broadcast(message: dict) -> None:
    """Send to every client; sockets that fail are pruned. Runs on the server loop only."""
    msg = json.dumps(message)
    sockets = list(active_connections)
    results = await asyncio.gather(*(ws.send_text(msg) for ws in sockets), return_exceptions=True)

    for ws, result in zip(sockets, results, strict=True):
        if isinstance(result, BaseException):
            logger.error(f"Failed to send, dropping connection: {type(result).__name__}")
            if ws in active_connections:
                active_connections.remove(ws)


def start_server_threaded(session=None, hub=None) -> uvicorn.Server:
    """Run uvicorn in a daemon thread; set `.should_exit = True` on the returned server to stop it."""

    app.state.session = session
    app.state.hub = hub
    telemetry.setup_telemetry(app)

    server = make_server()
    threading.Thread(target=server.run, daemon=True).start()
    return server

