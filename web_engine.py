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
from loguru import logger
import threading
import uvicorn

from utils.file_management import BASE_DIR
from video_handling.streaming import video_stream_generator, USE_LIVE_CAMERA

HOST: str = os.environ.get("VLM_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("VLM_PORT", "8000"))
IS_LOOPBACK: bool = HOST in ("127.0.0.1", "localhost", "::1")

active_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Worker threads hand broadcasts to this loop (see broadcast_sync).
    app.state.loop = asyncio.get_running_loop()
    yield
    app.state.loop = None


# No auth: expose the interactive API docs only on loopback.
_docs = {} if IS_LOOPBACK else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(lifespan=lifespan, **_docs)

# Serve static frontend
app.mount("/src", StaticFiles(directory=BASE_DIR / "src"), name="src")
app.mount("/presets", StaticFiles(directory=BASE_DIR / "presets"), name="presets")


@app.get("/")
async def get():
    return FileResponse(BASE_DIR / "src" / "page.html", media_type="text/html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(BASE_DIR / "src" / "favicon.ico", media_type="image/x-icon")


MAX_TEXT_CHARS = 2000


def emit(kind: str, text: str, **extra: float) -> dict:
    """Build a server -> client frame: {"type": "vlm"|"error"|"system", "text": str, ...}."""
    return {"type": kind, "text": text, **extra}


def parse_client_message(raw: str) -> tuple[dict | None, str | None]:
    """Return (message, None) for a valid client frame, else (None, error text)."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Invalid JSON"
    if not isinstance(msg, dict):
        return None, "Message must be a JSON object"
    text = msg.get("text", "")
    if not isinstance(text, str):
        return None, "text must be a string"
    if len(text) > MAX_TEXT_CHARS:
        return None, f"text too long (max {MAX_TEXT_CHARS} characters)"
    if not isinstance(msg.get("role", ""), str):
        return None, "role must be a string"
    return msg, None


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    prompt = ws.app.state.prompt
    settings = ws.app.state.settings

    # Browsers always send Origin; reject cross-site pages. Non-browser clients send none.
    origin = ws.headers.get("origin")
    if origin is not None and urlparse(origin).netloc != ws.headers.get("host"):
        logger.warning(f"Rejected WebSocket from origin {origin!r}")
        await ws.close(code=1008)
        return

    await ws.accept()
    active_connections.append(ws)

    try:
        while True:
            raw = await ws.receive_text()
            msg, err = parse_client_message(raw)
            if msg is None:
                logger.warning(f"Rejected client frame: {err}")
                await ws.send_text(json.dumps(emit("error", err)))
                continue
            role = msg.get("role")
            text = msg.get("text", "")
            logger.debug(f"Received msg: role={role!r} text={text[:80]!r}")
            temperature = msg.get("temperature", None)

            # Handle user commands
            if role == "user":
                if text == "toggle_stream":
                    USE_LIVE_CAMERA["value"] = not USE_LIVE_CAMERA["value"]
                    new_mode = "Live camera" if USE_LIVE_CAMERA["value"] else "Static video"
                    system_msg = emit("system", f"Switched to: {new_mode}")
                    await ws.send_text(json.dumps(system_msg))

                else:
                    resample_fps = msg.get("resample_fps", None)
                    try:
                        if temperature is not None:
                            settings.set_temperature(temperature)
                            logger.info(f"Temperature set to: {settings.get_temperature():.2f}")
                        if resample_fps is not None:
                            settings.set_resample_fps(resample_fps)
                            logger.info(f"Resample fps set to: {settings.get_resample_fps()}")
                    except ValueError as e:
                        await ws.send_text(json.dumps(emit("error", str(e))))
                        continue

                    if text:
                        prompt.set(text)
                        logger.info(f"Prompt updated to: {text}")

                    echo_msg = emit(
                        "system",
                        f"Prompt changed to: {text}, Temperature: {settings.get_temperature():.2f}, "
                        f"Resample: {settings.effective_resample_fps():.2f} fps",
                    )
                    await ws.send_text(json.dumps(echo_msg))

            else:
                logger.warning(f"Unhandled role: {role}")
                await ws.send_text(json.dumps(emit("error", f"Unknown role: {role}")))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        if ws in active_connections:
            active_connections.remove(ws)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        video_stream_generator(),
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
        logger.warning(f"Binding to {HOST}: the server has no authentication, anyone on the network can use it.")
    return uvicorn.Server(uvicorn.Config("web_engine:app", host=HOST, port=PORT, reload=False))


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
            logger.error(f"Failed to send, dropping connection: {result!r}")
            if ws in active_connections:
                active_connections.remove(ws)


def start_server_threaded(prompt=None, settings=None) -> uvicorn.Server:
    """Run uvicorn in a daemon thread; set `.should_exit = True` on the returned server to stop it."""

    app.state.prompt = prompt
    app.state.settings = settings

    server = make_server()
    threading.Thread(target=server.run, daemon=True).start()
    return server

