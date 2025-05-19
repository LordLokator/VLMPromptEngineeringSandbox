import asyncio
import datetime
import sys
import json
from types import NoneType
from fastapi import Request
from utils.button_presets import load_presets, load_user_presets, save_user_preset
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import threading
import uvicorn

from video_hanlding.streaming import video_stream_generator, USE_LIVE_CAMERA

TEMPERATURE: float = 0.01

active_connections: list[WebSocket] = []

app = FastAPI()

# Serve static frontend
app.mount("/src", StaticFiles(directory="src"), name="src")
app.mount("/presets", StaticFiles(directory="presets"), name="presets")


@app.get("/")
async def get():
    return HTMLResponse(open("src/page.html").read())


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("src/favicon.ico", media_type="image/x-icon")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    prompt = ws.app.state.prompt

    await ws.accept()
    active_connections.append(ws)

    try:
        while True:
            raw = await ws.receive_text()
            msg: dict[str, str] = json.loads(raw)
            logger.debug(f"Recieved msg: [ {msg} ]")

            role = msg.get("role")
            text = msg.get("text", "")
            temperature = msg.get("temperature", None)

            # Handle user commands
            if role == "user":
                if text == "toggle_stream":
                    USE_LIVE_CAMERA["value"] = not USE_LIVE_CAMERA["value"]
                    new_mode = "Live camera" if USE_LIVE_CAMERA["value"] else "Static video"
                    system_msg = {"role": "system", "text": f"Switched to: {new_mode}"}
                    await ws.send_text(json.dumps(system_msg))

                else:
                    if temperature is not None:
                        global TEMPERATURE

                        TEMPERATURE = float(temperature)
                        logger.info(f"Temperature set to: {TEMPERATURE:.2f}")

                    if text:
                        prompt.set(text)
                        logger.info(f"Prompt updated to: {text}")

                    echo_msg = {"role": "system", "text": f"Prompt changed to: {text}, Temperature: {TEMPERATURE:.2f}"}
                    await ws.send_text(json.dumps(echo_msg))

            else:
                logger.warning(f"Unhandled role: {role}")
                await ws.send_text(json.dumps({"role": "system", "text": f"Unknown role: {role}"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
        active_connections.remove(ws)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        video_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/vlm_output")
def vlm_output():
    return {
        "output": "Output dummy"
    }


@app.get("/metadata")
def metadata():
    return {
        "timestamp": datetime.datetime.now(),
        "tokens/sec": 35.2,
        "throughput": "12 fps",
        "session_id": "abc123"
    }


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
    data = await request.json()

    label = data.get("label", "").strip()[:25]
    text = data.get("text", "").strip()

    if label and text:
        save_user_preset(label, text)
        return {"status": "ok", "label": label}

    return {"status": "error", "reason": "Invalid input"}


def start_server():
    uvicorn.run("web_engine:app", host="0.0.0.0", port=8000, reload=False)


def broadcast_sync(msg: dict):
    try:
        asyncio.run(broadcast(msg))
    except RuntimeError:
        # Already inside an event loop (common in FastAPI)
        asyncio.create_task(broadcast(msg))


async def broadcast(message: str):
    to_remove = []

    msg = json.dumps(message)

    for ws in active_connections:
        try:
            asyncio.create_task(ws.send_text(msg))
        except Exception as e:
            logger.error(f"Failed to send: {e}")
            to_remove.append(ws)

    for ws in to_remove:
        try:
            if ws in active_connections:
                active_connections.remove(ws)

        except ValueError:
            logger.warning("Tried to remove non-existent WebSocket.")


def start_server_threaded(prompt=None):

    app.state.prompt = prompt

    t = threading.Thread(
        target=start_server,
        daemon=True
    )
    t.start()


if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.info("Starting server on http://localhost:8000")

    try:
        start_server_threaded()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected.")
