import datetime
import sys
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from streaming import video_stream_generator, USE_LIVE_CAMERA
from utils.button_presets import load_presets

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
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "toggle_stream":
                USE_LIVE_CAMERA["value"] = not USE_LIVE_CAMERA["value"]

                state = "Live camera" if USE_LIVE_CAMERA["value"] else "Static video"
                logger.info(f"Toggled stream source → {state}")

                await ws.send_text(f"Switched to: {state}")
            else:
                logger.info(f"Received from client: {msg}")

            await ws.send_text(f"Echo: {msg}")

    except WebSocketDisconnect:
        logger.info("Client disconnected")


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
    return load_presets()


if __name__ == "__main__":
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.info("Starting server on http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)
