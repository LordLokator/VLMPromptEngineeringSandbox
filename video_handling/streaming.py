import time
import cv2
from loguru import logger

from utils.file_management import full_path

FALLBACK_IMAGE = "src/error.jpeg"
USE_LIVE_CAMERA = {'value': False}
VIDEO_PATHS = [
    full_path("./static_video/street_view_1.mp4"),
    full_path("./static_video/street_view_2.mp4")
]
VIDEO_IDX = 0


def static_video():
    cap, fps, delay = _init_video_object(VIDEO_PATHS[VIDEO_IDX])

    while True:
        ret, frame = cap.read()
        if not ret:
            VIDEO_IDX = (VIDEO_IDX + 1) % len(VIDEO_PATHS)
            cap, fps, delay = _init_video_object(VIDEO_PATHS[VIDEO_IDX])
            continue

        start = time.time()
        yield frame
        elapsed = time.time() - start
        time.sleep(max(0, delay - elapsed))

def _init_video_object(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error("Failed to open static video.")
        return (None, 0, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1 / fps
    return (cap, fps, delay)

def live_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("Webcam not found. Showing fallback image.")
        fallback = cv2.imread(FALLBACK_IMAGE)
        while True:
            yield fallback
            time.sleep(1 / 5)
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            yield frame


def get_stream_source():
    return live_stream() if USE_LIVE_CAMERA['value'] else static_video()


def video_stream_generator():
    current_mode = USE_LIVE_CAMERA["value"]
    stream = get_stream_source()

    if stream is None:
        return
    for frame in stream:
        # Check if mode changed mid-stream
        if USE_LIVE_CAMERA["value"] != current_mode:
            logger.info("Stream mode changed — switching generator")
            return  # Kill generator

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
