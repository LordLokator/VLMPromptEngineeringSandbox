import time
import cv2
from loguru import logger

from utils.file_management import full_path

FALLBACK_IMAGE = "src/error.jpeg"
VIDEO_PATH = full_path("./static_video/street_view.mp4")
USE_LIVE_CAMERA = {'value': False}


def static_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error("Failed to open static video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1 / fps
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        yield frame
        time.sleep(delay)


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
