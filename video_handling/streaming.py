import time
import cv2
from loguru import logger

from utils.file_management import BASE_DIR

FALLBACK_IMAGE = str(BASE_DIR / "src" / "error.jpeg")
USE_LIVE_CAMERA = {'value': False}
VIDEO_PATHS = [
    str(BASE_DIR / "static_video" / "street_view_1.mp4"),
    str(BASE_DIR / "static_video" / "street_view_2.mp4"),
]
VIDEO_IDX = 0


def static_video(paced: bool = True):
    global VIDEO_IDX

    cap, fps, delay = _init_video_object(VIDEO_PATHS[VIDEO_IDX])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                VIDEO_IDX = (VIDEO_IDX + 1) % len(VIDEO_PATHS)
                cap, fps, delay = _init_video_object(VIDEO_PATHS[VIDEO_IDX])
                continue

            start = time.time()
            yield frame
            if paced:
                elapsed = time.time() - start
                time.sleep(max(0, delay - elapsed))
    finally:
        cap.release()


def _init_video_object(path: str) -> tuple[cv2.VideoCapture, float, float]:
    """Open a video file; raises RuntimeError (after releasing) if it cannot be opened."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        logger.error(f"Failed to open static video: {path}")
        raise RuntimeError(f"Failed to open static video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1 / fps
    return (cap, fps, delay)


def live_stream():
    # Note: the clip recorder and /video_feed each open their own decoder/camera
    # handle; a shared single capture is deliberately not implemented.
    cap = cv2.VideoCapture(0)
    try:
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
    finally:
        cap.release()


def get_stream_source(paced: bool = True):
    """Pick the active source. `paced` only affects static video: the clip recorder
    passes False because it paces frames itself; live_stream is paced by the camera."""
    return live_stream() if USE_LIVE_CAMERA['value'] else static_video(paced=paced)


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
