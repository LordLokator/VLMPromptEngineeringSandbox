import os
import cv2
from loguru import logger
import numpy as np

from utils.file_management import full_path


def write_video(frames: list[np.ndarray], video_path: str, fps: int = 60):
    """Write frames to a video file (synchronously)."""
    video_path = full_path(video_path)

    try:
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),
            fps,
            (frames[0].shape[1], frames[0].shape[0])
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

        logger.debug(f"Saved video: {os.path.basename(video_path)}")
        return True

    except Exception as e:
        logger.error(f"{type(e).__name__} at Video Writing")
        raise


MAX_EDGE = 768  # long edge of the frames sent to agents (design: capture)
JPEG_QUALITY = 80


def encode_jpeg(frame: np.ndarray, max_edge: int = MAX_EDGE, quality: int = JPEG_QUALITY) -> bytes:
    """One frame as JPEG bytes, scaled down (never up) so its long edge is at most `max_edge`."""
    height, width = frame.shape[:2]
    scale = max_edge / max(height, width)
    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()
