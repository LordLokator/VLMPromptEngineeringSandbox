import os
import threading
import cv2
from loguru import logger
import numpy as np

from utils.file_management import full_path


def write_video(frames: list[np.ndarray], video_path: str, fps: int = 60):
    def _write_video(frames: list[np.ndarray], video_path: str, fps: int = 60):
        """Write frames to a video file in a separate thread."""

        video_path = full_path(video_path)

        try:
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                fps,
                (frames[0].shape[1], frames[0].shape[0])
            )
            for frame in frames:
                writer.write(frame)
            writer.release()

            logger.debug(f"Saved video: {os.path.basename(video_path)}")

            return True

        except Exception as e:
            print(e)
            logger.error(f"{type(e).__name__} at Video Writing!")
            raise
            # return False

    threading.Thread(
        target=_write_video,
        daemon=True,
        args=(
            frames,
            video_path,
            fps
        ),
    ).start()
