# clip_handling.py

import threading
import time
from collections import deque
from typing import Iterator
import numpy as np
from loguru import logger

from video_hanlding.write_video import write_video
from streaming import get_stream_source


class ClipRecorder:
    def __init__(self, buffer_seconds: int = 3, fps: int = 10):
        self.buffer_size = buffer_seconds * fps
        self.buffer = deque(maxlen=self.buffer_size)
        self.fps = fps
        self.out_dir = "./tmp"
        self.running = False
        self.counter = 0

    def _run(self, every_seconds: int):
        logger.info("ClipRecorder started.")
        last_flush = time.time()
        current_stream = get_stream_source()

        while self.running:
            try:
                frame = next(current_stream)
            except StopIteration:
                current_stream = get_stream_source()  # 🔄 re-fetch stream
                continue

            self.buffer.append(frame)

            # Detect and respond to stream type change
            new_stream = get_stream_source()
            if new_stream is not current_stream:
                logger.info("Stream source changed. Switching.")
                current_stream = new_stream
                continue

            if time.time() - last_flush >= every_seconds:
                last_flush = time.time()
                if len(self.buffer) >= self.fps:
                    self._flush_clip()

    def start(self, frame_stream: Iterator[np.ndarray], every_seconds: int = 10):
        self.running = True
        threading.Thread(
            target=self._run,
            args=(frame_stream, every_seconds),
            daemon=True
        ).start()

    def _flush_clip(self):
        frames = list(self.buffer)
        filename = f"{self.out_dir}/clip_{self.counter:03d}.mp4"

        self.counter += 1
        logger.debug(f"Saving {len(frames)} buffered frames to [{filename}].")
        write_video(frames, filename, fps=self.fps)

    def stop(self):
        self.running = False
