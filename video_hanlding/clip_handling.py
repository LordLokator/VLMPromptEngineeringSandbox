# clip_handling.py

import os
import threading
import time
from collections import deque
from typing import Callable
from loguru import logger

from utils.file_management import full_path
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

        # Create './tmp' if it doesn't exists:
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def _run(self, every_seconds: int):
        logger.info("ClipRecorder started.")
        last_flush = time.time()
        current_stream = get_stream_source()

        while self.running:
            try:
                frame = next(current_stream)
            except StopIteration:
                current_stream = get_stream_source()
                continue

            self.buffer.append(frame)

            # Detect and respond to stream type change
            current_stream = get_stream_source()

            if time.time() - last_flush >= every_seconds:
                last_flush = time.time()
                if len(self.buffer) >= self.fps:
                    self._flush_clip()

    def start(
            self,
            every_seconds: int = 10,
            flush_callback: Callable = lambda _: None
    ):
        self.flush_callback = flush_callback
        self.running = True
        threading.Thread(
            target=self._run,
            args=(every_seconds,),
            daemon=True
        ).start()

    def _flush_clip(self):
        frames = list(self.buffer)
        filename = f"{self.out_dir}/clip_{self.counter:03d}.mp4"

        self.counter += 1
        logger.debug(f"Saving {len(frames)} buffered frames to [{filename}].")
        write_video(frames, filename, fps=self.fps)
        self.flush_callback(full_path(filename))

    def stop(self):
        self.running = False
