# clip_handling.py

import glob
import os
import threading
import time
from collections import deque
from typing import Callable
from loguru import logger

from utils.file_management import delete_files, full_path
from video_handling.write_video import write_video
from video_handling.streaming import get_stream_source


class ClipRecorder:
    def __init__(self, buffer_seconds: int = 3, fps: int = 10):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.out_dir = "./tmp"
        self.running = False
        self.counter = 0
        self.flush_callback: Callable = lambda path: print(f"!!! {path} !!!") # for debugging purposes

        # Create './tmp' if it doesn't exists:
        os.makedirs(self.out_dir, exist_ok=True)
        # ... clean if it does:
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))

    def _run(self):
        logger.info("ClipRecorder started.")
        stream = get_stream_source()

        while self.running:
            frames = []
            target_frames = int(self.buffer_seconds * self.fps)
            frame_interval = 1 / self.fps

            for _ in range(target_frames):
                try:
                    frame = next(stream)
                    frames.append(frame.copy())
                except StopIteration:
                    logger.warning("Stream ended, restarting...")
                    stream = get_stream_source()
                    continue

                time.sleep(frame_interval)

            filename = f"{self.out_dir}/clip_{self.counter:03d}.mp4"
            self.counter += 1

            try:
                write_video(frames, filename, fps=self.fps)
                full_pathname = full_path(filename)
                logger.debug(f"Calling flush_callback with: {full_pathname}")
                self.flush_callback(full_pathname)

            except Exception as e:
                logger.error(f"Video writing or callback failed: {e}")

    def start(
            self,
            flush_callback: Callable = lambda _: None
    ):
        self.flush_callback = flush_callback
        self.running = True

        threading.Thread(
            target=self._run,
            daemon=True
        ).start()

    def stop(self):
        self.running = False
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))
