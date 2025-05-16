# clip_handling.py

import glob
import os
import threading
import time
from collections import deque
from typing import Callable
from loguru import logger

from utils.file_management import delete_files, full_path
from video_hanlding.write_video import write_video
from streaming import get_stream_source


class ClipRecorder:
    def __init__(self, buffer_seconds: int = 3, fps: int = 10):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.out_dir = "./tmp"
        self.running = False
        self.counter = 0
        self.flush_callback: Callable = lambda path: print(path)

        # Create './tmp' if it doesn't exists:
        os.makedirs(self.out_dir, exist_ok=True)

        # If folder not empty, clear.
        # Should be something like __exit__ but unsure if app runs without erros.
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))

    def _run(self, every_seconds: int):
        logger.info("ClipRecorder started.")
        stream = get_stream_source()

        while self.running:
            frames = []
            target_frames = int(self.buffer_seconds * self.fps)
            frame_interval = 1 / self.fps

            for _ in range(target_frames):
                try:
                    frame = next(stream)
                except StopIteration:
                    logger.warning("Stream ended, restarting...")
                    stream = get_stream_source()
                    continue

                frames.append(frame.copy())
                time.sleep(frame_interval)

            filename = f"{self.out_dir}/clip_{self.counter:03d}.mp4"
            self.counter += 1
            write_video(frames, filename, fps=self.fps)

            if os.path.exists(filename):
                self.flush_callback(full_path(filename))

            # Wait until next interval round
            total_wait = every_seconds - self.buffer_seconds
            if total_wait > 0:
                time.sleep(total_wait)

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

    def stop(self):
        self.running = False
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))
