# clip_handling.py

import glob
import os
import queue
import threading
from typing import Any, Callable
from loguru import logger

from utils.errors import log_failure
from utils.file_management import BASE_DIR, delete_files, full_path
from video_handling.sampling import GridSampler, Segment
from video_handling.streaming import CaptureHub
from video_handling.write_video import write_video

MAX_PENDING_CLIPS = 2


class ClipRecorder:
    """Cuts clips from the hub's frames by their timestamps and writes them on its own thread."""

    def __init__(self, hub: CaptureHub, buffer_seconds: float = 3, fps: int = 10):
        self.hub = hub
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.out_dir = str(BASE_DIR / "tmp")
        self.running = False
        self.counter = 0
        self._thread: threading.Thread | None = None
        self.flush_callback: Callable = lambda path: logger.debug(f"Clip flushed: {path}")
        self._sampler = GridSampler(fps, round(buffer_seconds * fps), self._enqueue)
        self._pending: queue.Queue[list[Any]] = queue.Queue(maxsize=MAX_PENDING_CLIPS)

        # Create '<repo>/tmp' if it doesn't exists:
        os.makedirs(self.out_dir, exist_ok=True)
        # ... clean if it does:
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))

    def _enqueue(self, segment: Segment, start_t: float, frames: list[Any], times: list[float]) -> None:
        """Runs on the capture thread: never blocks. If writing falls behind, the oldest waiting clip goes."""
        while True:
            try:
                self._pending.put_nowait(frames)
                return
            except queue.Full:
                try:
                    self._pending.get_nowait()
                    logger.warning("Clip writing is falling behind, dropped a clip.")
                except queue.Empty:
                    pass

    def _run(self):
        logger.info("ClipRecorder started.")
        try:
            while self.running:
                try:
                    frames = self._pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not self.running:
                    break

                filename = f"{self.out_dir}/clip_{self.counter:03d}.mp4"
                self.counter += 1

                try:
                    write_video(frames, filename, fps=self.fps)
                    full_pathname = full_path(filename)
                    logger.debug(f"Calling flush_callback with: {full_pathname}")
                    self.flush_callback(full_pathname)

                except Exception as e:  # noqa: BLE001 - worker boundary: logged, the recorder keeps running
                    log_failure("Video writing or callback failed", e)
        finally:
            logger.info("ClipRecorder stopped.")

    def start(self, flush_callback: Callable = lambda _: None):
        self.flush_callback = flush_callback
        self.running = True

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self._thread.start()
        self.hub.add_sink(self._sampler)

    def stop(self, timeout: float = 2.0):
        """Stop cutting clips, wait for the writer thread, then delete leftover clips."""
        self.hub.remove_sink(self._sampler)
        self.running = False
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("ClipRecorder thread did not stop within the timeout.")
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))
