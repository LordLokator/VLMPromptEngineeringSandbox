# clip_handling.py

import glob
import os
import threading
import time
from typing import Callable
from loguru import logger

from utils.file_management import BASE_DIR, delete_files, full_path
from video_handling.write_video import write_video
from video_handling.streaming import USE_LIVE_CAMERA, get_stream_source


class ClipRecorder:
    def __init__(self, buffer_seconds: int = 3, fps: int = 10):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.out_dir = str(BASE_DIR / "tmp")
        self.running = False
        self.counter = 0
        self.on_error: Callable[[str], None] | None = None
        self._thread: threading.Thread | None = None
        self.flush_callback: Callable = lambda path: logger.debug(f"Clip flushed: {path}")

        # Create '<repo>/tmp' if it doesn't exists:
        os.makedirs(self.out_dir, exist_ok=True)
        # ... clean if it does:
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))

    def _open(self):
        return get_stream_source(paced=False, sample_fps=self.fps)

    def _run(self):
        logger.info("ClipRecorder started.")
        stream = None
        try:
            open_mode = USE_LIVE_CAMERA["value"]
            stream = self._open()

            while self.running:
                # Camera toggle: reopen the source at the next clip boundary.
                if USE_LIVE_CAMERA["value"] != open_mode:
                    logger.info("Capture mode changed, reopening stream.")
                    stream.close()
                    open_mode = USE_LIVE_CAMERA["value"]
                    stream = self._open()

                frames = []
                target_frames = int(self.buffer_seconds * self.fps)
                frame_interval = 1 / self.fps
                next_t = time.monotonic()

                while self.running and len(frames) < target_frames:
                    try:
                        frames.append(next(stream).copy())
                    except StopIteration:
                        logger.warning("Stream ended, restarting...")
                        stream = self._open()
                        continue
                    except RuntimeError as e:
                        logger.error(f"Capture source failed: {e}")
                        self.running = False
                        if self.on_error is not None:
                            self.on_error(f"Capture source failed: {e}")
                        return

                    # Pace once, by timestamp (the source itself is unpaced here).
                    next_t += frame_interval
                    time.sleep(max(0.0, next_t - time.monotonic()))

                if not self.running:
                    break  # stopped mid-clip: do not flush a partial clip

                filename = f"{self.out_dir}/clip_{self.counter:03d}.mp4"
                self.counter += 1

                try:
                    write_video(frames, filename, fps=self.fps)
                    full_pathname = full_path(filename)
                    logger.debug(f"Calling flush_callback with: {full_pathname}")
                    self.flush_callback(full_pathname)

                except Exception as e:  # noqa: BLE001 - worker boundary: logged, the recorder keeps running
                    logger.exception(f"Video writing or callback failed: {e}")
        finally:
            if stream is not None:
                stream.close()
            logger.info("ClipRecorder stopped.")

    def start(
            self,
            flush_callback: Callable = lambda _: None,
            on_error: Callable[[str], None] | None = None,
    ):
        self.flush_callback = flush_callback
        self.on_error = on_error
        self.running = True

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        """Stop capturing, wait for the thread (it closes the stream), then delete leftover clips."""
        self.running = False
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("ClipRecorder thread did not stop within the timeout.")
        delete_files(glob.glob(os.path.join(self.out_dir, "*.mp4")))
