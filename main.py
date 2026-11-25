# main.py
import sys
import threading
import time
import timeit
from functools import partial
from pathlib import Path
from typing import Callable, Protocol
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from prompting.prompting import Prompt, RuntimeSettings
from utils.vlm_wrapper import VLM
from web_engine import start_server_threaded, broadcast_sync
from video_handling.clip_handling import ClipRecorder

logger.remove()
logger.add(sys.stderr, level="DEBUG")

FPS = 10
SEC = 0.5 # Buffer in seconds
MAX_FRAMES = 20


class VLMLike(Protocol):
    def forward(self, video_path: str, fps: int, max_frames: int, temperature: float) -> tuple[str, float]: ...


def process_clip(
    clip_path: str,
    vlm: VLMLike,
    broadcast: Callable[[dict], None],
    *,
    get_temperature: Callable[[], float],
    get_fps: Callable[[], int],
    max_frames: int,
) -> None:
    try:
        start = timeit.default_timer()

        output, _temperature = vlm.forward(
            video_path=clip_path,
            fps=get_fps(),
            max_frames=max_frames,
            temperature=get_temperature()
        )

        delay = timeit.default_timer() - start

        _message = f"VLM Response (t: {_temperature:.2f} | Δ: {delay:.2f}s): {output}"
        logger.info(_message)

        # Send to frontend
        msg = {
            'output': output,
            'delay': delay,
            'status': 'OK'
        }
        broadcast(msg)

    except Exception as e:
        err = f"Failed to process: {e}"
        logger.error(err)
        msg = {
            'status': 'ERROR',
            'msg': err
        }
        broadcast(msg)

    finally:
        # README promises clips are deleted after inference.
        Path(clip_path).unlink(missing_ok=True)


def make_clip_handler(
    process: Callable[[str], None],
    executor: ThreadPoolExecutor,
    busy: threading.Lock,
) -> Callable[[str], None]:
    """Return a flush callback that drops (and deletes) clips while inference is in flight."""
    dropped = 0

    def run(clip_path: str) -> None:
        try:
            process(clip_path)
        finally:
            busy.release()

    def on_clip_flush(clip_path: str) -> None:
        nonlocal dropped
        if not busy.acquire(blocking=False):
            dropped += 1
            logger.warning(f"Inference busy, dropping clip (dropped so far: {dropped}): {clip_path}")
            Path(clip_path).unlink(missing_ok=True)
            return
        logger.info(f"Clip flushed: {clip_path}")
        try:
            executor.submit(run, clip_path)
        except RuntimeError:
            # Executor already shut down.
            busy.release()
            Path(clip_path).unlink(missing_ok=True)

    return on_clip_flush


def main():
    logger.info("Main app initialization...")

    # Create shared Prompt object
    prompt = Prompt()
    settings = RuntimeSettings(clip_fps=FPS)

    # Start server in separate thread
    start_server_threaded(prompt, settings)

    # Create Video stream
    recorder = ClipRecorder(
        buffer_seconds=SEC,
        fps=FPS
    )

    # Instantiate VLM
    vlm = VLM(prompt)

    executor = ThreadPoolExecutor(max_workers=1)
    process = partial(
        process_clip,
        vlm=vlm,
        broadcast=broadcast_sync,
        get_temperature=settings.get_temperature,
        get_fps=settings.get_resample_fps,
        max_frames=MAX_FRAMES,
    )

    on_clip_flush = make_clip_handler(process, executor, threading.Lock())

    recorder.start(flush_callback=on_clip_flush)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exited with KeyboardInterrupt.")
        executor.shutdown(wait=False, cancel_futures=True)
        recorder.stop()


if __name__ == "__main__":
    main()
