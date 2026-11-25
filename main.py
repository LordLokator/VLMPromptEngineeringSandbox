# main.py
import sys
import time
import timeit
from functools import partial
from typing import Callable, Protocol
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from prompting.prompting import Prompt
from utils.vlm_wrapper import VLM
from web_engine import start_server_threaded, broadcast_sync, TEMPERATURE
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
    fps: int,
    max_frames: int,
) -> None:
    try:
        start = timeit.default_timer()

        output, _temperature = vlm.forward(
            video_path=clip_path,
            fps=fps,
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


def main():
    logger.info("Main app initialization...")

    # Create shared Prompt object
    prompt = Prompt()

    # Start server in separate thread
    start_server_threaded(prompt)

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
        # Read at call time; note TEMPERATURE itself is bound at import (known bug, fixed in T31).
        get_temperature=lambda: TEMPERATURE,
        fps=FPS,
        max_frames=MAX_FRAMES,
    )

    def on_clip_flush(clip_path: str):
        logger.info(f"Clip flushed: {clip_path}")
        executor.submit(process, clip_path)

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
