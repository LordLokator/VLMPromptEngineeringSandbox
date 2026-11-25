# main.py
import signal
import sys
import threading
import timeit
from functools import partial
from pathlib import Path
from typing import Callable, Protocol
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from prompting.prompting import Prompt, RuntimeSettings
from utils import telemetry
from utils.backends import make_backend
from web_engine import start_server_threaded, broadcast_sync, emit
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

        fps, temperature = get_fps(), get_temperature()
        backend = getattr(vlm, "name", type(vlm).__name__)
        with telemetry.inference(backend, fps, max_frames, temperature) as set_backend:
            output, _temperature = vlm.forward(
                video_path=clip_path,
                fps=fps,
                max_frames=max_frames,
                temperature=temperature
            )
            # A fallback chain only knows which backend answered after forward() returns.
            set_backend(getattr(vlm, "name", backend))

        delay = timeit.default_timer() - start

        _message = f"VLM Response (t: {_temperature:.2f} | Δ: {delay:.2f}s): {output}"
        logger.info(_message)

        # Send to frontend
        broadcast(emit("vlm", output, delay=delay))

    except Exception as e:  # noqa: BLE001 - worker boundary: logged and reported to the UI
        # Inference can fail in many ways (CUDA, decoding, model); the worker must survive and tell the UI.
        logger.exception("Failed to process clip")
        broadcast(emit("error", f"Failed to process: {e}"))

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
            telemetry.record_drop()
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


def shutdown(recorder, executor: ThreadPoolExecutor, server, busy: threading.Lock, timeout: float = 5.0) -> None:
    """Stop capturing first, let the in-flight inference finish (bounded), then stop the web server."""
    logger.info("Shutting down...")
    recorder.stop()
    executor.shutdown(wait=False, cancel_futures=True)
    if busy.acquire(timeout=timeout):
        busy.release()
    else:
        logger.warning(f"Inference still running after {timeout:.0f}s, exiting anyway.")
    server.should_exit = True


def main():
    logger.info("Main app initialization...")

    # Create shared Prompt object
    prompt = Prompt()
    settings = RuntimeSettings(clip_fps=FPS)

    # Start server in separate thread
    server = start_server_threaded(prompt, settings)

    # Create Video stream
    recorder = ClipRecorder(
        buffer_seconds=SEC,
        fps=FPS
    )

    # Instantiate the backend (VLM_BACKEND, default: local HF model)
    vlm = make_backend(prompt)

    executor = ThreadPoolExecutor(max_workers=1)
    process = partial(
        process_clip,
        vlm=vlm,
        broadcast=broadcast_sync,
        get_temperature=settings.get_temperature,
        get_fps=settings.get_resample_fps,
        max_frames=MAX_FRAMES,
    )

    busy = threading.Lock()
    on_clip_flush = make_clip_handler(process, executor, busy)

    recorder.start(flush_callback=on_clip_flush, on_error=lambda text: broadcast_sync(emit("error", text)))

    stop_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        logger.info(f"Received signal {signum}, stopping.")
        stop_event.set()

    for name in ("SIGINT", "SIGTERM"):  # SIGTERM semantics differ on Windows; skip if absent
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    # Short timeouts keep the main thread responsive to signals on every platform.
    while not stop_event.wait(0.5):
        pass

    shutdown(recorder, executor, server, busy)


if __name__ == "__main__":
    main()
