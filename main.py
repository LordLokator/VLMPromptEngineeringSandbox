# main.py
import signal
import sys
import threading
import time
import timeit
from functools import partial
from pathlib import Path
from typing import Callable
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from prompting.prompting import Prompt, RuntimeSettings
from utils import telemetry
from utils.errors import log_failure
from utils.logging_setup import configure_logging, route_stdlib_logging
from utils.providers import GenerateRequest, Provider
from utils.providers.agents import AgentConfigError, agent_spec_from_env, build_agent
from utils.providers.settings import load_env_file
from utils.rounds import RoundSettings, request_for
from web_engine import start_server_threaded, broadcast_sync, emit
from video_handling.clip_handling import ClipRecorder, ClipSample
from video_handling.streaming import CaptureHub

FPS = 10
SEC = 0.5 # Buffer in seconds
MAX_FRAMES = 20


def build_request(
    clip: ClipSample, *, prompt: str, temperature: float, fps: int, max_frames: int, clip_fps: int = FPS
) -> GenerateRequest:
    """One round's request. Every agent gets the same one, so API agents get byte-identical frames."""
    return request_for(clip, RoundSettings(prompt, temperature, fps, clip_fps, max_frames))


def process_clip(
    clip: ClipSample,
    agent: Provider,
    broadcast: Callable[[dict], None],
    *,
    get_prompt: Callable[[], str],
    get_temperature: Callable[[], float],
    get_fps: Callable[[], int],
    get_source_epoch: Callable[[], int],
    max_frames: int,
) -> None:
    """Ask the agent about one clip. Nothing is broadcast if the source changed in the meantime."""
    try:
        start = timeit.default_timer()

        fps, temperature = get_fps(), get_temperature()
        request = build_request(clip, prompt=get_prompt(), temperature=temperature, fps=fps, max_frames=max_frames)
        with telemetry.inference(agent.label, fps, max_frames, temperature):
            result = agent.generate(request)

        delay = timeit.default_timer() - start

        logger.info(f"{agent.label} answered (t: {temperature:.2f} | Δ: {delay:.2f}s): {result.text}")

        # Send to frontend
        if clip.source_epoch != get_source_epoch():
            logger.info(f"Dropped the answer for clip {clip.clip_id}: it shows an earlier source ({clip.label}).")
            return
        broadcast(emit("vlm", result.text, delay=delay))

    except Exception as e:  # noqa: BLE001 - worker boundary: logged and reported to the UI
        # Inference can fail in many ways (CUDA, decoding, model, network); the worker must survive and tell the UI.
        message = log_failure("Failed to process clip", e)
        if clip.source_epoch == get_source_epoch():
            broadcast(emit("error", f"Failed to process: {message}"))

    finally:
        # README promises clips are deleted after inference.
        Path(clip.video_path).unlink(missing_ok=True)


def make_clip_handler(
    process: Callable[[ClipSample], None],
    executor: ThreadPoolExecutor,
    busy: threading.Lock,
    notify: Callable[[dict], None] | None = None,
    notice_interval: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[ClipSample], None]:
    """Return a flush callback that drops (and deletes) clips while inference is in flight.

    On a drop, `notify` gets a muted system frame, at most once per `notice_interval` seconds.
    """
    dropped = 0
    last_notice = float("-inf")

    def run(clip: ClipSample) -> None:
        try:
            process(clip)
        finally:
            busy.release()

    def on_clip_flush(clip: ClipSample) -> None:
        nonlocal dropped, last_notice
        clip_path = clip.video_path
        if not busy.acquire(blocking=False):
            dropped += 1
            telemetry.record_drop()
            logger.warning(f"Inference busy, dropping clip (dropped so far: {dropped}): {clip_path}")
            Path(clip_path).unlink(missing_ok=True)
            now = clock()
            if notify is not None and now - last_notice >= notice_interval:
                last_notice = now
                notify(emit("system", "Model busy, skipped a clip", muted=True))
            return
        logger.info(
            f"Clip {clip.clip_id} flushed: {clip.source_key} (epoch {clip.source_epoch}, loop {clip.loop}), "
            f"{clip.start_s:.1f}-{clip.end_s:.1f} s: {clip_path}"
        )
        try:
            executor.submit(run, clip)
        except RuntimeError:
            # Executor already shut down.
            busy.release()
            Path(clip_path).unlink(missing_ok=True)

    return on_clip_flush


def shutdown(
    hub, recorder, executor: ThreadPoolExecutor | None, server, busy: threading.Lock, timeout: float = 5.0
) -> None:
    """Stop capturing first, let the in-flight inference finish (bounded), then stop the web server.

    `executor` is None when a stop signal arrived before the agent finished loading.
    """
    logger.info("Shutting down...")
    recorder.stop()
    hub.stop()  # also ends the preview streams
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    if busy.acquire(timeout=timeout):
        busy.release()
    else:
        logger.warning(f"Inference still running after {timeout:.0f}s, exiting anyway.")
    server.should_exit = True


def main():
    load_env_file()  # keys and URLs from .env; real environment variables win
    configure_logging()
    logger.info("Main app initialization...")

    # Check the agent settings before anything starts, so a typo ends the app with one clear line.
    try:
        agent_spec = agent_spec_from_env("A")
    except AgentConfigError as e:
        logger.error(str(e))
        sys.exit(2)

    # Install handlers before the slow startup (an hf model load) so Ctrl+C always reaches a clean shutdown.
    # The handler only sets the event; startup checks it between stages.
    stop_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        logger.info(f"Received signal {signum}, stopping.")
        stop_event.set()

    for name in ("SIGINT", "SIGTERM"):  # SIGTERM semantics differ on Windows; skip if absent
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    # Create shared Prompt object
    prompt = Prompt()
    settings = RuntimeSettings(clip_fps=FPS, clip_seconds=SEC)

    # The only capture; the preview and the clip recorder both read from it. Not started yet.
    hub = CaptureHub(min_pass_s=SEC, on_error=lambda text: broadcast_sync(emit("error", text)))

    # Start server in separate thread
    server = start_server_threaded(prompt, settings, hub)

    # Cuts clips once an agent is there to take them (not started yet)
    recorder = ClipRecorder(
        hub,
        buffer_seconds=SEC,
        fps=FPS
    )

    executor: ThreadPoolExecutor | None = None
    busy = threading.Lock()
    try:
        hub.start()  # the video plays without, and before, any agent

        # Agent A (VLM_AGENT_A, default: demo). An hf agent loads the model here, synchronously.
        agent = build_agent(agent_spec)
        logger.info(f"Agent A: {agent.label}")
        route_stdlib_logging()  # libraries imported while loading may have added their own handlers

        if not stop_event.is_set():
            executor = ThreadPoolExecutor(max_workers=1)
            process = partial(
                process_clip,
                agent=agent,
                broadcast=broadcast_sync,
                get_prompt=prompt.get,
                get_temperature=settings.get_temperature,
                get_fps=settings.get_resample_fps,
                get_source_epoch=lambda: hub.spec.source_epoch,
                max_frames=MAX_FRAMES,
            )
            on_clip_flush = make_clip_handler(process, executor, busy, notify=broadcast_sync)
            recorder.start(flush_callback=on_clip_flush)

        # Short timeouts keep the main thread responsive to signals on every platform.
        while not stop_event.wait(0.5):
            pass
    finally:
        shutdown(hub, recorder, executor, server, busy)


if __name__ == "__main__":
    main()
