import signal
import sys
import threading

from loguru import logger

from prompting.prompting import Prompt, RuntimeSettings
from utils.logging_setup import configure_logging, route_stdlib_logging
from utils.providers import PROVIDERS, GenerateRequest
from utils.providers.agents import AgentConfigError, agent_spec_from_env
from utils.providers.registry import SLOTS, AgentRegistry
from utils.providers.settings import load_env_file
from utils.rounds import RoundSettings, request_for
from utils.session import CompareSession
from utils.spend import SpendController
from web_engine import IS_LOOPBACK, broadcast_sync, start_server_threaded
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


def shutdown(hub, recorder, session: CompareSession | None, server, timeout: float = 5.0) -> None:
    """Stop cutting clips, let the running round finish (bounded), stop capturing, then stop the web server.

    `session` is None when the app stopped before it was built.
    """
    logger.info("Shutting down...")
    recorder.stop()
    if session is not None and not session.stop(timeout):
        logger.warning(f"A round was still running after {timeout:.0f}s, exiting anyway.")
    hub.stop()  # also ends the preview streams
    server.should_exit = True


def main():
    load_env_file()  # keys and URLs from .env; real environment variables win
    configure_logging()
    logger.info("Main app initialization...")

    # Check the agent settings before anything starts, so a typo or a missing key ends the app with one clear line.
    # Paid providers are blocked on a non-loopback bind unless VLM_ALLOW_REMOTE_PAID=1.
    registry = AgentRegistry(bind_loopback=IS_LOOPBACK)
    try:
        for slot in SLOTS:
            spec = agent_spec_from_env(slot, providers=PROVIDERS)
            try:
                registry.check_spec(spec)
            except AgentConfigError as e:
                raise AgentConfigError(f"VLM_AGENT_{slot}={spec.text}: {e}") from e
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

    prompt = Prompt()
    settings = RuntimeSettings(clip_fps=FPS, clip_seconds=SEC)
    session: CompareSession | None = None

    # The only capture; the preview and the clip recorder both read from it. Not started yet.
    hub = CaptureHub(min_pass_s=SEC, on_error=lambda text: session.source_error(text) if session else None)
    session = CompareSession(
        registry=registry, spend=SpendController(), hub=hub, prompt=prompt, settings=settings,
        broadcast=broadcast_sync, clip_fps=FPS, max_frames=MAX_FRAMES,
    )

    server = start_server_threaded(session, hub)
    recorder = ClipRecorder(hub, buffer_seconds=SEC, fps=FPS)  # cuts clips once the agents are there
    exit_code = 0
    try:
        hub.start()  # the video plays without, and before, any agent

        # Agents A and B (VLM_AGENT_A / VLM_AGENT_B). With VLM_ENABLE_HF=1 the local model loads here, synchronously.
        try:
            registry.configure_from_env()  # logs "Agent A: ..." per slot
        except AgentConfigError as e:
            logger.error(str(e))
            exit_code = 2
            stop_event.set()
        route_stdlib_logging()  # libraries imported while loading may have added their own handlers
        session.broadcast_state()

        if not stop_event.is_set():
            logger.info("Ready. Analysis starts paused: press Run or Step in the page.")
            recorder.start(flush_callback=session.on_clip)

        # Short timeouts keep the main thread responsive to signals on every platform.
        while not stop_event.wait(0.5):
            pass
    finally:
        shutdown(hub, recorder, session, server)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
