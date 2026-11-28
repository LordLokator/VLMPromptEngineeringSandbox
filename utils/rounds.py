"""One round: the same clip and request to every enabled agent in parallel, each result reported as it arrives.

Not wired into the app yet (T87). Lockstep: a round ends when every agent has answered, failed or timed out, and
only then is the clip file deleted (once). One agent's error or timeout never affects the other.

A call that blocks inside an SDK cannot be killed. Each call runs on a daemon thread: after `call_timeout_s` the
round reports it as a timeout and moves on, a late answer is dropped, and a stuck thread never blocks shutdown.
Results from an earlier source (`source_epoch`) are still reported, marked `stale` ("previous source"), because the
call was made and may have cost money. A clip whose source is already stale when a worker takes it is not sent.
"""
import itertools
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from utils import telemetry
from utils.errors import classify_error, log_failure
from utils.providers import MAX_OUTPUT_TOKENS, SYSTEM_PROMPT, GenerateRequest, Result
from utils.providers.registry import Slot
from video_handling.clip_handling import ClipSample

# Above every provider's own timeout (30 s, Ollama 120 s; SDK retries are off), so normally the provider ends a slow
# call itself and this limit only catches a call that is stuck.
CALL_TIMEOUT_S = 130.0
STATUSES = ("ok", "error", "timeout", "skipped")


@dataclass(frozen=True)
class RoundSettings:
    """What the round sends, read once when the round starts."""

    prompt: str
    temperature: float | None
    resample_fps: int
    clip_fps: int
    max_frames: int = 20
    max_output_tokens: int = MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class AgentOutcome:
    round_id: int
    clip_id: int
    source_epoch: int
    slot: str
    label: str
    remote: bool
    status: str  # one of STATUSES; skipped: never sent
    stale: bool  # the source changed before the answer came: "previous source", still counted
    result: Result | None  # set when status is ok
    error_kind: str | None  # utils.errors kinds
    error_message: str | None  # safe for the UI
    elapsed_ms: float  # the runner's own wall clock, also for failures


@dataclass(frozen=True)
class RoundReport:
    round_id: int
    clip: ClipSample
    outcomes: tuple[AgentOutcome, ...]  # in arrival order


def request_for(clip: ClipSample, settings: RoundSettings) -> GenerateRequest:
    """One round's request. Every agent gets the same one, so API agents get byte-identical frames."""
    frames, times = clip.frames_at(settings.resample_fps, settings.clip_fps)
    return GenerateRequest(
        frames=frames,
        frame_times=times,
        system=SYSTEM_PROMPT,
        prompt=settings.prompt,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        video_path=clip.video_path,
        fps=settings.resample_fps,
        max_frames=settings.max_frames,
    )


class _Mailbox:
    """Where agent threads drop their outcomes. After `close`, `put` refuses, so nothing is lost in between."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: queue.Queue[AgentOutcome] = queue.Queue()
        self._closed = False

    def put(self, outcome: AgentOutcome) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._queue.put(outcome)
            return True

    def get(self, timeout: float) -> AgentOutcome:
        return self._queue.get(timeout=timeout)

    def close(self) -> list[AgentOutcome]:
        """Stop accepting; return what arrived but was not taken."""
        with self._lock:
            self._closed = True
            left = []
            while not self._queue.empty():
                left.append(self._queue.get_nowait())
            return left


def _delete(path: str) -> None:
    Path(path).unlink(missing_ok=True)


class RoundRunner:
    """Runs one round at a time. `start` returns at once; `run` blocks (tests, and the round thread itself).

    `on_round_start` gets (round id, clip, request) before any agent is asked, `on_result` each AgentOutcome as it
    arrives and `on_round_end` the RoundReport; both run on the round's
    thread and must not block for long. `on_late` gets an answer that came after its round gave up on it (on the
    agent's thread): it is not shown, but a paid call still has to be counted. `current_epoch` reads the live
    source epoch.
    """

    def __init__(
        self,
        *,
        current_epoch: Callable[[], int],
        on_round_start: Callable[[int, ClipSample, GenerateRequest], None] = lambda _i, _c, _r: None,
        on_result: Callable[[AgentOutcome], None] = lambda _o: None,
        on_round_end: Callable[[RoundReport], None] = lambda _r: None,
        on_late: Callable[[AgentOutcome], None] = lambda _o: None,
        call_timeout_s: float = CALL_TIMEOUT_S,
        delete_clip: Callable[[str], None] = _delete,
    ):
        self._current_epoch = current_epoch
        self._on_round_start = on_round_start
        self._on_result = on_result
        self._on_round_end = on_round_end
        self._on_late = on_late
        self._call_timeout_s = call_timeout_s
        self._delete_clip = delete_clip
        self._ids = itertools.count(1)
        self._busy = threading.Lock()
        self._stopping = threading.Event()

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def start(self, clip: ClipSample, slots: Sequence[Slot], settings: RoundSettings) -> bool:
        """Begin a round on a daemon thread. False (and the clip is left to the caller) when busy or stopped."""
        if self._stopping.is_set() or not self._busy.acquire(blocking=False):
            return False

        def body() -> None:
            try:
                self._run(clip, slots, settings)
            finally:
                self._busy.release()

        try:
            threading.Thread(target=body, name="round", daemon=True).start()
        except RuntimeError:  # interpreter shutting down
            self._busy.release()
            return False
        return True

    def run(self, clip: ClipSample, slots: Sequence[Slot], settings: RoundSettings) -> RoundReport | None:
        """Run a round on this thread. None if the runner is stopped or another round is running."""
        if self._stopping.is_set() or not self._busy.acquire(blocking=False):
            return None
        try:
            return self._run(clip, slots, settings)
        finally:
            self._busy.release()

    def stop(self, timeout: float = 5.0) -> bool:
        """No new rounds; wait (bounded) for the running one. False if a call is still stuck after `timeout`."""
        self._stopping.set()
        if self._busy.acquire(timeout=timeout):
            self._busy.release()
            return True
        logger.warning(f"A round was still running after {timeout:.0f}s; its calls are abandoned.")
        return False

    # -- internals ---------------------------------------------------------------------------------------------------

    def _run(self, clip: ClipSample, slots: Sequence[Slot], settings: RoundSettings) -> RoundReport | None:
        round_id = next(self._ids)
        active = [s for s in slots if s.enabled]
        outcomes: list[AgentOutcome] = []
        try:
            request = request_for(clip, settings)
            try:
                self._on_round_start(round_id, clip, request)
            except Exception as e:  # noqa: BLE001 - a broken listener must not stop the rounds
                log_failure("Round start listener failed", e)
            inbox = _Mailbox()  # closed when the round stops listening; later answers go to on_late
            for slot in active:
                threading.Thread(
                    target=self._call, args=(round_id, clip, slot, request, inbox), name=f"agent-{slot.name}",
                    daemon=True,
                ).start()

            deadline = time.monotonic() + self._call_timeout_s
            started = time.perf_counter()
            pending = {s.name: s for s in active}
            while pending and not self._stopping.is_set():
                try:
                    outcome = inbox.get(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
                except queue.Empty:
                    if time.monotonic() < deadline:
                        continue
                    for slot in pending.values():
                        self._report(outcomes, self._timed_out(round_id, clip, slot, started))
                    pending.clear()
                    break
                pending.pop(outcome.slot, None)
                self._report(outcomes, outcome)
            for left in inbox.close():  # landed after the last look: late, never lost
                self._late(left)
            if self._stopping.is_set():
                return None
            report = RoundReport(round_id, clip, tuple(outcomes))
            try:
                self._on_round_end(report)
            except Exception as e:  # noqa: BLE001 - a broken listener must not stop the rounds
                log_failure("Round end listener failed", e)
            return report
        finally:
            self._delete_clip(clip.video_path)

    def _report(self, outcomes: list[AgentOutcome], outcome: AgentOutcome) -> None:
        outcomes.append(outcome)
        try:
            self._on_result(outcome)
        except Exception as e:  # noqa: BLE001 - a broken listener must not stop the rounds
            log_failure("Result listener failed", e)

    def _late(self, outcome: AgentOutcome) -> None:
        logger.info(f"Agent {outcome.slot} answered after its round gave up on it ({outcome.status}).")
        try:
            self._on_late(outcome)
        except Exception as e:  # noqa: BLE001 - a broken listener must not kill the agent thread
            log_failure("Late result listener failed", e)

    def _timed_out(self, round_id: int, clip: ClipSample, slot: Slot, started: float) -> AgentOutcome:
        logger.warning(f"Agent {slot.name} ({slot.agent.label}) gave no answer within {self._call_timeout_s:.0f}s.")
        return AgentOutcome(
            round_id, clip.clip_id, clip.source_epoch, slot.name, slot.agent.label, slot.remote, "timeout",
            self._current_epoch() != clip.source_epoch, None, "timeout",
            f"No answer within {self._call_timeout_s:.0f} s", (time.perf_counter() - started) * 1000,
        )

    def _call(
        self, round_id: int, clip: ClipSample, slot: Slot, request: GenerateRequest, inbox: _Mailbox
    ) -> None:
        def outcome(status: str, result: Result | None = None, kind: str | None = None,
                    message: str | None = None, stale: bool = False) -> AgentOutcome:
            return AgentOutcome(
                round_id, clip.clip_id, clip.source_epoch, slot.name, slot.agent.label, slot.remote, status, stale,
                result, kind, message, (time.perf_counter() - start) * 1000,
            )

        def deliver(item: AgentOutcome) -> None:
            if not inbox.put(item):
                self._late(item)

        start = time.perf_counter()
        if self._current_epoch() != clip.source_epoch:
            deliver(outcome("skipped", message="The source changed before this agent was asked"))
            return
        try:
            with telemetry.inference(slot.agent.label, request.fps, request.max_frames, request.temperature or 0.0):
                result = slot.agent.generate(request)
        except Exception as e:  # noqa: BLE001 - worker boundary: any provider failure becomes an error result
            message = log_failure(f"Agent {slot.name} ({slot.agent.label}) failed", e)
            kind, _ = classify_error(e)
            status = "timeout" if kind == "timeout" else "error"
            deliver(outcome(status, kind=kind, message=message, stale=self._current_epoch() != clip.source_epoch))
            return
        deliver(outcome("ok", result=result, stale=self._current_epoch() != clip.source_epoch))
