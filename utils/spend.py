"""Spend control: Run / Pause / Step, pacing, the budget and the auto-pause rules (design: spend-protocol).

Not wired into the app yet (T87). The app starts paused. Run, Pause and Step always apply; pacing and the budget
only for rounds with a remote agent (the locality rule: remote means paid). One budget: `max_rounds` rounds and
`max_usd` of priced calls, whichever comes first. Run resumes what is left, Step uses it, and only `reset_budget`
refills it. A round already running may overshoot a cap.

Auto-pause: the budget is used up; the last browser disconnected; an authentication error (at once); three failures
in a row from one slot. The caller asks `admit` before each round and reports every agent outcome to `record`.
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

MAX_ROUNDS = 50
MAX_USD = 0.50
MIN_PAID_INTERVAL_S = 5.0
MAX_FAILURES_IN_A_ROW = 3
ACTIONS = ("run", "pause", "step", "reset_budget")

# Why the controller is paused; shown in the UI.
PAUSE_REASONS = {
    "start": "Paused: press Run or Step",
    "user": "Paused",
    "step": "Paused after one step",
    "budget": "Paused: budget used up; reset it to continue",
    "disconnect": "Paused: no browser connected",
    "auth": "Paused: authentication failed; check the key",
    "failures": "Paused: one agent failed three times in a row",
}


@dataclass(frozen=True)
class Admission:
    start: bool
    reason: str | None = None  # why not: paused | pacing | budget


class SpendController:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_rounds: int = MAX_ROUNDS,
        max_usd: float = MAX_USD,
        min_interval_s: float = MIN_PAID_INTERVAL_S,
        max_failures: int = MAX_FAILURES_IN_A_ROW,
    ):
        self._clock = clock
        self.max_rounds = max_rounds
        self.max_usd = max_usd
        self.min_interval_s = min_interval_s
        self.max_failures = max_failures
        self._lock = threading.Lock()
        self._running = False
        self._step_pending = False
        self._pause_reason: str | None = "start"
        self._pause_detail: str | None = None
        self._last_paid_start: float | None = None
        self._failures: dict[str, int] = {}
        self._reset_budget()

    def _reset_budget(self) -> None:
        self.rounds_used = 0
        self.usd_spent = 0.0
        self.paid_calls = 0
        self.unpriced_calls = 0

    # -- state -------------------------------------------------------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        return self.rounds_used >= self.max_rounds or self.usd_spent >= self.max_usd

    @property
    def running(self) -> bool:
        return self._running

    def _pause(self, reason: str, detail: str | None = None) -> None:
        self._running = False
        self._step_pending = False
        self._pause_reason = reason
        self._pause_detail = detail

    def snapshot(self) -> dict:
        """For the `state` frame."""
        with self._lock:
            wait = 0.0
            if self._last_paid_start is not None:
                wait = max(0.0, self.min_interval_s - (self._clock() - self._last_paid_start))
            return {
                "running": self._running,
                "step_pending": self._step_pending,
                "pause_reason": None if self._running else self._pause_reason,
                "pause_text": None if self._running else PAUSE_REASONS.get(self._pause_reason or "", None),
                "pause_slot": None if self._running else self._pause_detail,
                "rounds_used": self.rounds_used,
                "max_rounds": self.max_rounds,
                "usd_spent": round(self.usd_spent, 6),
                "max_usd": self.max_usd,
                "paid_calls": self.paid_calls,
                "unpriced_calls": self.unpriced_calls,
                "exhausted": self.exhausted,
                "next_paid_round_in_s": round(wait, 1),
            }

    # -- user actions -------------------------------------------------------------------------------------------------

    def control(self, action: str, remote: bool) -> str | None:
        """Apply a user action. `remote`: a remote agent is enabled. Returns why it was refused, or None."""
        with self._lock:
            if action == "pause":
                self._pause("user")
                return None
            if action == "reset_budget":
                self._reset_budget()
                return None
            if action not in ("run", "step"):
                return f"Unknown action {action!r}"
            if remote and self.exhausted:
                return PAUSE_REASONS["budget"]
            if action == "run":
                self._running, self._step_pending, self._pause_reason, self._pause_detail = True, False, None, None
            elif self._running:
                return "Already running: pause first to step"
            else:
                self._step_pending = True
            self._failures.clear()  # the user chose to try again
            return None

    def disconnected(self, remaining: int) -> None:
        """A browser left; with none left, pause."""
        if remaining <= 0:
            with self._lock:
                if self._running or self._step_pending:
                    self._pause("disconnect")

    # -- rounds -------------------------------------------------------------------------------------------------------

    def admit(self, remote: bool) -> Admission:
        """May a round start now? If so, it counts as started. `remote`: the round has a remote agent."""
        with self._lock:
            if not self._running and not self._step_pending:
                return Admission(False, "paused")
            if remote:
                if self.exhausted:
                    self._pause("budget")
                    return Admission(False, "budget")
                now = self._clock()
                if self._last_paid_start is not None and now - self._last_paid_start < self.min_interval_s:
                    return Admission(False, "pacing")
                self._last_paid_start = now
                self.rounds_used += 1
            if self._step_pending:
                self._pause("step")
            return Admission(True)

    def record(self, slot: str, status: str, remote: bool, cost_usd: float | None, error_kind: str | None) -> None:
        """One agent outcome, stale and late ones included (they were sent, so they may have cost money)."""
        if status == "skipped":
            return
        with self._lock:
            if remote:
                self.paid_calls += 1
                if cost_usd is not None:
                    self.usd_spent += cost_usd
                elif status == "ok":
                    self.unpriced_calls += 1
            if status == "ok":
                self._failures[slot] = 0
            else:
                self._failures[slot] = self._failures.get(slot, 0) + 1
            stopped = not self._running and not self._step_pending
            if error_kind == "auth":
                self._pause("auth", slot)
            elif self._failures[slot] >= self.max_failures and not stopped:
                self._pause("failures", slot)
            elif remote and self.usd_spent >= self.max_usd and not stopped:
                self._pause("budget")
