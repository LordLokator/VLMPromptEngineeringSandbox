"""Fake agents for tests: a `Provider` with scripted replies, delays, failures and hangs. No model, no network.

Use the `make_agent` fixture (tests/conftest.py): it releases every hanging agent at teardown. Fake usage and cost
carry `cost_source="fake"`, so they can never pass for real numbers.
"""
import threading
import time
from collections.abc import Callable, Iterable, Sequence

from utils.providers import GenerateRequest, ModelInfo, Result

FAKE_COST_SOURCE = "fake"
HANG_LIMIT_S = 30.0  # a hang the test forgot to release ends here, so a broken test fails instead of blocking CI


class FakeAgent:
    """Calls are numbered from 1. Per call, in order: record the request, sleep `delay`, hang if asked, then raise
    the exception scripted for that call or return the next reply (the last reply repeats).

    `completions` (shared between agents if passed in) records `(label, call number)` when a call returns or raises,
    so a test can check the order in which agents finished.
    """

    def __init__(
        self,
        provider: str = "demo",
        model: str = "fake",
        *,
        replies: Sequence[str] = ("fake answer",),
        delay: float = 0.0,
        sleep: Callable[[float], object] = time.sleep,
        raise_on: dict[int, BaseException] | None = None,
        hang_on: Iterable[int] = (),
        usage: tuple[int, int] | None = None,
        cost: float | None = None,
        models: Sequence[ModelInfo] = (),
        list_error: BaseException | None = None,
        completions: list[tuple[str, int]] | None = None,
    ):
        if not replies:
            raise ValueError("replies must not be empty")
        self.provider = provider
        self.model = model
        self.replies = list(replies)
        self.delay = delay
        self.sleep = sleep
        self.raise_on = dict(raise_on or {})
        self.hang_on = set(hang_on)
        self.usage = usage
        self.cost = cost
        self.models = list(models)
        self.list_error = list_error
        self.completions = completions if completions is not None else []
        self.requests: list[GenerateRequest] = []
        self.list_calls = 0
        self.release = threading.Event()  # ends every hang, now and later
        self.started = threading.Event()  # set when any call begins
        self._lock = threading.Lock()

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def calls(self) -> int:
        with self._lock:
            return len(self.requests)

    def list_models(self) -> list[ModelInfo]:
        with self._lock:
            self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return list(self.models)

    def generate(self, request: GenerateRequest) -> Result:
        with self._lock:
            self.requests.append(request)
            number = len(self.requests)
        self.started.set()
        start = time.perf_counter()
        try:
            if self.delay:
                self.sleep(self.delay)
            if number in self.hang_on and not self.release.wait(HANG_LIMIT_S):
                raise RuntimeError(f"{self.label}: call {number} was never released")
            if number in self.raise_on:
                raise self.raise_on[number]
            text = self.replies[min(number, len(self.replies)) - 1]
        finally:
            with self._lock:
                self.completions.append((self.label, number))
        input_tokens, output_tokens = self.usage if self.usage is not None else (None, None)
        return Result(
            text=text,
            finish_reason="stop",
            requested_model=self.model,
            reported_model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=None,
            reasoning_tokens=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=self.cost,
            cost_source=FAKE_COST_SOURCE if self.cost is not None else None,
            params_sent={"fake": True},
        )


class AgentFactory:
    """Makes fake agents and remembers them, so `release_all` can end every hang (the fixture's teardown)."""

    def __init__(self) -> None:
        self.made: list[FakeAgent] = []

    def __call__(self, *args, **kwargs) -> FakeAgent:
        agent = FakeAgent(*args, **kwargs)
        self.made.append(agent)
        return agent

    def release_all(self) -> None:
        for agent in self.made:
            agent.release.set()
