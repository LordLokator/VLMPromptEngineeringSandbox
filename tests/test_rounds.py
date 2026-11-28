"""Round runner with fake agents: arrival order, failures, timeouts, stale sources, clip deletion, shutdown."""
import threading
import time

import pytest

from utils.providers.agents import AgentSpec
from utils.providers.registry import Slot
from utils.rounds import RoundRunner, RoundSettings
from video_handling.clip_handling import ClipSample

FRAMES = tuple(f"jpeg-{i}".encode() for i in range(5))
TIMES = (0.0, 0.1, 0.2, 0.3, 0.4)
SETTINGS = RoundSettings(prompt="what happens?", temperature=0.3, resample_fps=5, clip_fps=10)
SENTINEL = "sk-" + "sentinel" * 4


def clip(epoch=0, clip_id=7, path="clip-7.mp4"):
    return ClipSample(clip_id, "sample:1", epoch, "Sample 1", 0.0, 0.5, 0, path, FRAMES, TIMES)


def slot(name, agent, enabled=True, remote=False):
    return Slot(name, AgentSpec(agent.provider, agent.model), agent, enabled, remote)


class Recorder:
    """Collects what the runner reports, and every clip deletion."""

    def __init__(self):
        self.results, self.ends, self.late, self.deleted = [], [], [], []
        self.lock = threading.Lock()
        self.late_seen = threading.Event()

    def runner(self, epoch=lambda: 0, **kwargs):
        def late(outcome):
            self.late.append(outcome)
            self.late_seen.set()

        return RoundRunner(
            current_epoch=epoch, on_result=self.results.append, on_round_end=self.ends.append, on_late=late,
            delete_clip=self.deleted.append, **kwargs,
        )


@pytest.fixture
def rec():
    return Recorder()


def test_both_answer_and_results_arrive_in_completion_order(rec, make_agent):
    slow = make_agent(model="slow", hang_on={1}, replies=["slow answer"])
    fast = make_agent(model="fast", replies=["fast answer"])

    def on_result(outcome):
        rec.results.append(outcome)
        assert rec.deleted == []  # the clip lives until every agent is done
        slow.release.set()  # A finishes only after B was reported

    runner = RoundRunner(current_epoch=lambda: 0, on_result=on_result, on_round_end=rec.ends.append,
                         delete_clip=rec.deleted.append)
    report = runner.run(clip(), [slot("A", slow), slot("B", fast)], SETTINGS)

    assert [(o.slot, o.status, o.result.text) for o in rec.results] == [
        ("B", "ok", "fast answer"), ("A", "ok", "slow answer"),
    ]
    assert report.outcomes == tuple(rec.results) and rec.ends == [report]
    assert rec.deleted == ["clip-7.mp4"]
    first = rec.results[0]
    assert (first.round_id, first.clip_id, first.source_epoch, first.stale) == (1, 7, 0, False)


def test_every_agent_gets_the_same_request(rec, make_agent):
    a, b = make_agent(model="a"), make_agent(model="b")
    rec.runner().run(clip(), [slot("A", a), slot("B", b)], SETTINGS)
    assert a.requests == b.requests
    (request,) = a.requests
    assert (request.frames, request.frame_times) == clip().frames_at(5, 10) == (FRAMES[1::2], TIMES[1::2])
    assert (request.prompt, request.temperature, request.video_path) == ("what happens?", 0.3, "clip-7.mp4")


def test_one_fails_one_times_out_and_the_round_still_ends(rec, make_agent):
    failing = make_agent(model="failing", raise_on={1: RuntimeError(f"bad key {SENTINEL}")})
    stuck = make_agent(model="stuck", hang_on={1})
    runner = rec.runner(call_timeout_s=0.3)
    t0 = time.monotonic()
    report = runner.run(clip(), [slot("A", failing), slot("B", stuck)], SETTINGS)
    assert time.monotonic() - t0 < 3

    by_slot = {o.slot: o for o in report.outcomes}
    assert (by_slot["A"].status, by_slot["A"].error_kind) == ("error", "internal")
    assert by_slot["A"].error_message == "Internal error (RuntimeError)"
    assert (by_slot["B"].status, by_slot["B"].error_kind) == ("timeout", "timeout")
    assert SENTINEL not in repr(report)
    assert rec.deleted == ["clip-7.mp4"]  # exactly once

    # The stuck call answers later: not shown, but handed on so a paid call can still be counted.
    stuck.release.set()
    assert rec.late_seen.wait(5)
    (late,) = rec.late
    assert (late.slot, late.status, late.round_id) == ("B", "ok", report.round_id)
    assert rec.deleted == ["clip-7.mp4"]


def test_a_provider_timeout_is_a_timeout(rec, make_agent):
    agent = make_agent(raise_on={1: TimeoutError("read timed out")})
    (outcome,) = rec.runner().run(clip(), [slot("A", agent)], SETTINGS).outcomes
    assert (outcome.status, outcome.error_kind, outcome.result) == ("timeout", "timeout", None)


def test_an_answer_after_a_source_change_is_reported_as_stale(rec, make_agent):
    epoch = [0]
    agent = make_agent(delay=1, sleep=lambda _s: epoch.__setitem__(0, 1))  # the source changes mid-call
    failing = make_agent(model="f", delay=1, sleep=lambda _s: None, raise_on={1: ConnectionError("down")})
    report = rec.runner(epoch=lambda: epoch[0]).run(clip(), [slot("A", agent, remote=True), slot("B", failing)],
                                                     SETTINGS)
    by_slot = {o.slot: o for o in report.outcomes}
    assert (by_slot["A"].status, by_slot["A"].stale, by_slot["A"].remote) == ("ok", True, True)
    assert by_slot["A"].result.text == "fake answer"  # still reported: it was paid for
    assert rec.results == list(report.outcomes)
    assert rec.deleted == ["clip-7.mp4"]


def test_a_clip_that_is_already_stale_is_not_sent(rec, make_agent):
    a, b = make_agent(model="a"), make_agent(model="b")
    report = rec.runner(epoch=lambda: 3).run(clip(epoch=2), [slot("A", a), slot("B", b)], SETTINGS)
    assert [o.status for o in report.outcomes] == ["skipped", "skipped"]
    assert a.calls == b.calls == 0
    assert rec.deleted == ["clip-7.mp4"]


def test_a_disabled_slot_is_not_asked(rec, make_agent):
    a, b = make_agent(model="a"), make_agent(model="b")
    report = rec.runner().run(clip(), [slot("A", a), slot("B", b, enabled=False)], SETTINGS)
    assert [o.slot for o in report.outcomes] == ["A"] and b.calls == 0


def test_one_round_at_a_time(rec, make_agent):
    agent = make_agent(hang_on={1})
    runner = rec.runner()
    assert runner.start(clip(), [slot("A", agent)], SETTINGS)
    assert agent.started.wait(5) and runner.busy
    assert not runner.start(clip(clip_id=8, path="clip-8.mp4"), [slot("A", agent)], SETTINGS)
    assert runner.run(clip(clip_id=9, path="clip-9.mp4"), [slot("A", agent)], SETTINGS) is None
    assert rec.deleted == []  # a refused clip is the caller's to delete
    agent.release.set()
    deadline = time.monotonic() + 5
    while runner.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not runner.busy and rec.deleted == ["clip-7.mp4"]
    assert runner.start(clip(clip_id=10, path="clip-10.mp4"), [slot("A", agent)], SETTINGS)
    assert runner.stop(5)


def test_shutdown_while_a_call_hangs_is_bounded(rec, make_agent):
    agent = make_agent(hang_on={1})
    runner = rec.runner(call_timeout_s=60)
    assert runner.start(clip(), [slot("A", agent)], SETTINGS)
    assert agent.started.wait(5)
    t0 = time.monotonic()
    runner.stop(timeout=0.2)
    assert time.monotonic() - t0 < 1.5
    deadline = time.monotonic() + 5  # the round thread notices the stop and cleans up
    while runner.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not runner.busy and rec.deleted == ["clip-7.mp4"]
    assert rec.ends == []  # nothing is reported during shutdown
    assert not runner.start(clip(clip_id=8, path="clip-8.mp4"), [slot("A", agent)], SETTINGS)
    # The agent thread is a daemon: it cannot keep the process alive.
    assert all(t.daemon for t in threading.enumerate() if t.name.startswith("agent-"))


def test_a_broken_listener_does_not_stop_the_round(make_agent):
    deleted = []

    def broken(_x):
        raise RuntimeError("listener bug")

    runner = RoundRunner(current_epoch=lambda: 0, on_result=broken, on_round_end=broken, delete_clip=deleted.append)
    report = runner.run(clip(), [slot("A", make_agent())], SETTINGS)
    assert report.outcomes[0].status == "ok" and deleted == ["clip-7.mp4"]
