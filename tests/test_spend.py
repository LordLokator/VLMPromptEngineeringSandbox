"""Spend controller on a fake clock: every state, cap, pacing rule and auto-pause trigger."""
import pytest

from utils.spend import PAUSE_REASONS, SpendController


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def controller(clock, **kwargs):
    return SpendController(clock=clock, **kwargs)


def test_starts_paused(clock):
    spend = controller(clock)
    assert spend.admit(remote=False).reason == "paused"
    assert spend.admit(remote=True).reason == "paused"
    state = spend.snapshot()
    assert not state["running"] and state["pause_reason"] == "start"
    assert state["pause_text"] == PAUSE_REASONS["start"]


def test_run_and_pause_apply_to_local_agents_without_pacing_or_budget(clock):
    spend = controller(clock, max_rounds=1)
    assert spend.control("run", remote=False) is None
    assert all(spend.admit(remote=False).start for _ in range(5))  # no pacing, no round cap
    assert spend.rounds_used == 0
    spend.control("pause", remote=False)
    assert spend.admit(remote=False).reason == "paused"
    assert spend.snapshot()["pause_reason"] == "user"


def test_pacing_between_paid_round_starts(clock):
    spend = controller(clock)
    spend.control("run", remote=True)
    assert spend.admit(remote=True).start
    clock.now += 4.9
    assert spend.snapshot()["next_paid_round_in_s"] == 0.1
    assert spend.admit(remote=True).reason == "pacing"
    assert spend.admit(remote=False).start  # a local-only round is not paced and does not reset the clock
    clock.now += 0.1
    assert spend.admit(remote=True).start
    assert spend.rounds_used == 2


def test_step_while_paused_runs_exactly_one_round(clock):
    spend = controller(clock)
    assert spend.control("step", remote=True) is None
    assert spend.snapshot()["step_pending"]
    assert spend.admit(remote=True).start
    assert spend.admit(remote=True).reason == "paused"
    clock.now += 60
    assert spend.admit(remote=True).reason == "paused"
    state = spend.snapshot()
    assert state["pause_reason"] == "step" and not state["step_pending"] and spend.rounds_used == 1


def test_step_waits_for_pacing_and_stays_pending(clock):
    spend = controller(clock)
    spend.control("step", remote=True)
    assert spend.admit(remote=True).start
    clock.now += 1
    spend.control("step", remote=True)
    assert spend.admit(remote=True).reason == "pacing"
    assert spend.snapshot()["step_pending"]
    clock.now += 4
    assert spend.admit(remote=True).start


def test_step_while_running_is_refused_and_pause_cancels_a_pending_step(clock):
    spend = controller(clock)
    spend.control("run", remote=False)
    assert spend.control("step", remote=False) == "Already running: pause first to step"
    spend.control("pause", remote=False)
    spend.control("step", remote=False)
    spend.control("pause", remote=False)
    assert spend.admit(remote=False).reason == "paused"


def test_round_cap_pauses_and_only_reset_refills(clock):
    spend = controller(clock, max_rounds=3)
    spend.control("run", remote=True)
    for _ in range(3):
        assert spend.admit(remote=True).start
        clock.now += 5
    assert spend.admit(remote=True).reason == "budget"
    state = spend.snapshot()
    assert state["pause_reason"] == "budget" and state["exhausted"] and state["rounds_used"] == 3
    assert spend.control("run", remote=True) == PAUSE_REASONS["budget"]
    assert spend.control("step", remote=True) == PAUSE_REASONS["budget"]
    assert spend.control("run", remote=False) is None  # local agents have no budget
    assert spend.admit(remote=False).start
    spend.control("pause", remote=False)

    spend.control("reset_budget", remote=True)
    assert spend.snapshot()["pause_reason"] == "user"  # a reset does not start anything
    assert spend.control("run", remote=True) is None and spend.admit(remote=True).start


def test_run_resumes_the_remaining_budget(clock):
    spend = controller(clock, max_rounds=3)
    spend.control("run", remote=True)
    spend.admit(remote=True)
    spend.control("pause", remote=True)
    clock.now += 10
    spend.control("run", remote=True)
    spend.admit(remote=True)
    assert spend.rounds_used == 2  # not refilled by Run


def test_usd_cap_pauses_and_a_round_in_flight_may_overshoot(clock):
    spend = controller(clock)
    spend.control("run", remote=True)
    assert spend.admit(remote=True).start
    spend.record("A", "ok", remote=True, cost_usd=0.30, error_kind=None)
    assert spend.running
    spend.record("B", "ok", remote=True, cost_usd=0.25, error_kind=None)  # same round: counted in full
    assert not spend.running and spend.snapshot()["pause_reason"] == "budget"
    assert spend.usd_spent == pytest.approx(0.55)
    clock.now += 10
    assert spend.control("step", remote=True) == PAUSE_REASONS["budget"]


def test_unpriced_calls_are_counted_and_shown(clock):
    spend = controller(clock)
    spend.record("A", "ok", remote=True, cost_usd=None, error_kind=None)
    spend.record("A", "error", remote=True, cost_usd=None, error_kind="bad_request")
    spend.record("B", "ok", remote=False, cost_usd=0.0, error_kind=None)  # local: no API fee, not a paid call
    spend.record("B", "skipped", remote=True, cost_usd=None, error_kind=None)  # never sent
    state = spend.snapshot()
    assert (state["paid_calls"], state["unpriced_calls"], state["usd_spent"]) == (2, 1, 0.0)


def test_stale_and_late_answers_count_like_any_other(clock):
    spend = controller(clock)
    for _ in range(2):  # the caller reports them through the same call
        spend.record("A", "ok", remote=True, cost_usd=0.01, error_kind=None)
    assert spend.usd_spent == pytest.approx(0.02) and spend.paid_calls == 2


def test_auth_error_pauses_at_once(clock):
    spend = controller(clock)
    spend.control("run", remote=True)
    spend.record("B", "error", remote=True, cost_usd=None, error_kind="auth")
    state = spend.snapshot()
    assert not state["running"] and state["pause_reason"] == "auth" and state["pause_slot"] == "B"


def test_three_failures_in_a_row_of_one_slot_pause(clock):
    spend = controller(clock)
    spend.control("run", remote=False)
    for status in ("error", "timeout", "ok", "error", "error"):  # an ok answer resets the count
        spend.record("A", status, remote=False, cost_usd=None, error_kind=None if status == "ok" else "internal")
    spend.record("B", "error", remote=False, cost_usd=None, error_kind="internal")  # B's count is its own
    assert spend.running  # A: 2 in a row, B: 1
    spend.record("A", "timeout", remote=False, cost_usd=None, error_kind="timeout")
    state = spend.snapshot()
    assert state["pause_reason"] == "failures" and state["pause_slot"] == "A"
    spend.control("run", remote=False)  # trying again starts a fresh count
    spend.record("A", "error", remote=False, cost_usd=None, error_kind="internal")
    assert spend.running


def test_last_browser_disconnecting_pauses(clock):
    spend = controller(clock)
    spend.control("run", remote=False)
    spend.disconnected(remaining=1)
    assert spend.running
    spend.disconnected(remaining=0)
    assert spend.snapshot()["pause_reason"] == "disconnect"
    spend.control("step", remote=False)
    spend.disconnected(remaining=0)
    assert spend.admit(remote=False).reason == "paused"  # a pending step is cancelled too


def test_disconnect_while_paused_keeps_the_reason(clock):
    spend = controller(clock)
    spend.disconnected(remaining=0)
    assert spend.snapshot()["pause_reason"] == "start"


def test_unknown_action_is_refused(clock):
    assert controller(clock).control("spend-all", remote=True) == "Unknown action 'spend-all'"


def test_a_late_answer_adds_its_cost_only(clock):
    spend = controller(clock)
    spend.control("run", remote=True)
    for _ in range(2):
        spend.record("A", "timeout", remote=True, cost_usd=None, error_kind="timeout")
        spend.record_late(remote=True, cost_usd=0.01)  # the stuck call answered after all
    assert spend.paid_calls == 2 and spend.usd_spent == pytest.approx(0.02)
    spend.record("A", "timeout", remote=True, cost_usd=None, error_kind="timeout")
    assert spend.snapshot()["pause_reason"] == "failures"  # late answers did not reset the streak
    spend.control("run", remote=True)
    spend.record_late(remote=True, cost_usd=0.60)
    assert spend.snapshot()["pause_reason"] == "budget"
    spend.record_late(remote=False, cost_usd=None)  # local: nothing to count
