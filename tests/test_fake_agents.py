"""The fake agent factory itself: each mode, and that a hang is always released."""
import threading

import pytest

from tests.fake_agents import FAKE_COST_SOURCE, AgentFactory, FakeAgent
from utils.providers import GenerateRequest, ModelInfo


def request(prompt="p"):
    return GenerateRequest(frames=(b"x",), frame_times=(0.0,), system="s", prompt=prompt, temperature=0.2)


def test_replies_in_order_then_the_last_repeats(make_agent):
    agent = make_agent(replies=["one", "two"])
    assert [agent.generate(request()).text for _ in range(3)] == ["one", "two", "two"]
    assert agent.calls == 3


def test_records_requests_and_reports_no_usage_by_default(make_agent):
    agent = make_agent()
    result = agent.generate(request("what?"))
    assert [r.prompt for r in agent.requests] == ["what?"]
    assert (result.input_tokens, result.output_tokens, result.cost_usd, result.cost_source) == (None,) * 4
    assert agent.label == "demo:fake"


def test_usage_and_cost_are_marked_fake(make_agent):
    result = make_agent(usage=(10, 3), cost=0.002).generate(request())
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (10, 3, 0.002)
    assert result.cost_source == FAKE_COST_SOURCE


def test_delay_uses_the_injected_sleep(make_agent):
    slept = []
    make_agent(delay=2.5, sleep=slept.append).generate(request())
    assert slept == [2.5]


def test_raises_on_the_given_call_only(make_agent):
    agent = make_agent(raise_on={2: TimeoutError("slow")})
    agent.generate(request())
    with pytest.raises(TimeoutError):
        agent.generate(request())
    assert agent.generate(request()).text == "fake answer"
    assert agent.completions == [("demo:fake", 1), ("demo:fake", 2), ("demo:fake", 3)]


def test_hang_until_released(make_agent):
    agent = make_agent(hang_on={1})
    out = []
    worker = threading.Thread(target=lambda: out.append(agent.generate(request()).text), daemon=True)
    worker.start()
    assert agent.started.wait(5)
    worker.join(0.1)
    assert worker.is_alive() and out == []
    agent.release.set()
    worker.join(5)
    assert out == ["fake answer"]


def test_release_all_ends_a_forgotten_hang():
    factory = AgentFactory()
    agent = factory(hang_on={1})
    worker = threading.Thread(target=agent.generate, args=(request(),), daemon=True)
    worker.start()
    assert agent.started.wait(5)
    factory.release_all()  # what the make_agent fixture does at teardown
    worker.join(5)
    assert not worker.is_alive()


def test_shared_completion_log_shows_the_finishing_order(make_agent):
    log = []
    slow = make_agent(model="slow", hang_on={1}, completions=log)
    fast = make_agent(model="fast", completions=log)
    worker = threading.Thread(target=slow.generate, args=(request(),), daemon=True)
    worker.start()
    assert slow.started.wait(5)
    fast.generate(request())
    slow.release.set()
    worker.join(5)
    assert log == [("demo:fast", 1), ("demo:slow", 1)]


def test_list_models_and_listing_failure(make_agent):
    agent = make_agent(models=[ModelInfo("m1", vision=True)])
    assert agent.list_models() == [ModelInfo("m1", vision=True)]
    failing = make_agent(list_error=ConnectionError("down"))
    with pytest.raises(ConnectionError):
        failing.list_models()
    assert (agent.list_calls, failing.list_calls) == (1, 1)


def test_is_thread_safe(make_agent):
    agent = make_agent()
    threads = [threading.Thread(target=agent.generate, args=(request(),)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert agent.calls == 20
    assert sorted(n for _label, n in agent.completions) == list(range(1, 21))


def test_empty_replies_are_rejected():
    with pytest.raises(ValueError):
        FakeAgent(replies=[])
