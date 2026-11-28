"""CompareSession without the web layer: which clips become rounds, stale results, late answers, busy drops."""
import time

from prompting.prompting import RuntimeSettings
from tests.conftest import StubPrompt
from utils.providers.registry import AgentRegistry
from utils.rounds import AgentOutcome
from utils.session import CompareSession, result_frame, thumbnail
from utils.spend import SpendController
from video_handling.clip_handling import ClipSample
from video_handling.streaming import CaptureHub


def clip(clip_id=1, epoch=0):
    return ClipSample(clip_id, "sample:1", epoch, "Sample 1", 0.0, 0.5, 0, f"c{clip_id}.mp4", (b"x",) * 5, (0.0,) * 5)


def make_session(make_agent, **agent_options):
    agents = []

    def build(spec):
        agents.append(make_agent(spec.provider, spec.model or "default", **agent_options))
        return agents[-1]

    registry = AgentRegistry(build=build)
    registry.set_agent("A", "demo", "frame-facts")
    registry.set_agent("B", "demo", "frame-motion")
    sent, deleted = [], []
    session = CompareSession(
        registry=registry, spend=SpendController(), hub=CaptureHub(), prompt=StubPrompt(), settings=RuntimeSettings(),
        broadcast=sent.append, clip_fps=10, max_frames=20, delete_clip=deleted.append, call_timeout_s=5,
    )
    return session, agents, sent, deleted


def wait_for(check):
    deadline = time.monotonic() + 5
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert check()


def test_a_stale_clip_never_becomes_a_round(make_agent):
    session, agents, sent, deleted = make_session(make_agent)
    session.spend.control("run", remote=False)
    session.hub.select("sample:2")
    session.on_clip(clip(epoch=0))
    assert deleted == ["c1.mp4"] and all(a.calls == 0 for a in agents) and sent == []


def test_a_busy_round_drops_the_clip_and_counts_it(make_agent):
    session, agents, sent, deleted = make_session(make_agent, hang_on={1})
    session.spend.control("run", remote=False)
    session.on_clip(clip(1))
    assert agents[0].started.wait(5)
    session.on_clip(clip(2))
    assert deleted == ["c2.mp4"] and session.busy_drops == 1
    for agent in agents:
        agent.release.set()
    wait_for(lambda: not session.runner.busy)
    assert sorted(deleted) == ["c1.mp4", "c2.mp4"]


def test_an_answer_after_a_switch_is_marked_previous_source(make_agent):
    session, agents, sent, deleted = make_session(make_agent, hang_on={1})
    session.spend.control("step", remote=False)
    session.on_clip(clip())
    assert all(a.started.wait(5) for a in agents)
    session.hub.select("sample:2")
    for agent in agents:
        agent.release.set()
    wait_for(lambda: not session.runner.busy)
    results = [f for f in sent if f["type"] == "result"]
    assert len(results) == 2 and all(r["stale"] and r["status"] == "ok" for r in results)


def test_a_late_answer_is_counted_but_not_shown(make_agent):
    session, agents, sent, deleted = make_session(make_agent, hang_on={1})
    session.runner._call_timeout_s = 0.2
    session.spend.control("step", remote=False)
    session.on_clip(clip())
    wait_for(lambda: not session.runner.busy)
    assert [f["status"] for f in sent if f["type"] == "result"] == ["timeout", "timeout"]
    shown = len(sent)
    for agent in agents:
        agent.release.set()
    wait_for(lambda: len(sent) >= shown + 2)
    assert [f["type"] for f in sent[shown:]] == ["state", "state"]  # counted, never a result card


def test_round_frame_thumbnails_are_small():
    import cv2
    import numpy as np

    jpeg = cv2.imencode(".jpg", np.zeros((432, 768, 3), np.uint8))[1].tobytes()
    url = thumbnail(jpeg)
    assert url.startswith("data:image/jpeg;base64,") and len(url) < 8000
    assert thumbnail(b"not a jpeg") is None


def test_result_frame_for_a_failure_has_no_text_and_a_safe_error():
    outcome = AgentOutcome(1, 2, 0, "A", "openai:gpt-x", True, "error", False, None, "auth",
                           "Authentication failed (AuthenticationError, HTTP 401)", 12.34)
    frame = result_frame(outcome, stale=False)
    assert frame["text"] is None and frame["error"]["kind"] == "auth" and frame["metrics"] == {"latency_ms": 12.3}
