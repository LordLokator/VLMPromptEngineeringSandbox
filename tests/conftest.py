import os
import sys
import types

import cv2
import pytest

# TestClient sends "Host: testserver"; only this test configuration allows it. Set before web_engine is imported,
# and overwritten, so a developer's own setting cannot change what the tests see.
os.environ["VLM_ALLOWED_HOSTS"] = "testserver"


_VideoCapture = cv2.VideoCapture


@pytest.fixture(autouse=True)
def no_webcam(request, monkeypatch):
    """No test may open a camera: a device index is refused, files still open. Opted-in manual tests may."""
    if request.node.get_closest_marker("manual") and os.environ.get("VLM_MANUAL_TESTS") == "1":
        return

    def files_only(source, *args):
        assert not isinstance(source, int), "a test tried to open a camera"
        return _VideoCapture(source, *args)

    monkeypatch.setattr(cv2, "VideoCapture", files_only)


class FakeTensor:
    """Stands in for torch.Tensor so the VLM wrapper runs without torch."""

    def to(self, *_args, **_kwargs):
        return self


@pytest.fixture
def make_agent():
    """Build fake agents (tests/fake_agents.py); every hanging one is released at teardown."""
    from tests.fake_agents import AgentFactory

    factory = AgentFactory()
    yield factory
    factory.release_all()


@pytest.fixture
def fake_torch(monkeypatch):
    mod = types.SimpleNamespace(Tensor=FakeTensor, bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class StubPrompt:
    """Prompt without the history file."""

    def __init__(self, value="describe"):
        self.value = value

    def get(self):
        return self.value

    def set(self, text):
        self.value = text


@pytest.fixture
def compare_app(monkeypatch, make_agent):
    """The web app with a real CompareSession: fake agents (A: demo:frame-facts, B: demo:frame-motion), a capture
    hub that is never started, a stub prompt, and clip deletion recorded instead of done.

    `compare_app.agents` maps "provider:model" to the fake agent built for it; `options` sets FakeAgent arguments
    for agents built from then on.
    """
    import types as _types

    from fastapi.testclient import TestClient

    import web_engine
    from prompting.prompting import RuntimeSettings
    from utils.providers.registry import AgentRegistry
    from utils.session import CompareSession
    from utils.spend import SpendController
    from video_handling.streaming import CaptureHub

    for name in ("VLM_AGENT_A", "VLM_AGENT_B", "VLM_ENABLE_HF", "VLM_ALLOW_REMOTE_PAID", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL", "VLM_COMPAT_BASE_URL", "VLM_COMPAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = _types.SimpleNamespace(agents={}, options={}, deleted=[])

    def build(spec):
        agent = make_agent(spec.provider, spec.model or "default", **env.options)
        env.agents[agent.label] = agent
        return agent

    registry = AgentRegistry(build=build)
    registry.set_agent("A", "demo", "frame-facts")
    registry.set_agent("B", "demo", "frame-motion")
    hub = CaptureHub()  # never started, so no source is opened
    env.prompt, env.settings, env.hub, env.registry = StubPrompt(), RuntimeSettings(), hub, registry
    env.session = CompareSession(
        registry=registry, spend=SpendController(), hub=hub, prompt=env.prompt, settings=env.settings,
        broadcast=web_engine.broadcast_sync, clip_fps=10, max_frames=20, delete_clip=env.deleted.append,
        call_timeout_s=5,
    )
    monkeypatch.setattr(web_engine.app.state, "session", env.session, raising=False)
    monkeypatch.setattr(web_engine.app.state, "hub", hub, raising=False)
    web_engine.active_connections.clear()
    # Entered, so the lifespan sets the event loop that broadcast_sync hands frames to.
    with TestClient(web_engine.app) as client:
        env.client = client
        yield env
        env.session.stop(1)
    web_engine.active_connections.clear()
