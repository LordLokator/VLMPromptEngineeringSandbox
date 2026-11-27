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
def fake_torch(monkeypatch):
    mod = types.SimpleNamespace(Tensor=FakeTensor, bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod
