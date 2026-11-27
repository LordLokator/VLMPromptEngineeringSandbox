import os
import sys
import types

import pytest

# TestClient sends "Host: testserver"; only this test configuration allows it. Set before web_engine is imported,
# and overwritten, so a developer's own setting cannot change what the tests see.
os.environ["VLM_ALLOWED_HOSTS"] = "testserver"


class FakeTensor:
    """Stands in for torch.Tensor so the VLM wrapper runs without torch."""

    def to(self, *_args, **_kwargs):
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    mod = types.SimpleNamespace(Tensor=FakeTensor, bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod
