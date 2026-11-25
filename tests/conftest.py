import sys
import types

import pytest


class FakeTensor:
    """Stands in for torch.Tensor so the VLM wrapper runs without torch."""

    def to(self, *_args, **_kwargs):
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    mod = types.SimpleNamespace(Tensor=FakeTensor, bfloat16="bf16")
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod
