"""Import the app and build the default agent with the GPU stack made unimportable. Exits non-zero on failure.

Run by CI after a plain `pip install .` in a new venv, and by tests/test_providers.py.
"""
import importlib.abc
import sys
from pathlib import Path

BLOCKED = {"torch", "transformers", "decord", "flash_attn", "accelerate"}


class BlockGpuStack(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked: the default install must not need it")
        return None


sys.meta_path.insert(0, BlockGpuStack())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402,F401 - the import itself is the check
from utils.providers.agents import agent_spec_from_env, build_agent  # noqa: E402

agent = build_agent(agent_spec_from_env("A"))
sys.stdout.write(f"ok: {agent.label}\n")
