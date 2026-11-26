#!/bin/bash
set -e

# Create virtual environment
echo "Creating virtual environment '.venv'..."
python3 -m venv .venv
source .venv/bin/activate
pip install ".[gpu]"
# flash-attn is built with --no-build-isolation, so the build helpers must be present first
pip install --upgrade pip setuptools wheel packaging
# flash-attn comes from an upstream commit, not PyPI: the latest PyPI release (2.8.3.post2, checked
# 2026-11-26) compiles as C++17, and torch 2.13 headers use C++20 syntax. This commit (2026-09-19,
# reports version 2.8.4) selects C++20 for torch >= 2.13. This CUDA build has not been run for this
# repo. Needs git and a CUDA 13 toolkit. Go back to PyPI once a release includes the fix.
FLASH_ATTN_COMMIT=edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d
pip install "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_COMMIT}" --no-build-isolation

# Log completion
echo "Setup completed successfully on $(date)"
