#!/bin/bash
set -e

# install VideoLLaMA3
if [ ! -d "VideoLLaMA3" ] ; then
    git clone "https://github.com/DAMO-NLP-SG/VideoLLaMA3.git" "VideoLLaMA3"
fi

# Create virtual environment
echo "Creating virtual environment '.venv'..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd VideoLLaMA3/
# flash-attn is built with --no-build-isolation, so the build helpers must be present first
pip install --upgrade pip setuptools wheel packaging
pip install flash-attn --no-build-isolation

# Log completion
echo "Setup completed successfully on $(date)"
