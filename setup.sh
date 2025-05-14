#!/bin/bash
set -e

# Log file
mkdir -p -m 700 logs
LOGFILE="./logs/setup_env.log"

# install VideoLLaMA3
if [ ! -d "VideoLLaMA3" ] ; then
    git clone "https://github.com/DAMO-NLP-SG/VideoLLaMA3.git" "VideoLLaMA3"
fi

# Create virtual environment
echo "Creating virtual environment '.venv'..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r vlm_requirements.txt
cd VideoLLaMA3/
pip install flash-attn --no-build-isolation

# Log completion
echo "Setup completed successfully on $(date)"
