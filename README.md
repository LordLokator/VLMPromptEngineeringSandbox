# Video Prompt Sandbox

Interactive Video-Language Model (VLM) Playground with Live Prompt Editing

----

A real-time sandbox for experimenting with VideoLLaMA3-7B on short video clips using a web UI.
Designed as a university project for the course Prompt Engineering (VITMAV82).

## The system:

- captures video (webcam or static file),

- buffers short clips,

- runs VLM inference,

- streams output to a browser UI,

- supports interactive prompt editing + preset management.

Tested on Ubuntu + RTX 4090.


<img width="1919" height="869" alt="image" src="https://github.com/user-attachments/assets/e5600752-73b2-404c-997b-cb4f48cd624d" />

----

## Features

- 🎥 Webcam or static video input (OpenCV)

- ⏱ Rolling clip buffer (few-second segments)

- 🧠 VLM inference via DAMO-NLP-SG/VideoLLaMA3-7B

- 🌡 Adjustable temperature (UI-controlled)

- 🧵 Thread-safe prompt updates

- 💾 Persistent user presets

- 🗂 Prompt history logging with timestamps

- 🌐 Live video streaming to browser

- 📜 Rich structured logging (Loguru)

----

## Architecture Overview

Pipeline:

```
Video Source (Webcam / File)
        ↓
ClipRecorder (temporal buffer → .mp4)
        ↓
ThreadPoolExecutor
        ↓
VideoLLaMA3-7B (Transformers)
        ↓
WebSocket broadcast
        ↓
Browser UI
```

Stack: FastAPI + uvicorn (WebSocket at `/ws`), OpenCV, vanilla JS frontend.

----

## Known Limitations

- The browser stream and the clip the VLM sees come from separate decoders at different positions, so they are not synchronised.

- No authentication; intended for localhost use only.

----

## Key Design Choices

- Clips are written to disk temporarily, then deleted after inference.

- Single-worker inference executor (avoids GPU contention).

- Prompt state shared via thread-safe Prompt object.

- Presets stored in JSON.

- Prompt history persisted only when content changes.

----

## Model Parameters

Model: DAMO-NLP-SG/VideoLLaMA3-7B

LLM Framework: HuggingFace Transformers

Attention: FlashAttention2

Precision: bfloat16

Device: cuda:0

----

## Hardware Requirements

### Tested on:

&rarr; Ubuntu

&rarr; RTX 4090 (24GB VRAM)

&rarr; CUDA compatible environment

### Memory:

&rarr; ~24GB VRAM (author-reported, not independently measured). The weights alone are about 15GB (7.6B parameters x 2 bytes, an estimate). No quantization.

&rarr; A CUDA GPU with FlashAttention2 support (Ampere or newer) is required.

### Latency:

&rarr; ~200ms at 10 FPS resampling (author-reported, not independently measured)

&rarr; ~400ms at 30–40 FPS in a similar setup (author-reported, not independently measured)

----

## Quick Setup

Python 3.11 is required (the pinned `numpy==1.24.4` has no wheels for Python 3.12 or newer).

```
git clone https://github.com/LordLokator/VLMPromptEngineeringSandbox VLMDemo
cd VLMDemo
bash setup.sh
source .venv/bin/activate
python main.py
```

Then open:

```
http://localhost:8000/
```

----

## Web UI

    Top Left:
    
        Live video stream
    
    Bottom Left:
    
        Preset buttons (random colors; built-in and saved user presets)
        
        Save as Preset (disabled if empty)
        
        Temperature slider

    Right panel:
    
        🟢 VLM outputs
        
        🔵 User messages
        
        ⚫ System acknowledgments

Presets stored in:

```
./presets/user_presets.json
```

Prompt history stored in:

```
./prompting/history.json
```

----

## Built-in Prompt Presets

- Dominant Action

- Crossing

- Vehicle Types

- Groups

- Unusual Behavior

- Activity

- Sun

- Parking Events

- OCR Pass

----

## Why This Project Exists

This was built for a university course on Prompt Engineering.

The goal was not to build another LLM chat interface, but to:

- explore prompt design for multimodal models

- experiment with temporal video chunking

- observe latency vs FPS trade-offs

- create a controlled sandbox for VLM behavior
