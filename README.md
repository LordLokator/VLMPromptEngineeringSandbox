# Video Prompt Sandbox

[![ci](https://github.com/LordLokator/VLMPromptEngineeringSandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/LordLokator/VLMPromptEngineeringSandbox/actions/workflows/ci.yml)

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

- Webcam or static video input (OpenCV)

- Rolling clip buffer (few-second segments)

- VLM inference via DAMO-NLP-SG/VideoLLaMA3-7B

- Adjustable temperature (UI-controlled)

- Adjustable resample FPS (UI-controlled, 1-10; snapped to the rate the processor really uses)

- Thread-safe prompt updates

- Persistent user presets

- Prompt history logging with timestamps

- Live video streaming to browser

- Browser UI reconnects automatically (backoff 1 s to 10 s) and shows server errors as chat messages

- Rich structured logging (Loguru)

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

Attention: FlashAttention2 (default, override with `VLM_ATTN_IMPL`)

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

Python 3.11 is expected (inferred from the `numpy==1.24.4` pin, which has no wheels for Python 3.12 or newer; not tested on other versions). The setup script does not clone VideoLLaMA3; the model is loaded from Hugging Face at a pinned revision.

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

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `VLM_HOST` | `127.0.0.1` | Bind address. Anything non-loopback logs a warning (no authentication) and disables the API docs. |
| `VLM_PORT` | `8000` | Port. |
| `VLM_ATTN_IMPL` | `flash_attention_2` | Attention implementation passed to the model. Other values are untested. |

The WebSocket rejects cross-site browser origins. There is no authentication, so keep it on localhost.

----

## Web UI

    Top Left:
    
        Live video stream
    
    Bottom Left:
    
        Preset buttons (random colors; built-in and saved user presets)
        
        Save as Preset (disabled if empty)
        
        Temperature slider
        
        Resample FPS slider

    Right panel:
    
        VLM outputs
        
        User messages
        
        System acknowledgments and server errors

Presets stored in:

```
./presets/user_presets.json
```

Prompt history stored in:

```
./prompting/history.json
```

----

## Development

No GPU needed; the model code is isolated in `utils/vlm_wrapper.py` and mocked in tests.

```
pip install ".[dev]"
pytest -q
ruff check .
```

CI (`.github/workflows/ci.yml`) runs lint, compile check and tests on Python 3.11 and 3.12, plus a dependency-resolution check.

### Observability (optional OpenTelemetry)

Off by default; nothing from `opentelemetry` is imported unless enabled.

```
pip install ".[otel]"
OTEL_ENABLED=1 python main.py                       # spans + metrics printed to the console
OTEL_ENABLED=1 OTEL_TRACES_EXPORTER=otlp OTEL_METRICS_EXPORTER=otlp   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 python main.py   # any OTLP/gRPC collector
```

Recorded: a `vlm.inference` span per clip (backend, fps, max frames, temperature; ERROR status on failure),
metrics `vlm.inference.duration`, `vlm.inference.errors`, `vlm.clips.dropped` (clips skipped while inference is busy),
`ws.connections`, `ws.messages`, and the FastAPI HTTP metrics/spans (`/video_feed` and per-message WebSocket spans excluded).
Prompt text, model output, clip paths and frames are never recorded. Also honours `OTEL_SERVICE_NAME` and
`OTEL_METRIC_EXPORT_INTERVAL` (default 60 s for the console metrics). Tested with in-memory exporters only; no collector was run.

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


## License

MIT, see [LICENSE](LICENSE).
