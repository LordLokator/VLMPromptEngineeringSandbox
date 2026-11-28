"""Check one provider from the command line: one call on frames of the bundled sample, or its model list.

    python -m utils.providers.check <provider> <model>      one call; a paid provider may charge for it
    python -m utils.providers.check <provider> --list       the models the provider offers (no generation)

Keys and URLs come from the environment or `.env` (see README, Configuration). The output never contains a key.
"""
import argparse
import json
import sys
from dataclasses import asdict

from loguru import logger

from utils.errors import log_failure
from utils.file_management import BASE_DIR
from utils.logging_setup import configure_logging, redact, route_stdlib_logging
from utils.providers import MAX_OUTPUT_TOKENS, PROVIDERS, SYSTEM_PROMPT, GenerateRequest
from utils.providers.settings import ProviderConfigError, load_env_file

SAMPLE = BASE_DIR / "static_video" / "street_view_1.mp4"
CLIP_FPS = 10
CLIP_FRAMES = 5  # one 0.5 s clip, as the app cuts it
DEFAULT_PROMPT = "Describe what happens in these frames in one sentence."
CHECK_PROVIDERS = tuple(p for p in PROVIDERS if p != "hf")  # hf needs the app's clip file and a GPU


def sample_frames(path=SAMPLE, start_s: float = 0.0) -> tuple[tuple[bytes, ...], tuple[float, ...]]:
    """One clip's frames as the app samples them: a 10 FPS grid from `start_s`, the latest frame at each point."""
    import cv2

    from video_handling.write_video import encode_jpeg

    capture = cv2.VideoCapture(str(path))
    frames, times, held = [], [], None
    try:
        grid = [start_s + i / CLIP_FPS for i in range(CLIP_FRAMES)]
        while len(frames) < CLIP_FRAMES:
            ok, frame = capture.read()
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if ok else float("inf")
            while held is not None and len(frames) < CLIP_FRAMES and grid[len(frames)] < t - 1e-3:
                frames.append(encode_jpeg(held))
                times.append(grid[len(times)])
            if not ok:
                break
            held = frame
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return tuple(frames), tuple(times)


def _write(text: str) -> None:
    sys.stdout.write(redact(text) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m utils.providers.check", description=__doc__.splitlines()[0])
    parser.add_argument("provider", choices=CHECK_PROVIDERS)
    parser.add_argument("model", nargs="?", help="model id; omit with --list")
    parser.add_argument("--list", action="store_true", help="list the provider's models instead of calling one")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fps", type=int, default=CLIP_FPS, help="resample FPS, 1-10 (default 10: all 5 frames)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args(argv)

    try:
        load_env_file()
    except ProviderConfigError as e:
        configure_logging()
        logger.error(str(e))
        return 2
    configure_logging()

    from utils.providers.agents import AgentConfigError, AgentSpec, build_provider
    from video_handling.sampling import loader_indices

    model = args.model
    if model is None and not args.list:
        parser.error("give a model, or --list")
    try:
        if model is None:  # --list: the model does not matter, but every provider is built with one
            model = "frame-facts" if args.provider == "demo" else "list"
        provider = build_provider(AgentSpec(args.provider, model))
    except (ProviderConfigError, AgentConfigError) as e:
        logger.error(str(e))
        return 2
    try:
        if args.list:
            for info in provider.list_models():
                vision = {True: "vision", False: "no vision", None: "vision unknown"}[info.vision]
                _write(f"{info.id}\t{vision}")
            return 0
        frames, times = sample_frames()
        picked = loader_indices(len(frames), CLIP_FPS, args.fps)
        request = GenerateRequest(
            frames=tuple(frames[i] for i in picked),
            frame_times=tuple(times[i] for i in picked),
            system=SYSTEM_PROMPT,
            prompt=args.prompt,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        result = provider.generate(request)
    except Exception as e:  # noqa: BLE001 - command-line boundary: the safe message only, never the raw error
        _write(f"FAILED: {log_failure(f'{args.provider} check failed', e)}")
        return 1
    finally:
        route_stdlib_logging()  # an SDK may have added a handler on first use
    _write(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    if not result.text:
        _write(f"Note: the answer is empty (finish reason: {result.finish_reason}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
