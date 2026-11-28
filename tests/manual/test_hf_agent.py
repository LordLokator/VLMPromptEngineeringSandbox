"""The local model as an agent, on a CUDA GPU. Skipped unless VLM_MANUAL_TESTS=1 and VLM_ENABLE_HF=1.

Downloads the model repository at the pinned revision (several GB) on first use and loads it on cuda:0, with the
headless OpenCV build that every install now uses. No API is called and nothing costs money.
"""
import os

import cv2
import pytest

from main import FPS, MAX_FRAMES, build_request
from utils.file_management import BASE_DIR
from video_handling.clip_handling import ClipSample
from video_handling.write_video import encode_jpeg, write_video

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(os.environ.get("VLM_MANUAL_TESTS") != "1", reason="manual: set VLM_MANUAL_TESTS=1"),
    pytest.mark.skipif(os.environ.get("VLM_ENABLE_HF") != "1", reason="loads the model: set VLM_ENABLE_HF=1"),
]


def first_frames(path, count):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def test_hf_agent_answers_a_recorded_clip(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from utils.providers.hf import HFProvider

    frames = first_frames(BASE_DIR / "static_video" / "street_view_1.mp4", 5)
    assert len(frames) == 5
    clip_path = str(tmp_path / "clip.mp4")
    write_video(frames, clip_path, fps=FPS)
    clip = ClipSample(0, "sample:1", 0, "Sample 1", 0.0, 0.5, 0, clip_path,
                      tuple(encode_jpeg(f) for f in frames), tuple(i / FPS for i in range(5)))

    agent = HFProvider()
    for fps in (2, 5, 10):
        request = build_request(clip, prompt="Describe the scene in one sentence.", temperature=0.2, fps=fps,
                                max_frames=MAX_FRAMES)
        result = agent.generate(request)
        assert result.text.strip(), f"empty answer at {fps} fps"
        assert result.params_sent["fps"] == fps
