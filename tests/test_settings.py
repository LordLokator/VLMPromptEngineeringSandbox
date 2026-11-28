import threading

import pytest

from prompting.prompting import RuntimeSettings


def test_defaults():
    s = RuntimeSettings()
    assert s.get_temperature() == 0.01
    assert s.get_resample_fps() == 10


def test_temperature_clamped():
    s = RuntimeSettings()
    assert s.set_temperature(0) == 0.01
    assert s.set_temperature(5) == 1.0
    assert s.set_temperature("0.7") == 0.7
    assert s.get_temperature() == 0.7


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "abc", None])
def test_temperature_rejects_invalid(bad):
    s = RuntimeSettings()
    with pytest.raises(ValueError):
        s.set_temperature(bad)
    assert s.get_temperature() == 0.01


def test_threads_do_not_raise():
    s = RuntimeSettings()
    errors = []

    def work(i):
        try:
            for _ in range(200):
                s.set_temperature(i / 10)
                s.set_resample_fps(i + 1)
                s.get_temperature()
                s.effective_resample_fps()
        except Exception as e:  # noqa: BLE001 - collect for the assertion
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert 0.01 <= s.get_temperature() <= 1.0


@pytest.mark.parametrize("v,requested", [(0, 1), (1, 1), (3, 3), (7, 7), (10, 10), (25, 10)])
def test_resample_clamp(v, requested):
    s = RuntimeSettings(clip_fps=10)
    s.set_resample_fps(v)
    assert s.get_resample_fps() == requested


# Frames the pinned loader returned per slider value for the app's 5-frame, 10 fps clips (container check, T71).
LOADER_FRAMES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5}


@pytest.mark.parametrize("v", sorted(LOADER_FRAMES))
def test_effective_rate_matches_loader_frames(v):
    s = RuntimeSettings(clip_fps=10, clip_seconds=0.5)
    assert s.set_resample_fps(v) == pytest.approx(LOADER_FRAMES[v] / 0.5)


def test_resample_rejects_non_numeric():
    with pytest.raises(ValueError):
        RuntimeSettings().set_resample_fps("x")


def test_resample_rejects_infinity():
    with pytest.raises(ValueError):
        RuntimeSettings().set_resample_fps(float("inf"))


@pytest.mark.parametrize("v", sorted(LOADER_FRAMES))
def test_agent_frames_match_loader_frames(v):
    """API agents get the frames the local model's loader picks from the same clip."""
    from video_handling.sampling import loader_indices

    indices = loader_indices(5, 10, v)
    assert len(indices) == LOADER_FRAMES[v]
    assert indices == sorted(set(indices)) and all(0 <= i < 5 for i in indices)


@pytest.mark.parametrize("n, fps, expected", [
    (0, 10, []), (1, 1, [0]), (1, 10, [0]), (3, 1, [1]), (3, 3, [1]), (3, 10, [0, 1, 2]),
    (5, 0, [2]), (5, 99, list(range(5))),
])
def test_loader_indices_on_short_clips_and_out_of_range_fps(n, fps, expected):
    from video_handling.sampling import loader_indices

    assert loader_indices(n, 10, fps) == expected


def test_jpeg_long_edge_and_quality():
    import cv2
    import numpy as np

    from video_handling.write_video import encode_jpeg

    def size(jpeg):
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]

    frame = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    assert size(encode_jpeg(frame)) == (432, 768)
    assert size(encode_jpeg(frame[:, :600])) == (768, 427)  # portrait: the height is the long edge
    assert size(encode_jpeg(frame[:100, :200])) == (100, 200)  # never scaled up
    small = frame[:100, :200]
    ok, expected = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert encode_jpeg(small) == expected.tobytes()
