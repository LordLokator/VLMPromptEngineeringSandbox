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


@pytest.mark.parametrize("v", [0, 1, 3, 7, 10, 25])
def test_effective_rate_matches_stride(v):
    s = RuntimeSettings(clip_fps=10)
    clamped = min(max(v, 1), 10)
    assert s.set_resample_fps(v) == pytest.approx(10 / (10 // clamped))
    assert s.get_resample_fps() <= 10


def test_resample_rejects_non_numeric():
    with pytest.raises(ValueError):
        RuntimeSettings().set_resample_fps("x")


def test_resample_rejects_infinity():
    with pytest.raises(ValueError):
        RuntimeSettings().set_resample_fps(float("inf"))
