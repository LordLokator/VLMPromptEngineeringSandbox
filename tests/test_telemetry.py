import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from main import make_clip_handler, process_clip
from utils import telemetry
from video_handling.clip_handling import ClipSample

PROMPT = "SECRET-PROMPT-TEXT"


def sample(path):
    return ClipSample(0, "sample:1", 0, "Sample 1", 0.0, 0.5, 0, str(path))


class FakeVLM:
    name = "fake/backend"

    def __init__(self, fail=False):
        self.fail = fail

    def forward(self, video_path, fps, max_frames, temperature):
        if self.fail:
            raise RuntimeError("boom")
        return PROMPT, temperature


class GoodVLM(FakeVLM):
    name = "good"


def run(vlm):
    process_clip(
        sample("c.mp4"), vlm, lambda m: None,
        get_temperature=lambda: 0.5, get_fps=lambda: 3, get_source_epoch=lambda: 0, max_frames=20,
    )


def test_disabled_imports_no_opentelemetry():
    code = (
        "import sys; from unittest import mock; import utils.telemetry as t; "
        "t.setup_telemetry(mock.Mock()); t.record_drop(); "
        "sys.exit(any(m.startswith('opentelemetry') for m in sys.modules))"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "OTEL_ENABLED"}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_disabled_helpers_are_noops():
    telemetry.reset()
    with telemetry.inference("x", 1, 1, 0.1):
        pass
    telemetry.record_drop()
    telemetry.record_ws_message("user")
    telemetry.ws_connection(1)


@pytest.fixture
def otel():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    telemetry.configure(tp, MeterProvider(metric_readers=[reader]))
    yield exporter, reader
    telemetry.reset()


def points(reader):
    """{metric name: [data points]} from the in-memory reader."""
    data = reader.get_metrics_data()
    out = {}
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out[m.name] = list(m.data.data_points)
    return out


def test_inference_span_and_histogram_ok(otel):
    exporter, reader = otel
    run(FakeVLM())
    (span,) = exporter.get_finished_spans()
    assert span.name == "vlm.inference"
    assert span.attributes["vlm.backend"] == "fake/backend"
    assert span.attributes["vlm.fps"] == 3
    assert span.attributes["vlm.temperature"] == 0.5
    (hist,) = points(reader)["vlm.inference.duration"]
    assert hist.attributes["outcome"] == "ok"
    assert "vlm.inference.errors" not in points(reader)


def test_chain_fallback_labels_answering_backend(otel):
    from utils.backends import ChainBackend

    exporter, reader = otel
    run(ChainBackend([FakeVLM(fail=True), GoodVLM()]))
    (span,) = exporter.get_finished_spans()
    assert span.attributes["vlm.backend"] == "good"
    assert points(reader)["vlm.inference.duration"][0].attributes["vlm.backend"] == "good"


def test_error_sets_status_and_counts(otel):
    from opentelemetry.trace import StatusCode

    exporter, reader = otel
    run(FakeVLM(fail=True))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    p = points(reader)
    assert p["vlm.inference.errors"][0].value == 1
    assert p["vlm.inference.duration"][0].attributes["outcome"] == "error"


def test_drop_counter_increments_when_busy(otel, tmp_path):
    _, reader = otel
    gate = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    handler = make_clip_handler(lambda p: gate.wait(5), executor, threading.Lock())
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    handler(sample(a))
    handler(sample(b))
    gate.set()
    executor.shutdown(wait=True)
    assert points(reader)["vlm.clips.dropped"][0].value == 1


def test_ws_role_label_is_bounded(otel):
    _, reader = otel
    for role in ("user", "evil-1", "evil-2", None):
        telemetry.record_ws_message(role)
    telemetry.ws_connection(1)
    p = points(reader)
    assert {dp.attributes["role"] for dp in p["ws.messages"]} == {"user", "other"}
    assert p["ws.connections"][0].value == 1


def test_no_prompt_text_in_telemetry(otel):
    exporter, reader = otel
    run(FakeVLM())
    dump = repr([s.attributes for s in exporter.get_finished_spans()]) + repr(points(reader))
    assert PROMPT not in dump
    assert "c.mp4" not in dump


def test_exception_text_stays_out_of_telemetry(otel):
    # Real decode/model errors quote the clip path (and could quote the prompt).
    class LeakyVLM(FakeVLM):
        def forward(self, video_path, fps, max_frames, temperature):
            raise OSError(f"cannot decode {video_path} for {PROMPT}")

    exporter, reader = otel
    run(LeakyVLM())
    (span,) = exporter.get_finished_spans()
    assert span.attributes["error.type"] == "OSError"
    dump = span.to_json() + repr(points(reader))
    assert PROMPT not in dump
    assert "c.mp4" not in dump
