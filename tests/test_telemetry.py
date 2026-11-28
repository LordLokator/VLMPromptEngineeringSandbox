import subprocess
import sys

import pytest

from utils import telemetry
from utils.providers.agents import AgentSpec
from utils.providers.registry import Slot
from utils.rounds import RoundRunner, RoundSettings
from video_handling.clip_handling import ClipSample

PROMPT = "SECRET-PROMPT-TEXT"


def sample(path):
    return ClipSample(0, "sample:1", 0, "Sample 1", 0.0, 0.5, 0, str(path), (b"x",) * 5, (0.0,) * 5)


def run(agent):
    """One round with one agent: the span and metrics per agent call come from the round runner."""
    runner = RoundRunner(current_epoch=lambda: 0, delete_clip=lambda _p: None)
    slot = Slot("A", AgentSpec(agent.provider, agent.model), agent, True, False)
    runner.run(sample("c.mp4"), [slot], RoundSettings(PROMPT, 0.5, 3, 10))


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


def test_inference_span_and_histogram_ok(otel, make_agent):
    exporter, reader = otel
    run(make_agent(model="fake-model"))
    (span,) = exporter.get_finished_spans()
    assert span.name == "vlm.inference"
    assert span.attributes["vlm.backend"] == "demo:fake-model"
    assert span.attributes["vlm.fps"] == 3
    assert span.attributes["vlm.temperature"] == 0.5
    (hist,) = points(reader)["vlm.inference.duration"]
    assert hist.attributes["outcome"] == "ok"
    assert "vlm.inference.errors" not in points(reader)


def test_error_sets_status_and_counts(otel, make_agent):
    from opentelemetry.trace import StatusCode

    exporter, reader = otel
    run(make_agent(raise_on={1: RuntimeError("boom")}))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    p = points(reader)
    assert p["vlm.inference.errors"][0].value == 1
    assert p["vlm.inference.duration"][0].attributes["outcome"] == "error"


def test_drop_counter_increments_when_busy(otel, compare_app):
    _, reader = otel
    compare_app.options = {"hang_on": {1}}
    compare_app.registry.set_agent("A", "demo", "frame-motion")
    session = compare_app.session
    session.spend.control("run", remote=False)
    session.on_clip(sample("a.mp4"))
    assert compare_app.agents["demo:frame-motion"].started.wait(5)
    session.on_clip(sample("b.mp4"))  # the first round is still running
    for agent in compare_app.agents.values():
        agent.release.set()
    assert points(reader)["vlm.clips.dropped"][0].value == 1


def test_ws_type_label_is_bounded(otel):
    _, reader = otel
    for kind in ("control", "set_agent", "evil-1", "evil-2", None):
        telemetry.record_ws_message(kind)
    telemetry.ws_connection(1)
    p = points(reader)
    assert {dp.attributes["type"] for dp in p["ws.messages"]} == {"control", "set_agent", "other"}
    assert p["ws.connections"][0].value == 1


def test_no_prompt_text_in_telemetry(otel, make_agent):
    exporter, reader = otel
    run(make_agent(replies=[PROMPT]))  # an answer that quotes the prompt
    dump = repr([s.attributes for s in exporter.get_finished_spans()]) + repr(points(reader))
    assert PROMPT not in dump
    assert "c.mp4" not in dump


def test_exception_text_stays_out_of_telemetry(otel, make_agent):
    # Real decode/model errors quote the clip path (and could quote the prompt).
    exporter, reader = otel
    run(make_agent(raise_on={1: OSError(f"cannot decode c.mp4 for {PROMPT}")}))
    (span,) = exporter.get_finished_spans()
    assert span.attributes["error.type"] == "OSError"
    dump = span.to_json() + repr(points(reader))
    assert PROMPT not in dump
    assert "c.mp4" not in dump


def test_route_error_text_stays_off_the_server_span(compare_app, monkeypatch):
    """With the FastAPI instrumentation on, an unhandled route error must leave its message nowhere in the span."""
    pytest.importorskip("opentelemetry.instrumentation.fastapi")
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import web_engine

    sentinel = "/private/path/" + "sentinel-route-error"

    def broken():
        raise OSError(f"cannot read {sentinel}")

    monkeypatch.setattr(web_engine, "load_presets", broken)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = web_engine.app
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    app.middleware_stack = None  # rebuilt on the next request, now with the instrumentation
    try:
        response = compare_app.client.get("/preset_buttons")
    finally:
        FastAPIInstrumentor.uninstrument_app(app)
    assert response.status_code == 500 and response.text == "Internal Server Error"
    spans = exporter.get_finished_spans()
    assert spans
    dump = " ".join(span.to_json() for span in spans)
    assert sentinel not in dump and "cannot read" not in dump
