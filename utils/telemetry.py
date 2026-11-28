"""Optional OpenTelemetry traces and metrics. Off unless OTEL_ENABLED=1.

Install the extra first: `pip install ".[otel]"`. When disabled, nothing from
`opentelemetry` is imported and every helper below is a no-op. Standard OTEL_* variables apply
(OTEL_TRACES_EXPORTER / OTEL_METRICS_EXPORTER = console | otlp, OTEL_EXPORTER_OTLP_ENDPOINT,
OTEL_SERVICE_NAME, OTEL_METRIC_EXPORT_INTERVAL); the default is `console`, so no infrastructure is needed.
Prompt text, responses, clip paths and frames are never recorded.
"""
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from loguru import logger

_state: "_Telemetry | None" = None


class _Telemetry:
    def __init__(self, tracer_provider: Any, meter_provider: Any):
        self.tracer = tracer_provider.get_tracer("vlm-prompt-sandbox")
        meter = meter_provider.get_meter("vlm-prompt-sandbox")
        self.duration = meter.create_histogram("vlm.inference.duration", unit="s")
        self.errors = meter.create_counter("vlm.inference.errors")
        self.dropped = meter.create_counter("vlm.clips.dropped")
        self.messages = meter.create_counter("ws.messages")
        self.connections = meter.create_up_down_counter("ws.connections")


def enabled() -> bool:
    return os.environ.get("OTEL_ENABLED") == "1"


def configure(tracer_provider: Any, meter_provider: Any) -> None:
    """Activate recording with the given providers (tests pass in-memory ones)."""
    global _state
    _state = _Telemetry(tracer_provider, meter_provider)


def reset() -> None:
    global _state
    _state = None


def _exporter_choice(var: str) -> str:
    return os.environ.get(var, "console").strip().lower()


def setup_telemetry(app: Any) -> None:
    """Build providers from the environment and instrument the FastAPI app. No-op unless enabled."""
    if not enabled():
        return
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.error("OTEL_ENABLED=1 but OpenTelemetry is missing: pip install '.[otel]'")
        return

    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "vlm-prompt-sandbox")})

    if _exporter_choice("OTEL_TRACES_EXPORTER") == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        span_exporter: Any = OTLPSpanExporter()
    else:
        span_exporter = ConsoleSpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    if _exporter_choice("OTEL_METRICS_EXPORTER") == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        metric_exporter: Any = OTLPMetricExporter()
    else:
        metric_exporter = ConsoleMetricExporter()
    meter_provider = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(metric_exporter)])

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    configure(tracer_provider, meter_provider)
    # /video_feed is one endless MJPEG response; skip per-message WebSocket receive/send spans.
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        excluded_urls="video_feed",
        exclude_spans=["receive", "send"],
    )
    logger.info("OpenTelemetry enabled.")


@contextmanager
def inference(backend: str, fps: int, max_frames: int, temperature: float) -> Iterator[None]:
    """Span + duration histogram + error counter around one agent call. Re-raises errors."""
    state = _state
    if state is None:
        yield
        return
    attrs = {"vlm.backend": backend, "vlm.fps": fps, "vlm.max_frames": max_frames, "vlm.temperature": temperature}
    outcome = "error"
    start = time.perf_counter()
    try:
        # The SDK's default error handling copies the exception message and stack trace onto the span, and
        # those can quote clip paths or prompt text. Record the exception type only.
        with state.tracer.start_as_current_span(
            "vlm.inference", attributes=attrs, record_exception=False, set_status_on_exception=False
        ) as span:
            try:
                yield
            except Exception as e:
                from opentelemetry.trace import Status, StatusCode

                span.set_attribute("error.type", type(e).__name__)
                span.set_status(Status(StatusCode.ERROR, type(e).__name__))
                raise
        outcome = "ok"
    finally:
        state.duration.record(time.perf_counter() - start, {"vlm.backend": backend, "outcome": outcome})
        if outcome == "error":
            state.errors.add(1, {"vlm.backend": backend})


def record_drop() -> None:
    if _state is not None:
        _state.dropped.add(1)


def record_ws_message(kind: object) -> None:
    if _state is not None:
        from utils.session import CLIENT_TYPES

        # Client-controlled: collapse to the known frame types plus "other", so the label cannot explode.
        _state.messages.add(1, {"type": kind if kind in CLIENT_TYPES else "other"})


def ws_connection(delta: int) -> None:
    if _state is not None:
        _state.connections.add(delta)
