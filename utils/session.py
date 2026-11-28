"""The comparison session: clips in, rounds out, client messages handled, `state` kept current.

It joins the agent registry, the round runner, spend control and the capture hub. Frames go out through `broadcast`
(to every browser); `handle` returns the frames meant only for the sender. Client handlers must run off the server's
event loop (the web layer uses a worker thread): they take the session lock and broadcast while holding it, so a
source switch and a result can never cross (the epoch is read and the frame sent under one lock).
"""
import base64
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from utils import telemetry
from utils.providers import PROVIDERS, GenerateRequest
from utils.providers.agents import AgentConfigError
from utils.providers.pricing import price_date
from utils.providers.registry import SLOTS, AgentRegistry, Slot
from utils.providers.settings import ANTHROPIC_URL, OPENAI_URL, check_model_id, is_local, ProviderConfigError
from utils.rounds import CALL_TIMEOUT_S, AgentOutcome, RoundReport, RoundRunner, RoundSettings
from utils.spend import ACTIONS, SpendController
from video_handling.clip_handling import ClipSample
from video_handling.source import SAMPLE_PATHS, WEBCAM_KEY, make_spec

CLIENT_TYPES = ("update", "control", "set_agent", "list_models", "set_source")
SERVER_TYPES = ("state", "round", "result", "round_end", "models", "system", "error")
THUMB_EDGE = 160  # px, long edge of the frame thumbnails in the `round` frame
THUMB_QUALITY = 70
CONFIRM_WEBCAM = (
    "Frames from the server's webcam would be sent to a remote provider. Confirm to continue."
)
_DEFAULT_HOSTS = {"openai": urlsplit(OPENAI_URL).hostname, "anthropic": urlsplit(ANTHROPIC_URL).hostname}


def message(kind: str, text: str, **extra: Any) -> dict:
    """A `system` or `error` frame."""
    return {"type": kind, "text": text, **extra}


def thumbnail(jpeg: bytes) -> str | None:
    """A small JPEG data URL of one frame, or None if it cannot be decoded."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = THUMB_EDGE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii") if ok else None


def destination(slot: Slot) -> str | None:
    """Where this slot's frames go: a host name, or None when they stay on this machine."""
    if slot.local:
        return None
    base_url = getattr(slot.agent, "base_url", None)
    host = urlsplit(base_url).hostname if isinstance(base_url, str) else None
    return host or _DEFAULT_HOSTS.get(slot.spec.provider) or slot.spec.provider


def _delete(path: str) -> None:
    Path(path).unlink(missing_ok=True)


class CompareSession:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        spend: SpendController,
        hub: Any,
        prompt: Any,
        settings: Any,
        broadcast: Callable[[dict], None],
        clip_fps: int,
        max_frames: int,
        call_timeout_s: float = CALL_TIMEOUT_S,
        delete_clip: Callable[[str], None] = _delete,
    ):
        self.registry = registry
        self.spend = spend
        self.hub = hub
        self.prompt = prompt
        self.settings = settings
        self._broadcast = broadcast
        self.clip_fps = clip_fps
        self.max_frames = max_frames
        self._delete_clip = delete_clip
        self._lock = threading.RLock()
        self.busy_drops = 0
        self.runner = RoundRunner(
            current_epoch=lambda: self.hub.spec.source_epoch,
            on_round_start=self._round_start,
            on_result=self._result,
            on_round_end=self._round_end,
            on_late=self._late,
            call_timeout_s=call_timeout_s,
            delete_clip=delete_clip,
        )

    # -- state -----------------------------------------------------------------------------------------------------

    def state(self) -> dict:
        spec = self.hub.spec
        sources = [make_spec(f"sample:{n}", 0) for n in range(1, len(SAMPLE_PATHS) + 1)] + [make_spec(WEBCAM_KEY, 0)]
        return {
            "type": "state",
            "providers": [
                {"id": a.provider, "available": a.available, "reason": a.reason} for a in self.registry.providers()
            ],
            "slots": [
                {
                    "slot": s.name, "provider": s.spec.provider, "model": s.agent.model, "label": s.agent.label,
                    "enabled": s.enabled, "remote": s.remote, "destination": destination(s),
                }
                for s in self.registry.snapshot()
            ],
            "run": self.spend.snapshot(),
            "settings": {
                "prompt": self.prompt.get(),
                "temperature": self.settings.get_temperature(),
                "resample_fps": self.settings.get_resample_fps(),
                "effective_fps": self.settings.effective_resample_fps(),
            },
            "source": {"key": spec.source_key, "label": spec.label, "epoch": spec.source_epoch,
                       "error": self.hub.error},
            "sources": [{"key": s.source_key, "label": s.label} for s in sources],
            "price_date": price_date(),
            "busy_drops": self.busy_drops,
        }

    def broadcast_state(self) -> None:
        with self._lock:
            self._broadcast(self.state())

    def source_error(self, text: str) -> None:
        """The capture hub reports a source error (once per source)."""
        with self._lock:
            self._broadcast(message("error", text))
            self._broadcast(self.state())

    # -- clips and rounds ------------------------------------------------------------------------------------------

    def on_clip(self, clip: ClipSample) -> None:
        """The recorder's flush callback: start a round, or delete the clip."""
        with self._lock:
            if clip.source_epoch != self.hub.spec.source_epoch:
                self._delete_clip(clip.video_path)
                return
            if self.runner.busy:  # lockstep: the slower agent sets the pace
                self.busy_drops += 1
                telemetry.record_drop()
                self._delete_clip(clip.video_path)
                return
            slots = self.registry.snapshot()
            active = [s for s in slots if s.enabled]
            if not active:
                self._delete_clip(clip.video_path)
                return
            before = self.spend.snapshot()
            admission = self.spend.admit(any(s.remote for s in active))
            if not admission.start:  # paused, pacing or budget: not a busy drop
                self._delete_clip(clip.video_path)
                self._state_if_changed(before)
                return
            settings = RoundSettings(
                prompt=self.prompt.get(),
                temperature=self.settings.get_temperature(),
                resample_fps=self.settings.get_resample_fps(),
                clip_fps=self.clip_fps,
                max_frames=self.max_frames,
            )
            logger.info(
                f"Clip {clip.clip_id}: {clip.source_key} (epoch {clip.source_epoch}), "
                f"{clip.start_s:.1f}-{clip.end_s:.1f} s, to {', '.join(s.agent.label for s in active)}"
            )
            if not self.runner.start(clip, slots, settings):
                self._delete_clip(clip.video_path)
            self._state_if_changed(before)

    def _state_if_changed(self, before: dict) -> None:
        after = self.spend.snapshot()
        keys = ("running", "step_pending", "pause_reason", "rounds_used", "exhausted")
        if any(before[k] != after[k] for k in keys):
            self._broadcast(self.state())

    def _round_start(self, round_id: int, clip: ClipSample, request: GenerateRequest) -> None:
        thumbs = [t for t in (thumbnail(f) for f in request.frames) if t is not None]
        with self._lock:
            self._broadcast({
                "type": "round", "round_id": round_id, "clip_id": clip.clip_id, "source_epoch": clip.source_epoch,
                "source_key": clip.source_key, "source_label": clip.label, "start_s": clip.start_s,
                "end_s": clip.end_s, "loop": clip.loop, "prompt": request.prompt,
                "frame_times": list(request.frame_times), "thumbnails": thumbs,
                "slots": [s.name for s in self.registry.snapshot() if s.enabled],
            })

    def _result(self, outcome: AgentOutcome) -> None:
        with self._lock:
            stale = outcome.stale or outcome.source_epoch != self.hub.spec.source_epoch
            before = self.spend.snapshot()
            self._record(outcome)
            self._broadcast(result_frame(outcome, stale))
            self._state_if_changed(before)

    def _late(self, outcome: AgentOutcome) -> None:
        with self._lock:
            cost = outcome.result.cost_usd if outcome.result is not None else None
            self.spend.record_late(outcome.remote, cost)
            self._broadcast(self.state())

    def _record(self, outcome: AgentOutcome) -> None:
        cost = outcome.result.cost_usd if outcome.result is not None else None
        self.spend.record(outcome.slot, outcome.status, outcome.remote, cost, outcome.error_kind)

    def _round_end(self, report: RoundReport) -> None:
        with self._lock:
            self._broadcast({
                "type": "round_end", "round_id": report.round_id, "clip_id": report.clip.clip_id,
                "statuses": {o.slot: o.status for o in report.outcomes},
            })
            self._broadcast(self.state())

    # -- client messages -------------------------------------------------------------------------------------------

    def handle(self, msg: dict) -> list[dict]:
        """One parsed client frame (a dict with a known `type`). Returns frames for the sender only."""
        kind = msg.get("type")
        try:
            if kind == "update":
                return self._update(msg)
            if kind == "control":
                return self._control(msg)
            if kind == "set_agent":
                return self._set_agent(msg)
            if kind == "list_models":
                return self._list_models(msg)
            if kind == "set_source":
                return self._set_source(msg)
        except _BadRequest as e:
            return [message("error", str(e), request=kind)]
        return [message("error", f"Unknown message type: {kind!r}")]

    def disconnected(self, remaining: int) -> None:
        """A browser left. Quick and lock-free, so the web layer can call it on the event loop even while its
        handler is being cancelled. With nobody left there is no one to tell; with someone left nothing changed."""
        self.spend.disconnected(remaining)

    def stop(self, timeout: float = 5.0) -> bool:
        return self.runner.stop(timeout)

    def _update(self, msg: dict) -> list[dict]:
        text, temperature, fps = msg.get("text"), msg.get("temperature"), msg.get("resample_fps")
        if text is not None and not isinstance(text, str):
            raise _BadRequest("text must be a string")
        try:
            if temperature is not None:
                self.settings.set_temperature(temperature)
            if fps is not None:
                self.settings.set_resample_fps(fps)
        except ValueError as e:
            raise _BadRequest(str(e)) from e
        if text and text.strip():
            self.prompt.set(text.strip())
        self.broadcast_state()
        return [message("system", "Settings updated: the next round uses them")]

    def _control(self, msg: dict) -> list[dict]:
        action = msg.get("action")
        if action not in ACTIONS:
            raise _BadRequest(f"action must be one of: {', '.join(ACTIONS)}")
        with self._lock:
            refused = self.spend.control(action, self.registry.any_remote())
            if refused is None:
                self._broadcast(self.state())
        return [] if refused is None else [message("error", refused, request="control")]

    def _set_agent(self, msg: dict) -> list[dict]:
        slot, provider, model = msg.get("slot"), msg.get("provider"), msg.get("model")
        enabled, confirm = msg.get("enabled", True), msg.get("confirm", False)
        if slot not in SLOTS or provider not in PROVIDERS or not isinstance(enabled, bool):
            raise _BadRequest(f"set_agent needs slot (A or B), provider (one of: {', '.join(PROVIDERS)}) "
                              "and a boolean enabled")
        if model is not None and not isinstance(model, str):
            raise _BadRequest("model must be a string")
        model = model.strip() if model else None
        try:
            if model:
                check_model_id(model)
        except ProviderConfigError as e:
            raise _BadRequest(str(e)) from e
        with self._lock:
            remote = not is_local(provider, model or "")
            if enabled and remote and self.hub.spec.is_webcam and confirm is not True:
                return [message("error", CONFIRM_WEBCAM, request="set_agent", confirm="webcam_remote")]
            try:
                self.registry.set_agent(slot, provider, model, enabled)
            except AgentConfigError as e:
                raise _BadRequest(str(e)) from e
            self._broadcast(self.state())
        return []

    def _list_models(self, msg: dict) -> list[dict]:
        provider, refresh = msg.get("provider"), msg.get("refresh", False)
        if not isinstance(provider, str) or not isinstance(refresh, bool):
            raise _BadRequest("list_models needs provider and a boolean refresh")
        listing = self.registry.list_models(provider, refresh=refresh)  # may call the provider: no lock held
        return [{
            "type": "models", "provider": listing.provider, "error": listing.error,
            "models": [{"id": m.id, "vision": m.vision} for m in listing.models],
        }]

    def _set_source(self, msg: dict) -> list[dict]:
        key, confirm = msg.get("source_key"), msg.get("confirm", False)
        if not isinstance(key, str):
            raise _BadRequest("source_key must be a string")
        with self._lock:
            if key == WEBCAM_KEY and self.registry.any_remote() and confirm is not True:
                return [message("error", CONFIRM_WEBCAM, request="set_source", confirm="webcam_remote")]
            try:
                self.hub.select(key)
            except ValueError as e:
                raise _BadRequest(str(e)) from e
            self._broadcast(self.state())
        return []


class _BadRequest(ValueError):
    """A client frame with a bad field; the message is safe to send back."""


def result_frame(outcome: AgentOutcome, stale: bool) -> dict:
    result = outcome.result
    metrics: dict[str, Any] = {"latency_ms": round(outcome.elapsed_ms, 1)}
    if result is not None:
        metrics = {
            "latency_ms": round(result.latency_ms, 1),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "cost_usd": result.cost_usd,
            "cost_source": result.cost_source,
            "requested_model": result.requested_model,
            "reported_model": result.reported_model,
            "params_sent": result.params_sent,
        }
    return {
        "type": "result", "round_id": outcome.round_id, "clip_id": outcome.clip_id,
        "source_epoch": outcome.source_epoch, "slot": outcome.slot, "label": outcome.label, "remote": outcome.remote,
        "status": outcome.status, "stale": stale,
        "text": result.text if result is not None else None,
        "finish_reason": result.finish_reason if result is not None else None,
        "error": None if outcome.error_kind is None and outcome.error_message is None
        else {"kind": outcome.error_kind, "message": outcome.error_message},
        "metrics": metrics,
    }
