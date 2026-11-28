"""One capture per source. A single thread decodes the selected source; the preview and the clipper share it."""
import os
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Protocol

import cv2
from cv2.typing import MatLike
from loguru import logger

from utils.errors import DecodeError, log_failure
from utils.file_management import BASE_DIR
from video_handling.sampling import MediaTimer, Segment
from video_handling.source import DEFAULT_KEY, MEDIA, WEBCAM_KEY, MediaLibrary, SourceSpec, make_spec

FALLBACK_IMAGE = str(BASE_DIR / "src" / "error.jpeg")
PREVIEW_MAX_HEIGHT = 720
TICK_S = 0.05  # while waiting for the next frame, sinks learn this often that time passes
MAX_LAG_S = 1.0  # further behind than this, playback continues from now instead of catching up
WEBCAM_RETRY_S = 0.05
WEBCAM_TIMEOUT_S = 3.0  # this long without a readable frame is a source error
WEBCAM_GAP_S = 1.0  # a longer pause between webcam frames starts a new segment
MAX_FRAME_EDGES = (1920, 1080)  # long and short edge; larger files decode too slowly to play in real time
MAX_FILE_BYTES = 1 << 30  # 1 GiB: a sample or media file above this is refused before it is opened


class CaptureError(DecodeError, RuntimeError):
    """A video source could not be opened or read.

    `reason` is a fixed text that is safe to show (no path, no decoder output); the message itself is only logged.
    """

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason


class FrameSink(Protocol):
    """Called on the capture thread, so both methods must return quickly."""

    def feed(self, segment: Segment, t: float, frame: MatLike) -> None: ...

    def advance(self, segment: Segment, t: float) -> None: ...


def open_capture(spec: SourceSpec) -> cv2.VideoCapture:
    return cv2.VideoCapture(0 if spec.path is None else spec.path)


def preview_jpeg(frame: MatLike) -> bytes:
    height, width = frame.shape[:2]
    if height > PREVIEW_MAX_HEIGHT:
        size = (round(width * PREVIEW_MAX_HEIGHT / height), PREVIEW_MAX_HEIGHT)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise CaptureError("Could not encode a preview frame")
    return jpeg.tobytes()


def check_file_size(path: str, name: str) -> None:
    """Raise a CaptureError if the file is gone or larger than MAX_FILE_BYTES."""
    try:
        size = os.stat(path).st_size
    except OSError as e:
        raise CaptureError(f"Video file not found: {name}", "the file is gone") from e
    if size > MAX_FILE_BYTES:
        raise CaptureError(f"Video file too large: {name}", f"the file is larger than {MAX_FILE_BYTES >> 20} MB")


def check_frame_size(frame: MatLike, name: str) -> None:
    """Raise a CaptureError if a frame is larger than MAX_FRAME_EDGES (either orientation)."""
    shape = getattr(frame, "shape", None)  # test doubles hand out tokens, not arrays
    if shape is None:
        return
    height, width = shape[:2]
    if max(height, width) > MAX_FRAME_EDGES[0] or min(height, width) > MAX_FRAME_EDGES[1]:
        raise CaptureError(
            f"Video frames too large: {name} ({width}x{height})",
            f"frames of {width}x{height} are larger than {MAX_FRAME_EDGES[0]}x{MAX_FRAME_EDGES[1]}",
        )


class CaptureHub:
    """Owns the only decoder. The selected source loops until another one is selected or it fails.

    After a failure the thread idles until a source is selected. The decoder is opened, read and released on the
    capture thread only. A read that never returns (possible with a webcam) delays a switch until it does.
    """

    def __init__(
        self,
        source_key: str = DEFAULT_KEY,
        *,
        min_pass_s: float = 0.5,
        on_error: Callable[[str], None] | None = None,
        open_source: Callable[[SourceSpec], cv2.VideoCapture] = open_capture,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        wait: Callable[[float], object] | None = None,
        media: MediaLibrary | None = None,
    ):
        self.min_pass_s = min_pass_s  # a shorter file holds its last frame this long, so it still fills a clip
        self.on_error = on_error
        self._open_source = open_source
        self.media = media if media is not None else MEDIA  # the folder `media:<n>` keys are looked up in
        self._clock = clock
        self._wall_clock = wall_clock
        self._wake = threading.Event()  # set by select() and stop(): cuts any wait short
        self._wait = wait or self._wake.wait
        self._cond = threading.Condition()  # guards the spec, the latest frame, the error and `stopped`
        self._spec = make_spec(source_key, 0, self.media)
        self._frame: MatLike | None = None
        self._seq = 0
        self._error: str | None = None
        self._stopped = False
        self._sinks: tuple[FrameSink, ...] = ()
        self._encode_lock = threading.Lock()
        self._encoded: tuple[int, bytes] = (0, b"")
        self._thread: threading.Thread | None = None

    # -- control -----------------------------------------------------------------------------------------------

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def stopped(self) -> bool:
        return self._stopped

    def select(self, source_key: str) -> SourceSpec:
        """Switch sources (selecting the current one restarts it). Raises ValueError for an unknown key."""
        with self._cond:
            self._spec = make_spec(source_key, self._spec.source_epoch + 1, self.media)
            self._error = None
            spec = self._spec
            self._wake.set()
        logger.info(f"Source selected: {spec.source_key} (epoch {spec.source_epoch})")
        return spec

    def toggle(self) -> SourceSpec:
        """The webcam if a file is selected, otherwise the first sample."""
        return self.select(DEFAULT_KEY if self._spec.is_webcam else WEBCAM_KEY)

    def add_sink(self, sink: FrameSink) -> None:
        with self._cond:
            self._sinks = (*self._sinks, sink)

    def remove_sink(self, sink: FrameSink) -> None:
        with self._cond:
            self._sinks = tuple(s for s in self._sinks if s is not sink)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """End the capture thread and every preview stream."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
            self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("Capture thread did not stop within the timeout.")

    # -- preview -----------------------------------------------------------------------------------------------

    def next_preview(self, after_seq: int, timeout: float = 1.0) -> tuple[int, bytes] | None:
        """Wait for a frame newer than `after_seq`; return (seq, JPEG). Each frame is encoded once for all viewers.

        After `timeout` the latest frame is returned again. None: there is no frame yet, or the hub stopped.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._stopped or self._seq > after_seq, timeout)
            if self._stopped or self._frame is None:
                return None
            seq, frame = self._seq, self._frame
        with self._encode_lock:
            if self._encoded[0] < seq:
                self._encoded = (seq, preview_jpeg(frame))
            return self._encoded

    def _show(self, spec: SourceSpec, frame: MatLike) -> bool:
        with self._cond:
            if spec is not self._spec or self._stopped:
                return False
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()
        return True

    # -- capture thread ----------------------------------------------------------------------------------------

    def _current(self, spec: SourceSpec) -> bool:
        return spec is self._spec and not self._stopped

    def _run(self) -> None:
        logger.info("Capture started.")
        while not self._stopped:
            with self._cond:  # together, so that only a later select() or stop() leaves the event set
                self._wake.clear()
                spec = self._spec
            try:
                if spec.is_webcam:
                    self._play_webcam(spec)
                else:
                    self._play_file(spec)
            except Exception as e:  # noqa: BLE001 - capture boundary (cv2.error is no RuntimeError)
                self._fail(spec, e)
                while self._current(spec):
                    self._wake.wait(0.5)
        logger.info("Capture stopped.")

    def _fail(self, spec: SourceSpec, e: Exception) -> None:
        message = f"Capture source failed: {log_failure('Capture source failed', e)}"
        if isinstance(e, CaptureError) and e.reason:
            message += f": {e.reason}"
        with self._cond:
            if not self._current(spec):
                return
            self._error = message
        fallback = cv2.imread(FALLBACK_IMAGE)  # for the preview only: no clips are cut from it
        if fallback is not None:
            self._show(spec, fallback)
        if self.on_error is not None:
            try:
                self.on_error(message)
            except Exception as report_error:  # noqa: BLE001 - reporting must not end the capture thread
                log_failure("Could not report the capture error", report_error)

    def _publish(self, segment: Segment, t: float, frame: MatLike) -> None:
        if self._show(segment.spec, frame):
            for sink in self._sinks:
                sink.feed(segment, t, frame)

    def _advance(self, segment: Segment, t: float) -> None:
        if self._current(segment.spec):
            for sink in self._sinks:
                sink.advance(segment, t)

    def _pace(self, segment: Segment, origin: float, until: float) -> float | None:
        """Wait until `until` seconds into the pass. Returns the (possibly moved) origin, None after a switch."""
        while self._current(segment.spec):
            now = self._clock()
            remaining = origin + until - now
            if remaining <= 0:
                return now - until if -remaining > MAX_LAG_S else origin
            self._wait(min(remaining, TICK_S))
            self._advance(segment, min(self._clock() - origin, until))
        return None

    def _play_file(self, spec: SourceSpec) -> None:
        name = Path(spec.path or "").name  # no directories: they name the user and the install location
        loop = 0
        while self._current(spec):
            check_file_size(spec.path or "", name)
            cap = self._open_source(spec)  # a new decoder per pass: seeking is not reliable in every container
            try:
                if not cap.isOpened():
                    raise CaptureError(f"Failed to open video: {name}", "the file cannot be opened as a video")
                self._play_pass(cap, Segment(spec, loop, loop, 0.0), name)
            finally:
                cap.release()
            loop += 1

    def _play_pass(self, cap: cv2.VideoCapture, segment: Segment, name: str) -> None:
        timer = MediaTimer(cap.get(cv2.CAP_PROP_FPS))
        origin: float | None = None
        last_t, last_step = 0.0, timer.frame_s
        while self._current(segment.spec):
            ok, frame = cap.read()
            if not ok:
                break
            t = timer.stamp(cap.get(cv2.CAP_PROP_POS_MSEC))
            if origin is None:
                check_frame_size(frame, name)
                origin = self._clock()  # after the first decode: opening takes time
            else:
                last_step = t - last_t
                origin = self._pace(segment, origin, t)
                if origin is None:
                    return
            last_t = t
            self._publish(segment, t, frame)
        if not self._current(segment.spec):
            return
        if origin is None:
            raise CaptureError(f"Video yielded no frames: {name}", "no frame could be decoded")
        # The last frame stays on screen as long as the one before it did.
        end = max(last_t + last_step, self.min_pass_s)
        if self._pace(segment, origin, end) is not None:
            self._advance(segment, end)

    def _play_webcam(self, spec: SourceSpec) -> None:
        cap = self._open_source(spec)
        try:
            if not cap.isOpened():
                raise CaptureError("Webcam not found", "no webcam found on the server")
            segment: Segment | None = None
            origin = last_frame = 0.0
            failing_since: float | None = None
            number = 0
            while self._current(spec):
                ok, frame = cap.read()  # paced by the camera
                now = self._clock()
                if not ok:
                    failing_since = now if failing_since is None else failing_since
                    if now - failing_since >= WEBCAM_TIMEOUT_S:
                        raise CaptureError("Webcam stopped delivering frames", "the webcam stopped delivering frames")
                    self._wait(WEBCAM_RETRY_S)
                    continue
                failing_since = None
                if segment is None or now - last_frame > WEBCAM_GAP_S:
                    segment = Segment(spec, number, None, self._wall_clock())
                    origin = now
                    number += 1
                last_frame = now
                self._publish(segment, now - origin, frame)
        finally:
            cap.release()


def mjpeg_stream(hub: CaptureHub) -> Generator[bytes, None, None]:
    """The preview as multipart JPEG parts. It only reads what the capture thread publishes."""
    seq = 0
    try:
        while not hub.stopped:
            item = hub.next_preview(seq)
            if item is None:
                continue
            seq, jpeg = item
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    except (cv2.error, CaptureError) as e:
        log_failure("Video feed ended", e)
