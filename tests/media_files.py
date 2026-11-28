"""Video files for tests, generated into a test's tmp dir. Nothing here is committed or downloaded.

`mediarecorder_webm` writes what a browser's MediaRecorder produces, as far as the decoder can tell: a WebM whose
Segment and Clusters have unknown size, with no Duration, no Cues, no frame rate (DefaultDuration) and uneven
millisecond timestamps. OpenCV encodes the VP8 frames; this module re-muxes them by hand (EBML is simple enough).
"""
from pathlib import Path

import cv2
import numpy as np

# Matroska / WebM element ids (https://www.matroska.org/technical/elements.html)
EBML_HEADER = 0x1A45DFA3
SEGMENT = 0x18538067
INFO = 0x1549A966
TIMESTAMP_SCALE = 0x2AD7B1
MUXING_APP = 0x4D80
WRITING_APP = 0x5741
TRACKS = 0x1654AE6B
TRACK_ENTRY = 0xAE
DEFAULT_DURATION = 0x23E383
CLUSTER = 0x1F43B675
CLUSTER_TIMESTAMP = 0xE7
SIMPLE_BLOCK = 0xA3
BLOCK_GROUP = 0xA0
BLOCK = 0xA1
UNKNOWN_SIZE = bytes([0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])


def write_video(path: Path, n_frames: int, fps: float = 20, size: tuple[int, int] = (32, 32),
                fourcc: str = "mp4v") -> Path:
    """Frame i is a flat grey of brightness i * 12 (mod 255). `size` is (width, height)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"this OpenCV build cannot write {fourcc}")
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 12 % 255, dtype=np.uint8))
    writer.release()
    return path


def corrupt(path: Path) -> Path:
    """Random bytes with a video name."""
    path.write_bytes(np.random.default_rng(7).integers(0, 256, 64_000, dtype=np.uint8).tobytes())
    return path


def truncated(path: Path, n_frames: int = 40) -> Path:
    """The first fifth of a valid MP4. OpenCV's MP4 writer puts the index (moov) last, so it is cut off."""
    write_video(path, n_frames)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 5])
    return path


def zero_frames(path: Path) -> Path:
    """An MP4 writer opened and closed without a frame."""
    return write_video(path, 0)


# -- EBML ------------------------------------------------------------------------------------------------------


def _read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int]:
    first = data[pos]
    length = 1
    while length <= 8 and not first & (0x80 >> (length - 1)):
        length += 1
    value = first if keep_marker else first & (0xFF >> length)
    for byte in data[pos + 1: pos + length]:
        value = (value << 8) | byte
    return value, pos + length


def _children(data: bytes) -> list[tuple[int, bytes]]:
    """(id, payload) of each element in `data`. Unknown sizes are not expected in OpenCV's output."""
    out, pos = [], 0
    while pos < len(data):
        element_id, pos = _read_vint(data, pos, keep_marker=True)
        size, pos = _read_vint(data, pos, keep_marker=False)
        out.append((element_id, data[pos: pos + size]))
        pos += size
    return out


def _id_bytes(element_id: int) -> bytes:
    return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big")


def _size(n: int) -> bytes:
    length = 1
    while n >= (1 << (7 * length)) - 1:
        length += 1
    return ((1 << (7 * length)) | n).to_bytes(length, "big")


def _element(element_id: int, payload: bytes, unknown_size: bool = False) -> bytes:
    return _id_bytes(element_id) + (UNKNOWN_SIZE if unknown_size else _size(len(payload))) + payload


def _uint(element_id: int, value: int) -> bytes:
    return _element(element_id, value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big"))


def _frames(segment: bytes) -> list[tuple[bytes, int]]:
    """(track number vint + flags-free block data, flags) for every video block, in order."""
    frames = []
    for element_id, payload in _children(segment):
        if element_id != CLUSTER:
            continue
        for child_id, block in _children(payload):
            if child_id == BLOCK_GROUP:
                block = next(p for i, p in _children(block) if i == BLOCK)
                flags = 0x80  # a Block carries no keyframe flag; mark it as SimpleBlock would
            elif child_id == SIMPLE_BLOCK:
                flags = None
            else:
                continue
            _track, pos = _read_vint(block, 0, keep_marker=False)
            flags = block[pos + 2] if flags is None else flags
            frames.append((block[pos + 3:], flags))
    return frames


def mediarecorder_webm(path: Path, times_ms: list[int], size: tuple[int, int] = (64, 48)) -> Path:
    """A browser-recording-shaped WebM whose frame i has the timestamp times_ms[i] (and brightness i * 12)."""
    source = path.with_suffix(".opencv.webm")
    write_video(source, len(times_ms), fps=30, size=size, fourcc="VP80")
    elements = dict(_children(source.read_bytes()))
    source.unlink()
    segment = elements[SEGMENT]
    parts = dict(_children(segment))
    entry = next(p for i, p in _children(parts[TRACKS]) if i == TRACK_ENTRY)
    entry = b"".join(_element(i, p) for i, p in _children(entry) if i != DEFAULT_DURATION)
    info = _uint(TIMESTAMP_SCALE, 1_000_000) + _element(MUXING_APP, b"Chrome") + _element(WRITING_APP, b"Chrome")
    frames = _frames(segment)
    assert len(frames) == len(times_ms), (len(frames), len(times_ms))
    clusters = []
    for start in range(0, len(frames), 10):  # MediaRecorder starts a new cluster every few frames
        base = times_ms[start]
        blocks = b"".join(
            _element(SIMPLE_BLOCK, b"\x81" + (t - base).to_bytes(2, "big", signed=True) + bytes([flags]) + data)
            for (data, flags), t in zip(frames[start: start + 10], times_ms[start: start + 10], strict=True)
        )
        clusters.append(_element(CLUSTER, _uint(CLUSTER_TIMESTAMP, base) + blocks, unknown_size=True))
    body = _element(INFO, info) + _element(TRACKS, _element(TRACK_ENTRY, entry)) + b"".join(clusters)
    path.write_bytes(_element(EBML_HEADER, elements[EBML_HEADER]) + _element(SEGMENT, body, unknown_size=True))
    return path
