"""What can be captured. A spec never changes; selecting a source makes a new one with a higher epoch."""
from dataclasses import dataclass
from pathlib import Path

from utils.file_management import BASE_DIR

SAMPLE_PATHS = [
    str(BASE_DIR / "static_video" / "street_view_1.mp4"),
    str(BASE_DIR / "static_video" / "street_view_2.mp4"),
]
WEBCAM_KEY = "webcam"
DEFAULT_KEY = "sample:1"


@dataclass(frozen=True)
class SourceSpec:
    source_key: str  # names a selectable source: "sample:<n>" or "webcam"
    source_epoch: int  # counts source changes, so that work started on an earlier source can be told apart
    label: str  # for display; never contains a directory
    path: str | None = None  # None: the webcam attached to the server

    @property
    def is_webcam(self) -> bool:
        return self.path is None


def make_spec(source_key: str, source_epoch: int) -> SourceSpec:
    """Raises ValueError for a key that names no source."""
    if source_key == WEBCAM_KEY:
        return SourceSpec(source_key, source_epoch, "Webcam")
    for number, path in enumerate(SAMPLE_PATHS, start=1):
        if source_key == f"sample:{number}":
            return SourceSpec(source_key, source_epoch, f"Sample {number} ({Path(path).name})", path)
    raise ValueError(f"Unknown source: {source_key!r}")
