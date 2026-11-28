"""What can be captured. A spec never changes; selecting a source makes a new one with a higher epoch.

Selectable sources: the bundled samples, the `.mp4` / `.webm` files directly in `media/`, and the server webcam. A
client only ever sends a key the server issued (`sample:<n>`, `media:<n>`, `webcam`); no path or file name is taken
from a client, so a key cannot name a file outside these two folders.
"""
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from utils.file_management import BASE_DIR

SAMPLE_PATHS = [
    str(BASE_DIR / "static_video" / "street_view_1.mp4"),
    str(BASE_DIR / "static_video" / "street_view_2.mp4"),
]
MEDIA_DIR = BASE_DIR / "media"  # git-ignored; the user copies files here
MEDIA_SUFFIXES = (".mp4", ".webm")
MEDIA_MAX_FILES = 100  # listed per scan, in name order; keeps the `state` frame small
WEBCAM_KEY = "webcam"
DEFAULT_KEY = "sample:1"
_MEDIA_KEY = re.compile(r"media:([1-9][0-9]{0,5})")


@dataclass(frozen=True)
class SourceSpec:
    source_key: str  # names a selectable source: "sample:<n>", "media:<n>" or "webcam"
    source_epoch: int  # counts source changes, so that work started on an earlier source can be told apart
    label: str  # for display; never contains a directory
    path: str | None = None  # None: the webcam attached to the server

    @property
    def is_webcam(self) -> bool:
        return self.path is None


class MediaLibrary:
    """The video files directly in one folder. Each file name keeps its number for the life of the process.

    So `media:3` names the same file before and after a rescan, and a key one client saw can never select a different
    file that another client's rescan brought in. A file that disappeared is no longer listed or selectable.
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self._lock = threading.Lock()
        self._numbers: dict[str, int] = {}
        self._listed: dict[int, str] = {}

    def scan(self) -> bool:
        """Read the folder again. Returns True if the listing changed. A missing folder lists nothing."""
        names = sorted(self._names())[:MEDIA_MAX_FILES]
        with self._lock:
            for name in names:
                self._numbers.setdefault(name, len(self._numbers) + 1)
            listed = {self._numbers[name]: name for name in names}
            changed = listed != self._listed
            self._listed = listed
        return changed

    def _names(self) -> list[str]:
        try:
            entries = list(os.scandir(self.folder))
        except OSError:  # missing or unreadable: nothing to list
            return []
        # Regular files only, not links: a link could point anywhere. Names come from the folder, never a client.
        return [
            e.name for e in entries
            if e.is_file(follow_symlinks=False) and not e.name.startswith(".")
            and e.name.lower().endswith(MEDIA_SUFFIXES)
        ]

    def listing(self) -> list[tuple[int, str]]:
        """(number, file name) of the last scan, in name order."""
        with self._lock:
            return sorted(self._listed.items(), key=lambda item: item[1])

    def path(self, number: int) -> Path | None:
        with self._lock:
            name = self._listed.get(number)
        if name is None:
            return None
        path = self.folder / name
        return path if path.parent == self.folder else None  # a listed name has no separator; checked anyway


MEDIA = MediaLibrary(MEDIA_DIR)


def make_spec(source_key: str, source_epoch: int, media: MediaLibrary | None = None) -> SourceSpec:
    """Raises ValueError for a key that names no source (a `media:` key only as of the last scan)."""
    if source_key == WEBCAM_KEY:
        return SourceSpec(source_key, source_epoch, "Webcam")
    for number, path in enumerate(SAMPLE_PATHS, start=1):
        if source_key == f"sample:{number}":
            return SourceSpec(source_key, source_epoch, f"Sample {number} ({Path(path).name})", path)
    match = _MEDIA_KEY.fullmatch(source_key)
    if match:
        file = (media or MEDIA).path(int(match.group(1)))
        if file is not None:
            return SourceSpec(source_key, source_epoch, f"{file.name} (media)", str(file))
    raise ValueError(f"Unknown source: {source_key!r}")


def selectable(media: MediaLibrary | None = None) -> list[SourceSpec]:
    """Every source a client may select, as of the last media scan: samples, media files, the webcam."""
    keys = [f"sample:{n}" for n in range(1, len(SAMPLE_PATHS) + 1)]
    keys += [f"media:{number}" for number, _name in (media or MEDIA).listing()]
    return [make_spec(key, 0, media) for key in [*keys, WEBCAM_KEY]]
