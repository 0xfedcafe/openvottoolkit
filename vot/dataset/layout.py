"""Low-level VOT sequence directory layout.

This module is the single source of truth for the on-disk VOT sequence format: the
``sequence`` metadata file, the ``groundtruth.txt`` annotations and the ``list.txt``
directory index. Higher-level modules (:mod:`vot.dataset.common`,
:mod:`vot.dataset.preparation`, :mod:`vot.dataset.statistics`) build on these primitives.

The actual per-channel frame filename pattern is declared in each sequence's metadata
(``channels.<name>``, e.g. ``color/%05dv.jpg``); :data:`DEFAULT_FRAME_MASK` is only the
pattern the toolkit *writes* for sequences it creates, not a universal constant.

This module deliberately depends only on :mod:`vot.utilities` and ``cv2`` -- never on
``vot.dataset`` itself -- so it can be imported without triggering circular imports.
"""

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from vot.utilities import read_properties, write_properties

logger = logging.getLogger("vot")

#: Printf-style frame filename mask the toolkit writes for sequences it creates. Existing
#: sequences may use any mask; the real pattern is read from the ``channels.*`` metadata.
DEFAULT_FRAME_MASK = "%08d.jpg"
#: Name of the per-sequence metadata file.
METADATA_FILE = "sequence"
#: Name of the single-object groundtruth annotation file.
GROUNDTRUTH_FILE = "groundtruth.txt"
#: Name of the directory index file listing sequence names.
LIST_FILE = "list.txt"
#: Image file extensions recognised when scanning a sequence directory.
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tiff")


def frame_filename(index_1based: int) -> str:
    """Returns a frame filename in the toolkit's default mask for a 1-based index.

    Intended for *writing* newly-created sequences; readers must honour the per-channel
    pattern declared in the sequence metadata instead.

    :param index_1based: The 1-based frame index.

    :returns: The frame filename (e.g. ``00000001.jpg``)."""
    return DEFAULT_FRAME_MASK % index_1based


def list_image_files(directory: str | Path) -> list[Path]:
    """Lists image files in a directory in sorted order, considering common extensions.

    :param directory: The directory to scan.

    :returns: Sorted list of image file paths."""
    directory = Path(directory)
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(f"*.{ext}"))
        files.extend(directory.glob(f"*.{ext.upper()}"))
    files.sort()
    return files


def detect_frame_pattern(directory: str | Path) -> str | None:
    """Infers a printf-style frame filename pattern from the images in a directory.

    Legacy sequences carry no metadata declaring their frame mask, so it is recovered
    from the files on disk: the directory is scanned for an image file whose name is a
    (possibly zero-padded) frame index plus an extension, and the printf mask it follows
    is returned (e.g. ``%08d.png``).

    :param directory: The directory to scan.

    :returns: The frame pattern, or ``None`` if no numbered image files were found."""
    for image in list_image_files(directory):
        if image.stem.isdigit():
            return "%0{}d{}".format(len(image.stem), image.suffix)
    return None


def image_size(path: str | Path) -> tuple[int, int] | None:
    """Returns the ``(width, height)`` of an image file using OpenCV.

    :param path: The image path.

    :returns: The image dimensions, or ``None`` if the file is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def read_metadata(path: str | Path, *, coerce: bool = True, defaults: bool = True) -> dict:
    """Reads the ``sequence`` metadata file from a sequence directory.

    This is the canonical metadata reader shared by sequence loading and sequence preparation.

    :param path: The sequence root directory.
    :param coerce: When True, ``height``/``width``/``length``/``fps`` values are cast to ``int``.
    :param defaults: When True, the ``format`` and ``channel.default`` defaults are applied,
        ``fps`` defaults to 30 and the ``root`` directory is recorded in the result.

    :raises FileNotFoundError: If the metadata file is missing.
    :returns: The metadata mapping."""
    path = Path(path)
    metadata_file = path / METADATA_FILE
    if not metadata_file.is_file():
        raise FileNotFoundError("Sequence metadata file not found: {}".format(metadata_file))

    metadata: dict = {}
    if defaults:
        metadata["format"] = "default"
        metadata["channel.default"] = "color"

    metadata.update(read_properties(str(metadata_file)))

    if coerce:
        for key in ("height", "width", "length", "fps"):
            if key in metadata:
                metadata[key] = int(metadata[key])

    if defaults:
        metadata.setdefault("fps", 30)
        metadata["root"] = str(path)

    return metadata


def write_metadata(path: str | Path, metadata: Mapping) -> None:
    """Writes a ``sequence`` metadata file into a sequence directory.

    :param path: The sequence root directory.
    :param metadata: The metadata mapping to serialise."""
    write_properties(str(Path(path) / METADATA_FILE), dict(metadata))


def channel_keys(metadata: Mapping) -> list[str]:
    """Returns the channel names declared in a sequence metadata mapping.

    Channel names are extracted from the ``channels.<name>`` metadata keys.

    :param metadata: A sequence metadata mapping.

    :returns: Sorted list of channel names (e.g. ``["color", "depth", "ir"]``)."""
    return sorted(key.split(".", 1)[1] for key in metadata
                  if key.startswith("channels.") and "." in key)


class SequenceList:
    """Manager for a directory's ``list.txt`` sequence index.

    ``list.txt`` lists one sequence name per line. This class is the single point through
    which that file is read, written, appended to and removed from.
    """

    def __init__(self, directory: str | Path) -> None:
        """Binds the manager to the ``list.txt`` inside the given directory.

        :param directory: Directory holding (or to hold) the ``list.txt`` file."""
        self._path = Path(directory) / LIST_FILE

    @property
    def path(self) -> Path:
        """Returns the path of the managed ``list.txt`` file."""
        return self._path

    def exists(self) -> bool:
        """Returns whether the ``list.txt`` file exists."""
        return self._path.is_file()

    def read(self) -> list[str] | None:
        """Reads the sequence names listed in ``list.txt``.

        :returns: The list of names (blank lines skipped), or ``None`` if there is no
            ``list.txt`` file."""
        if not self._path.is_file():
            return None
        return [line.strip() for line in self._path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def write(self, names: Iterable[str]) -> None:
        """Writes the given sequence names to ``list.txt``, one per line, overwriting any
        existing content.

        :param names: The sequence names to write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fp:
            for name in names:
                fp.write("{}\n".format(name))

    def append(self, name: str) -> bool:
        """Appends a sequence name as a new line, creating the file if needed.

        A trailing newline is added after a pre-existing last entry that lacks one, so two
        names never end up on the same line. The entry is not added again if it is already
        present.

        :param name: Sequence name to append.

        :returns: True if the name was appended, False if it was already listed."""
        content = ""
        if self._path.exists():
            content = self._path.read_text(encoding="utf-8")
            if name in [line.strip() for line in content.splitlines()]:
                logger.info("Sequence '%s' already listed in %s", name, self._path)
                return False

        prefix = "\n" if content and not content.endswith("\n") else ""
        with open(self._path, "a", encoding="utf-8") as fp:
            fp.write(prefix + name + "\n")

        logger.info("Added '%s' to %s", name, self._path)
        return True

    def remove(self, name: str) -> bool:
        """Removes the line equal to ``name`` from ``list.txt``.

        The file is rewritten without the matching entry; other lines keep their order.

        :param name: Sequence name to remove.

        :returns: True if an entry was removed, False if the file or entry was not found."""
        if not self._path.is_file():
            return False

        lines = self._path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if line.strip() != name]
        if len(kept) == len(lines):
            return False

        with open(self._path, "w", encoding="utf-8") as fp:
            if kept:
                fp.write("\n".join(kept) + "\n")
        logger.info("Removed '%s' from %s", name, self._path)
        return True
