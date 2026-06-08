"""Sequence preparation: creating and transforming VOT-format sequences on disk.

Library functions for importing external footage as VOT sequences and for transforming
existing sequences (slicing, subsampling, reversing, speed-up variants). These operate on
sequence *directories* and are exposed as the ``vot sequences <subcommand>`` CLI.

The frame transforms (:func:`take_slice`, :func:`subsample_sequence`, :func:`reverse_sequence`,
:func:`delayed_init_variants`) build on the dataset object layer: a sequence is loaded with
:func:`vot.dataset.load_sequence`, frame-remapped with :class:`vot.dataset.proxy.FrameMapSequence`
and written back with :func:`vot.dataset.common.write_sequence`. This makes them honour the real
per-channel frame pattern declared in each sequence's metadata and handle multi-channel,
multi-object sequences, tags and values transparently.
"""

import logging
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence as TypingSequence

from vot.dataset import Sequence, load_sequence
from vot.region import Special, SpecialCode
from vot.dataset.common import write_sequence
from vot.dataset.layout import (DEFAULT_FRAME_MASK, GROUNDTRUTH_FILE, METADATA_FILE,
                                SequenceList, channel_keys, frame_filename, image_size,
                                list_image_files, read_metadata, write_metadata)
from vot.dataset.proxy import FrameMapSequence
from vot.dataset.statistics import find_size_range_windows, verify_slice

logger = logging.getLogger("vot")


def _sequence_length(sequence_dir: str | Path) -> int:
    """Returns the frame count of a sequence directory, inferred via the dataset loader.

    :param sequence_dir: The sequence root directory.

    :returns: The number of frames."""
    return len(load_sequence(str(sequence_dir)))


def _rewrite_sequence(source: Sequence, dest_dir: str | Path, frame_map: Iterable[int],
                      preserve_layout: bool = False) -> Path:
    """Writes a frame-remapped copy of a loaded sequence to a new directory.

    The remapped sequence is produced with :class:`FrameMapSequence` and written with
    :func:`write_sequence`, so all channels, objects, tags and values follow the remapping and
    the real per-channel frame pattern is honoured.

    :param source: The loaded source sequence.
    :param dest_dir: Destination directory for the rewritten sequence.
    :param frame_map: 0-based source frame indices, in output order.
    :param preserve_layout: When True, reproduce the source's exact per-channel subdirectory and
        filename pattern instead of normalizing to ``<channel>/%08d.jpg``. Defaults to False.

    :raises ValueError: If the frame selection is empty.
    :returns: The destination directory."""
    dest_dir = Path(dest_dir)
    proxy = FrameMapSequence(source, [int(i) for i in frame_map])
    if len(proxy) == 0:
        raise ValueError("Frame selection is empty for sequence '{}'".format(source.name))
    write_sequence(str(dest_dir), proxy, preserve_layout=preserve_layout)
    logger.info("Wrote %s (%d frames)", dest_dir, len(proxy))
    return dest_dir


def extract_frames(video_path: str | Path, output_dir: str | Path, fps: float | None = None,
                   start_frame: int = 1, quality: int = 2, channel: str = "color") -> int:
    """Extracts frames from a video file into a directory using ffmpeg.

    Requires ``ffmpeg`` on PATH. Frames are written with the default frame mask starting from
    ``start_frame``. When ``channel`` is set the frames are nested under a ``<channel>/``
    subdirectory (e.g. ``color/%08d.jpg``), matching the layout produced by
    :func:`vot.dataset.common.write_sequence`; pass an empty ``channel`` to write them flat in
    ``output_dir``.

    :param video_path: Path to the input video.
    :param output_dir: Directory to write extracted frames to; created if missing.
    :param fps: Target frame rate. ``None`` keeps the source rate.
    :param start_frame: First frame number in the output filename sequence. Defaults to 1.
    :param quality: ffmpeg ``-q:v`` value (1 = best, 5 = worst). Defaults to 2.
    :param channel: Channel subdirectory to nest frames under. Defaults to ``color``;
        an empty string writes the frames directly into ``output_dir``.

    :returns: Number of frames written.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frame_dir = output_dir / channel if channel else output_dir
    frame_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["ffmpeg", "-i", str(video_path)]
    if fps:
        cmd.extend(["-vf", "fps={}".format(fps)])
    cmd.extend(["-q:v", str(quality), "-start_number", str(start_frame),
                str(frame_dir / DEFAULT_FRAME_MASK)])

    logger.info("Extracting frames from %s into %s", video_path, frame_dir)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    extracted = sorted(frame_dir.glob("*.jpg"))
    logger.info("Extracted %d frames", len(extracted))
    return len(extracted)


def write_sequence_metadata(sequence_dir: str | Path, fps: int | float = 30,
                            width: int | None = None, height: int | None = None,
                            length: int | None = None, force: bool = False,
                            channel: str = "color") -> bool:
    """Writes a VOT-format ``sequence`` metadata file in a sequence directory.

    Missing ``width``/``height``/``length`` are inferred from on-disk frames or ``groundtruth.txt``
    when present. Frames are expected under a ``<channel>/`` subdirectory (the layout written by
    :func:`extract_frames` / :func:`vot.dataset.common.write_sequence`); pass an empty ``channel``
    for a flat single-folder sequence.

    :param sequence_dir: The sequence root directory.
    :param fps: Frame rate. Defaults to 30.
    :param width: Frame width in pixels. Defaults to auto-detect from the first frame.
    :param height: Frame height in pixels. Defaults to auto-detect from the first frame.
    :param length: Sequence length. Defaults to inferred from groundtruth or image count.
    :param force: Overwrite an existing ``sequence`` file. Defaults to False.
    :param channel: Channel subdirectory holding the frames. Defaults to ``color``; an empty
        string treats the frames as lying directly in ``sequence_dir``.

    :raises RuntimeError: If the length or frame dimensions cannot be determined.
    :returns: True if the file was written, False if it already existed and ``force`` was False.
    """
    sequence_dir = Path(sequence_dir)
    seq_file = sequence_dir / METADATA_FILE
    if seq_file.exists() and not force:
        logger.info("Sequence metadata exists, skipping: %s", seq_file)
        return False

    frame_dir = sequence_dir / channel if channel else sequence_dir
    image_files = list_image_files(frame_dir)

    if length is None:
        gt_path = sequence_dir / GROUNDTRUTH_FILE
        if gt_path.exists():
            with open(gt_path, "r") as fp:
                length = sum(1 for line in fp if line.strip())
        if not length:
            length = len(image_files)
    if not length:
        raise RuntimeError("Cannot determine sequence length for {}".format(sequence_dir))

    if (width is None or height is None) and image_files:
        size = image_size(image_files[0])
        if size is not None:
            inferred_width, inferred_height = size
            width = width or inferred_width
            height = height or inferred_height
    if width is None or height is None:
        raise RuntimeError("Cannot determine frame dimensions for {} - pass width/height explicitly".format(sequence_dir))

    channel_name = channel or "color"
    channel_pattern = "{}/{}".format(channel, DEFAULT_FRAME_MASK) if channel else DEFAULT_FRAME_MASK
    metadata = {
        "channel.default": channel_name,
        "channels.{}".format(channel_name): channel_pattern,
        "fps": str(int(fps)),
        "width": str(int(width)),
        "height": str(int(height)),
        "length": str(int(length)),
    }
    write_metadata(sequence_dir, metadata)
    logger.info("Wrote %s (%dx%d @ %dfps, %d frames)", seq_file, width, height, int(fps), length)
    return True


def import_video(video_path: str | Path, output_base: str | Path, name: str | None = None,
                 fps: float | None = None, quality: int = 2, channel: str = "color") -> Path:
    """Imports a video file as a new VOT sequence directory.

    A directory ``<output_base>/<name>`` is created (``name`` defaults to the video file name
    without its extension), every video frame is extracted into it and a ``sequence`` metadata
    file is written. The sequence name is also appended to ``<output_base>/list.txt`` so the
    workspace picks it up. A ``groundtruth.txt`` is intentionally NOT created: the sequence
    cannot be evaluated until per-frame annotations are added by hand.

    :param video_path: Path to the input video file.
    :param output_base: Directory in which the sequence folder is created; created if missing.
    :param name: Sequence (folder) name. Defaults to the video file stem.
    :param fps: Target frame rate. ``None`` keeps the source rate.
    :param quality: ffmpeg ``-q:v`` value (1 = best, 5 = worst). Defaults to 2.
    :param channel: Channel subdirectory to nest frames under. Defaults to ``color``
        (``channels.color=color/%08d.jpg``); an empty string writes a flat sequence.

    :raises FileNotFoundError: If the video file does not exist.
    :raises FileExistsError: If the target sequence directory already exists and is not empty.
    :raises RuntimeError: If no frames could be extracted from the video.
    :returns: Path to the created sequence directory.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError("Video file not found: {}".format(video_path))

    sequence_name = name or video_path.stem
    sequence_dir = Path(output_base) / sequence_name

    if sequence_dir.exists() and any(sequence_dir.iterdir()):
        raise FileExistsError(
            "Target sequence directory already exists and is not empty: {}".format(sequence_dir))
    sequence_dir.mkdir(parents=True, exist_ok=True)

    count = extract_frames(video_path, sequence_dir, fps=fps, quality=quality, channel=channel)
    if count == 0:
        raise RuntimeError("No frames were extracted from {}".format(video_path))

    # ``fps`` may be ``None`` (keep source rate, which ffmpeg does not report back here);
    # fall back to 30 for the metadata file, matching ``write_sequence_metadata``'s default.
    write_sequence_metadata(sequence_dir, fps=fps if fps else 30, force=True, channel=channel)

    SequenceList(output_base).append(sequence_name)

    gt_path = sequence_dir / GROUNDTRUTH_FILE
    logger.warning(
        "Sequence '%s' created at %s (%d frames). No groundtruth.txt was generated - "
        "add %s with one region per frame (%d lines of 'x,y,w,h'); the first line is the "
        "initialization region required before the sequence can be evaluated.",
        sequence_name, sequence_dir, count, gt_path, count)

    return sequence_dir


def remove_sequence(sequences_dir: str | Path, name: str) -> bool:
    """Removes a sequence: deletes its directory and its ``list.txt`` entry.

    :param sequences_dir: Directory containing the sequence folder and ``list.txt``.
    :param name: Sequence name to remove.

    :raises FileNotFoundError: If neither the directory nor a ``list.txt`` entry exists.
    :returns: True if anything was removed.
    """
    sequences_dir = Path(sequences_dir)
    sequence_dir = sequences_dir / name

    removed_dir = False
    if sequence_dir.is_dir():
        shutil.rmtree(sequence_dir)
        removed_dir = True
        logger.info("Removed sequence directory %s", sequence_dir)
    else:
        logger.warning("Sequence directory not found: %s", sequence_dir)

    removed_entry = SequenceList(sequences_dir).remove(name)

    if not removed_dir and not removed_entry:
        raise FileNotFoundError("No sequence '{}' found in {}".format(name, sequences_dir))

    return removed_dir or removed_entry


def discover_sequences(sequences_dir: str | Path) -> list[str]:
    """Returns the sequence names found in a directory.

    Reads ``list.txt`` when present; otherwise scans for subdirectories that contain a
    ``sequence`` metadata file.

    :param sequences_dir: Directory to inspect.

    :returns: Sequence names (list-file order, or sorted directory names as a fallback).
    """
    sequences_dir = Path(sequences_dir)
    listed = SequenceList(sequences_dir).read()
    if listed is not None:
        return listed

    if not sequences_dir.is_dir():
        return []
    return sorted(p.name for p in sequences_dir.iterdir()
                  if p.is_dir() and (p / METADATA_FILE).exists())


@dataclass
class SequenceInfo:
    """Summary of a single sequence directory, used for listings.

    :var name: Sequence name.
    :var present: Whether the sequence directory exists on disk.
    :var channels: Available channels (e.g. ``color``, ``depth``, ``ir``).
    :var length: Frame count declared in the metadata file, or ``None`` if not declared.
    :var inferred_length: Frame count inferred from on-disk contents when ``length`` is not
        declared, or ``None`` if it could not be inferred either.
    :var has_groundtruth: Whether a ``groundtruth.txt`` file is present.
    """
    name: str
    present: bool
    channels: list[str]
    length: int | None
    inferred_length: int | None
    has_groundtruth: bool


def _infer_sequence_length(seq_dir: Path, metadata: dict) -> int | None:
    """Infers a sequence's frame count from its on-disk contents.

    Used by :func:`collect_sequence_info` for sequences whose metadata file omits the ``length``
    field (common for externally imported datasets). The ``groundtruth.txt`` line count is used
    when available; otherwise the image files of each declared channel directory are counted.

    :param seq_dir: The sequence root directory.
    :param metadata: Parsed sequence metadata, used to locate the channel image directories.

    :returns: The inferred frame count, or ``None`` if it cannot be determined.
    """
    gt_file = seq_dir / GROUNDTRUTH_FILE
    if gt_file.exists():
        with open(gt_file, "r") as fp:
            count = sum(1 for line in fp if line.strip())
        if count:
            return count

    # No usable groundtruth: fall back to counting channel image files.
    directories = [seq_dir]
    for key in channel_keys(metadata):
        pattern = metadata.get("channels." + key)
        if pattern:
            directories.append(seq_dir / Path(pattern).parent)
    for directory in directories:
        count = len(list_image_files(directory))
        if count:
            return count
    return None


def collect_sequence_info(sequences_dir: str | Path) -> list[SequenceInfo]:
    """Collects per-sequence summary information for every sequence in a directory.

    The sequence names come from :func:`discover_sequences`; for each one the ``sequence``
    metadata file is read to determine the available channels and frame count. When the metadata
    does not declare a ``length``, the frame count is inferred from disk via
    :func:`_infer_sequence_length` and reported separately as ``inferred_length``.

    :param sequences_dir: Directory containing the sequences and (optionally) ``list.txt``.

    :returns: One :class:`SequenceInfo` per sequence, in listing order.
    """
    sequences_dir = Path(sequences_dir)
    infos: list[SequenceInfo] = []

    for name in discover_sequences(sequences_dir):
        seq_dir = sequences_dir / name
        present = seq_dir.is_dir()

        channels: list[str] = []
        length: int | None = None
        inferred_length: int | None = None
        if present:
            try:
                metadata = read_metadata(seq_dir, coerce=False, defaults=False)
            except FileNotFoundError:
                metadata = {}
            channels = channel_keys(metadata)
            raw_length = metadata.get("length")
            if raw_length:
                try:
                    length = int(raw_length)
                except (TypeError, ValueError):
                    length = None
            if length is None:
                inferred_length = _infer_sequence_length(seq_dir, metadata)

        has_groundtruth = present and (seq_dir / GROUNDTRUTH_FILE).exists()
        infos.append(SequenceInfo(name, present, channels, length, inferred_length, has_groundtruth))

    return infos


def _yolo_line_to_xywh(yolo_line: str, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    """Converts a single YOLO annotation line to pixel-space ``(x, y, w, h)``.

    :param yolo_line: A whitespace-delimited line ``class cx cy w h`` with normalized coordinates.
    :param image_width: Image width in pixels.
    :param image_height: Image height in pixels.

    :returns: The bounding box as ``(x_top_left, y_top_left, width, height)``, or ``None`` if the line is malformed.
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    try:
        _, center_x_norm, center_y_norm, width_norm, height_norm = (float(p) for p in parts[:5])
    except ValueError:
        return None
    width_px = width_norm * image_width
    height_px = height_norm * image_height
    return center_x_norm * image_width - width_px / 2, center_y_norm * image_height - height_px / 2, width_px, height_px


def yolo_to_vot(source_dir: str | Path, dest_dir: str | Path, fps: int = 30) -> int:
    """Converts a directory of YOLO-format frames and labels into a VOT sequence.

    The source directory must contain ``.png`` images and matching ``.txt`` YOLO label files. The
    destination receives the frames under a ``color/`` channel subdirectory, a ``groundtruth.txt``
    and a ``sequence`` metadata file, matching the canonical layout produced by
    :func:`write_sequence` / :func:`extract_frames` / :func:`import_video`.

    :param source_dir: Source directory with ``.png`` + ``.txt`` pairs.
    :param dest_dir: Destination directory for the VOT sequence; created if missing.
    :param fps: Frame rate to record in the metadata file. Defaults to 30.

    :raises FileNotFoundError: If no PNG files are found in the source directory.
    :returns: Number of frames written.
    """
    from PIL import Image

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    # Frames are nested under a "color/" channel subdirectory so the on-disk layout matches the
    # metadata written below (channels.color=color/%08d.jpg). Writing them flat in dest_dir would
    # leave the loader looking for dest_dir/color/00000001.jpg and failing to find the frames.
    channel = "color"
    frame_dir = dest_dir / channel
    frame_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() == ".png")
    if not png_files:
        raise FileNotFoundError("No PNG files in {}".format(source_dir))

    groundtruth_lines: list[str] = []
    unknown_region = str(Special(SpecialCode.UNKNOWN))
    image_width: int | None = None
    image_height: int | None = None

    for index, png_path in enumerate(png_files, 1):
        with Image.open(png_path) as image:
            frame_width, frame_height = image.size
            if image_width is None:
                image_width, image_height = frame_width, frame_height
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            image.save(frame_dir / frame_filename(index), "JPEG", quality=95)

        label_path = png_path.with_suffix(".txt")
        bbox: tuple[float, float, float, float] | None = None
        if label_path.exists():
            with open(label_path, "r") as fp:
                first_line = fp.readline().strip()
            if first_line:
                bbox = _yolo_line_to_xywh(first_line, frame_width, frame_height)
        if bbox is not None:
            x, y, w, h = bbox
            groundtruth_lines.append("{:.2f},{:.2f},{:.2f},{:.2f}".format(x, y, w, h))
        else:
            # No usable label -> unknown groundtruth, not a (scored) zero rectangle.
            groundtruth_lines.append(unknown_region)

    with open(dest_dir / GROUNDTRUTH_FILE, "w") as fp:
        fp.write("\n".join(groundtruth_lines) + "\n")

    if image_width is None or image_height is None:
        size = image_size(frame_dir / frame_filename(1))
        if size is not None:
            image_width, image_height = size

    write_sequence_metadata(dest_dir, fps=fps, width=image_width, height=image_height,
                            length=len(png_files), force=True, channel=channel)
    return len(png_files)


def write_anchor_file(output_path: str | Path, num_frames: int) -> None:
    """Writes an ``anchor.value`` file with the VOT multistart convention.

    The first frame is ``1.0`` (anchor), the last is ``-1.0`` (terminator) and middle frames are ``0.0``.
    A single-frame sequence receives a lone ``1.0``.

    :param output_path: Destination file path.
    :param num_frames: Total number of frames in the sequence.

    :raises ValueError: If ``num_frames`` is less than 1.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    output_path = Path(output_path)
    if num_frames == 1:
        lines = ["1.0"]
    else:
        lines = ["1.0"] + ["0.0"] * (num_frames - 2) + ["-1.0"]
    output_path.write_text("\n".join(lines) + "\n")


def generate_anchors_for_sequence(sequence_dir: str | Path, force: bool = False) -> bool:
    """Writes ``<sequence_dir>/anchor.value`` inferred from the sequence's frame count.

    The frame count is taken from ``groundtruth.txt`` (or the on-disk image count if groundtruth is
    missing). When both are available and disagree, a warning is logged and the groundtruth count is used.

    :param sequence_dir: The sequence root directory.
    :param force: Overwrite an existing ``anchor.value`` file. Defaults to False.

    :raises RuntimeError: If neither images nor groundtruth are present.
    :returns: True if the file was written, False if it already existed and ``force`` was False.
    """
    sequence_dir = Path(sequence_dir)
    anchor_file = sequence_dir / "anchor.value"
    if anchor_file.exists() and not force:
        logger.info("Anchor file exists, skipping: %s", anchor_file)
        return False

    gt_count = 0
    gt_file = sequence_dir / GROUNDTRUTH_FILE
    if gt_file.exists():
        with open(gt_file, "r") as fp:
            gt_count = sum(1 for line in fp if line.strip())

    image_count = len(list_image_files(sequence_dir))
    num_frames = gt_count or image_count
    if num_frames == 0:
        raise RuntimeError("Could not determine frame count for {}".format(sequence_dir))

    if gt_count and image_count and gt_count != image_count:
        logger.warning("Frame count mismatch in %s: groundtruth=%d, images=%d", sequence_dir, gt_count, image_count)

    write_anchor_file(anchor_file, num_frames)
    logger.info("Wrote anchor file %s with %d entries", anchor_file, num_frames)
    return True


def take_slice(source_dir: str | Path, begin_frame: int, end_frame: int,
               output_dir: str | Path) -> Path:
    """Copies a 1-based inclusive slice of frames into a new sequence directory.

    All channels, objects, tags and values are sliced consistently.

    :param source_dir: Source VOT sequence directory.
    :param begin_frame: First frame to include (1-based).
    :param end_frame: Last frame to include (1-based, inclusive).
    :param output_dir: Destination directory.

    :raises ValueError: If the slice bounds are out of range.
    :returns: The destination directory.
    """
    source = load_sequence(str(source_dir))
    length = len(source)
    if begin_frame < 1 or end_frame > length or begin_frame > end_frame:
        raise ValueError("Invalid slice {}..{} for sequence of length {}".format(
            begin_frame, end_frame, length))

    output_dir = _rewrite_sequence(source, output_dir, range(begin_frame - 1, end_frame))
    logger.info("Wrote slice %s (frames %d-%d)", output_dir, begin_frame, end_frame)
    return output_dir


def subsample_sequence(source_dir: str | Path, step: int, output_dir: str | Path) -> Path:
    """Keeps every ``step``-th frame of a sequence and writes the subsampled copy to a new directory.

    :param source_dir: Source VOT sequence directory.
    :param step: Subsampling step (2 = every 2nd frame, etc.).
    :param output_dir: Destination directory.

    :raises ValueError: If ``step`` is less than 1.
    :returns: The destination directory.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    source = load_sequence(str(source_dir))
    output_dir = _rewrite_sequence(source, output_dir, range(0, len(source), step))
    logger.info("Subsampled %s step=%d -> %s", source_dir, step, output_dir)
    return output_dir


def reverse_sequence(source_dir: str | Path, dest_dir: str | Path) -> Path:
    """Reverses the frame order and annotations of a VOT sequence.

    The new sequence name is appended to ``<dest_dir>/../list.txt`` so a workspace
    whose ``sequences/`` directory holds the result picks it up automatically.

    :param source_dir: Source VOT sequence directory.
    :param dest_dir: Destination directory for the reversed sequence; created if missing.

    :returns: The destination directory.
    """
    source = load_sequence(str(source_dir))
    dest_dir = _rewrite_sequence(source, dest_dir, reversed(range(len(source))), preserve_layout=True)
    SequenceList(dest_dir.parent).append(dest_dir.name)
    logger.info("Reversed sequence written to %s", dest_dir)
    return dest_dir


def delayed_init_variants(source_dir: str | Path, count: int, repetitions: int,
                          output_base: str | Path | None = None) -> list[Path]:
    """Creates ``repetitions`` delayed-init variants of a sequence, each stripping ``count`` more frames.

    Variant ``k`` strips ``k * count`` initial frames. New directories are created next to
    ``source_dir`` (or under ``output_base`` when provided) and named ``<source_name>_<k>_<count>``.
    Each successfully created variant is appended to ``<base_dir>/list.txt``.

    :param source_dir: Source VOT sequence directory.
    :param count: Number of frames to strip per repetition.
    :param repetitions: Number of variants to produce.
    :param output_base: Parent directory for the variants. Defaults to ``source_dir.parent``.

    :raises FileNotFoundError: If the source sequence does not exist.
    :raises FileExistsError: If a variant directory already exists.
    :returns: Paths of the created variant directories, in order.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError("Source sequence does not exist: {}".format(source_dir))
    base_dir = Path(output_base) if output_base else source_dir.parent

    source = load_sequence(str(source_dir))
    length = len(source)
    if repetitions * count >= length:
        raise ValueError(
            "Cannot strip {} frames over {} repetitions from a {}-frame sequence".format(
                count, repetitions, length))

    sequence_list = SequenceList(base_dir)
    created: list[Path] = []
    for repetition in range(1, repetitions + 1):
        new_folder = base_dir / "{}_{}_{}".format(source_dir.name, repetition, count)
        if new_folder.exists():
            raise FileExistsError("Variant directory already exists: {}".format(new_folder))
        _rewrite_sequence(source, new_folder, range(repetition * count, length))
        sequence_list.append(new_folder.name)
        created.append(new_folder)
    return created


def create_size_slices(sequence_dir: str | Path,
                       size_ranges: TypingSequence[tuple[float, float, str]],
                       output_dir: str | Path, target_frames: int = 100,
                       min_bbox_movements: float = 1.5, prefer_no_speedup: bool = True,
                       check_initial_size_only: bool = False) -> dict[str, dict | None]:
    """Creates one slice per ``(size_min, size_max, label)`` range, applying speed-up when necessary.

    Each slice is written to ``<output_dir>/<sequence_name>_<label>_<target_frames>f_frames<start>-<end>``
    and accompanied by a human-readable ``slice_info.txt``. Every created slice name is also appended
    to ``<output_dir>/list.txt``.

    :param sequence_dir: The source sequence root directory.
    :param size_ranges: Sequence of ``(size_min, size_max, label)`` tuples driving the slice generation.
    :param output_dir: Directory where slices are written; created if missing.
    :param target_frames: Target slice length. Defaults to 100.
    :param min_bbox_movements: Minimum movement quality threshold. Defaults to 1.5.
    :param prefer_no_speedup: Prefer windows with natural motion over sped-up ones. Defaults to True.
    :param check_initial_size_only: Use the size-at-init policy. Defaults to False.

    :returns: A mapping ``label -> {"path", "window", "verification"}`` or ``None`` for labels with no candidate.
    """
    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_list = SequenceList(output_dir)

    results: dict[str, dict | None] = {}
    for size_min, size_max, label in size_ranges:
        candidates = find_size_range_windows(sequence_dir, size_min, size_max, target_frames,
                                             min_bbox_movements, check_initial_size_only)
        if not candidates:
            results[label] = None
            continue

        best = candidates[0] if prefer_no_speedup else max(candidates, key=lambda x: x["quality_score"])
        start = best["start_frame"]
        end = best["end_frame"]
        slice_name = "{}_{}_{}f_frames{}-{}".format(sequence_dir.name, label, target_frames, start, end)
        slice_path = output_dir / slice_name
        if slice_path.exists():
            shutil.rmtree(slice_path)

        speed_up = best["speed_up_needed"]
        if speed_up == 1:
            take_slice(sequence_dir, start + 1, end + 1, slice_path)
        else:
            extended_end = min(start + target_frames * speed_up, _sequence_length(sequence_dir))
            temp_dir = output_dir / ".tmp_{}".format(slice_name)
            try:
                take_slice(sequence_dir, start + 1, extended_end, temp_dir)
                subsample_sequence(temp_dir, speed_up, slice_path)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        verification = verify_slice(slice_path, size_min, size_max, check_initial_size_only)
        sequence_list.append(slice_path.name)
        results[label] = {"path": str(slice_path), "window": best, "verification": verification}

        info_lines = [
            "Slice Metadata",
            "=" * 50,
            "",
            "Source Dataset: {}".format(sequence_dir.name),
            "Original Frame Range: {} - {}".format(start, end),
            "Speed-Up Factor: {}x".format(speed_up),
            "Final Frame Count: {}".format(verification["length"]),
            "",
            "Target Size Range: {:.1f} - {:.1f} px ({})".format(size_min, size_max, label),
            "Actual Size Range: {:.1f} - {:.1f} px".format(verification["size_min"], verification["size_max"]),
            "Average Size: {:.1f} px".format(verification["size_avg"]),
            "BBox Movements: {:.2f}x object size".format(verification["bbox_movements"]),
            "Average Velocity: {:.1f} px/s".format(verification["avg_velocity"]),
        ]
        (slice_path / "slice_info.txt").write_text("\n".join(info_lines) + "\n")

    return results


def create_speedup_sequence(source_dir: str | Path, speedup_factor: int,
                            output_dir: str | Path, start_frame: int = 0,
                            end_frame: int | None = None) -> Path:
    """Creates a single subsampled (speed-up) variant of a sequence over a 0-based inclusive frame range.

    A ``speedup_factor`` of ``1`` produces a plain slice without subsampling. The new sequence name
    is appended to ``<output_dir>/../list.txt``.

    :param source_dir: Source VOT sequence directory.
    :param speedup_factor: Subsampling factor (2 = every 2nd frame).
    :param output_dir: Destination directory; recreated if it already exists.
    :param start_frame: First frame to include (0-based). Defaults to 0.
    :param end_frame: Last frame to include (0-based). Defaults to the last frame.

    :returns: The destination directory.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if end_frame is None:
        end_frame = _sequence_length(source_dir) - 1
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if speedup_factor == 1:
        take_slice(source_dir, start_frame + 1, end_frame + 1, output_dir)
    else:
        temp_dir = output_dir.parent / ".tmp_{}".format(output_dir.name)
        try:
            take_slice(source_dir, start_frame + 1, end_frame + 1, temp_dir)
            subsample_sequence(temp_dir, speedup_factor, output_dir)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    SequenceList(output_dir.parent).append(output_dir.name)
    return output_dir


def create_speedup_experiments(source_dir: str | Path, output_dir: str | Path,
                               speedup_factors: TypingSequence[int] = (2, 3, 4, 5),
                               start_frame: int = 0, end_frame: int | None = None,
                               sequence_prefix: str | None = None) -> list[Path]:
    """Creates one speed-up variant per factor in ``speedup_factors``.

    Output sequences are named ``<prefix>_speedup_<k>x_<frame_count>f`` under ``output_dir``. Failures on
    individual factors are logged but do not abort the others.

    :param source_dir: Source VOT sequence directory.
    :param output_dir: Parent directory for the generated variants; created if missing.
    :param speedup_factors: Iterable of subsampling factors to generate. Defaults to (2, 3, 4, 5).
    :param start_frame: First frame to include (0-based). Defaults to 0.
    :param end_frame: Last frame to include (0-based). Defaults to the last frame.
    :param sequence_prefix: Prefix used in the variant directory names. Defaults to the source folder name.

    :returns: Paths of successfully generated variants.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = sequence_prefix or source_dir.name

    if end_frame is None:
        end_frame = _sequence_length(source_dir) - 1
    total_frames = end_frame - start_frame + 1

    created: list[Path] = []
    for factor in speedup_factors:
        expected = total_frames // factor
        name = "{}_speedup_{}x_{}f".format(prefix, factor, expected)
        try:
            path = create_speedup_sequence(source_dir, factor, output_dir / name, start_frame, end_frame)
            created.append(path)
        except Exception as e:
            logger.error("Failed to create %dx speedup: %s", factor, e)
    return created


def create_baseline_slice(source_dir: str | Path, output_dir: str | Path,
                          start_frame: int = 0, end_frame: int | None = None,
                          concatenate_path: str | Path | None = None) -> Path:
    """Creates a 1x baseline slice (no subsampling), optionally concatenating another sequence after it.

    Concatenation appends the frames and groundtruth of ``concatenate_path`` after the slice and
    updates the ``length`` field in the metadata file. The new sequence name is also appended to
    ``<output_dir>/../list.txt`` so a workspace whose ``sequences/`` directory holds the slice
    picks it up automatically (same convention as :func:`import_video`).

    :param source_dir: Source VOT sequence directory.
    :param output_dir: Destination directory; recreated if it already exists.
    :param start_frame: First frame to include (0-based). Defaults to 0.
    :param end_frame: Last frame to include (0-based). Defaults to the last frame.
    :param concatenate_path: Optional sequence directory whose frames are appended after the slice.

    :returns: The destination directory.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if end_frame is None:
        end_frame = _sequence_length(source_dir) - 1

    take_slice(source_dir, start_frame + 1, end_frame + 1, output_dir)

    if concatenate_path is not None:
        _concatenate_sequence(output_dir, Path(concatenate_path))

    SequenceList(output_dir.parent).append(output_dir.name)

    return output_dir


def _concatenate_sequence(target_dir: Path, appendee_dir: Path) -> None:
    """Appends another sequence's frames and groundtruth after an existing sequence directory.

    The frames of every channel shared by both sequences are copied after the target's frames; the
    ``groundtruth.txt`` of the appendee is appended and the target's ``length`` metadata is updated.

    :param target_dir: The (canonically written) target sequence directory to extend.
    :param appendee_dir: The sequence directory whose frames are appended.
    """
    target = load_sequence(str(target_dir))
    appendee = load_sequence(str(appendee_dir))
    base_length = len(target)

    for channel in target.channels():
        appendee_channel = appendee.channel(channel)
        if appendee_channel is None:
            logger.warning("Channel '%s' missing in %s; frames skipped", channel, appendee_dir)
            continue
        # Append using the target channel's own directory and filename mask so a custom
        # layout (e.g. flat frames, or "%03d.jpeg") is preserved rather than overwritten
        # with the canonical "<channel>/%08d.jpg".
        target_channel = target.channel(channel)
        layout = target_channel.disk_layout if target_channel is not None else None
        if layout is None:
            logger.warning("Channel '%s' in %s is not file-backed; frames skipped", channel, target_dir)
            continue
        channel_dir, file_pattern = Path(layout[0]), layout[1]
        channel_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(appendee_channel)):
            source_file = Path(appendee_channel.filename(index))
            if source_file.is_file():
                shutil.copy2(source_file, channel_dir / (file_pattern % (base_length + index + 1)))

    appendee_gt = appendee_dir / GROUNDTRUTH_FILE
    target_gt = target_dir / GROUNDTRUTH_FILE
    appended_count = len(appendee)
    # Keep groundtruth aligned with the now-longer frame count: one region per appended
    # frame, padded with the UNKNOWN special code (not a zero rectangle, which would be
    # scored as a real region) when the appendee carries no groundtruth of its own.
    if target_gt.exists():
        if appendee_gt.exists():
            gt_lines = [line for line in appendee_gt.read_text().splitlines() if line.strip()]
        else:
            logger.warning("No groundtruth in %s; padding %d appended frames as unknown", appendee_dir, appended_count)
            gt_lines = []
        unknown = str(Special(SpecialCode.UNKNOWN))
        gt_lines = (gt_lines + [unknown] * appended_count)[:appended_count]
        existing = target_gt.read_text()
        separator = "" if not existing or existing.endswith("\n") else "\n"
        with open(target_gt, "a") as out_fp:
            out_fp.write(separator + ("\n".join(gt_lines) + "\n" if gt_lines else ""))

    metadata = read_metadata(target_dir, coerce=False, defaults=False)
    metadata["length"] = str(base_length + appended_count)
    write_metadata(target_dir, metadata)


def reverse_xml_annotations(input_file: str | Path,
                            output_file: str | Path | None = None) -> Path:
    """Reverses the frame numbering in a CVAT-style XML annotations file.

    Tries ``<box frame=N>`` elements first, then falls back to ``<image id=N>``. Output is written to
    ``<input_stem>_reversed<ext>`` unless ``output_file`` is provided.

    :param input_file: Source XML file.
    :param output_file: Optional explicit output path.

    :raises FileNotFoundError: If the input file does not exist.
    :raises RuntimeError: If no recognised frame-bearing elements are present.
    :returns: The output file path.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError("XML file not found: {}".format(input_file))
    output_path = Path(output_file) if output_file else \
        input_file.with_name("{}_reversed{}".format(input_file.stem, input_file.suffix))

    tree = ET.parse(input_file)
    root = tree.getroot()

    boxes = root.findall(".//box[@frame]")
    if boxes:
        frame_numbers = set()
        for box in boxes:
            attr = box.get("frame")
            if attr is not None and attr.lstrip("-").isdigit():
                frame_numbers.add(int(attr))
        if not frame_numbers:
            raise RuntimeError("No valid frame attributes found on <box> elements")
        min_frame, max_frame = min(frame_numbers), max(frame_numbers)
        for box in boxes:
            attr = box.get("frame")
            if attr is None:
                continue
            try:
                original_frame = int(attr)
            except ValueError:
                continue
            box.set("frame", str(max_frame - (original_frame - min_frame)))
    else:
        images = root.findall(".//image")
        if not images:
            raise RuntimeError("No <box frame=...> or <image> elements found")
        total = len(images)
        frame_entries: list[tuple[int, ET.Element]] = []
        for image in images:
            identifier = image.get("id")
            if identifier is None or not identifier.lstrip("-").isdigit():
                continue
            frame_entries.append((int(identifier), image))
        frame_entries.sort(key=lambda t: t[0])
        for new_index_zero_based, (original, element) in enumerate(frame_entries):
            new_index = total - 1 - new_index_zero_based
            element.set("id", str(new_index))
            name = element.get("name")
            if name and str(original).zfill(8) in name:
                element.set("name", name.replace(str(original).zfill(8), str(new_index).zfill(8)))

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
