"""Reading and writing sequences in the VOT-compatible directory format.

This module loads :class:`Sequence` objects from disk and writes them back. The on-disk
layout primitives (the ``sequence`` metadata file, ``list.txt`` index, frame mask) live in
:mod:`vot.dataset.layout`; this module builds the :class:`Sequence`/:class:`Channel`
objects on top of them.
"""

import os
import shutil
import logging
from collections.abc import Mapping
from pathlib import Path

import cv2

from vot.dataset import Channel, DatasetException, Sequence, BasedSequence, PatternFileListChannel, SequenceData, pad_to_length
from vot.dataset.layout import (DEFAULT_FRAME_MASK, GROUNDTRUTH_FILE, METADATA_FILE,
                                SequenceList, detect_frame_pattern, read_metadata,
                                write_metadata)
from vot.region.io import write_trajectory, read_trajectory
from vot.region import Special, SpecialCode
from vot.utilities import Progress, localize_path, read_properties

logger = logging.getLogger("vot")


def _load_channel(source: str | Path, length: int | None = None) -> PatternFileListChannel:
    """Loads a channel from the given source.

    :param source: The channel source: either a printf frame pattern or a bare directory
        (in which case the default frame mask is appended).
    :param length: The expected channel length. When ``None`` the channel scans the disk.

    :returns: The loaded channel."""
    source = Path(source)
    if source.suffix == "":
        source = source / DEFAULT_FRAME_MASK
    return PatternFileListChannel(str(source), end=length, check_files=length is None)


def _discover_channels(root: Path) -> dict[str, str]:
    """Discovers the frame channels of a sequence that has no channel metadata.

    Each standard channel name (``color``, ``depth``, ``ir``) is registered when a
    subdirectory of that name holds numbered image files. When none of those
    subdirectories exist the frames are taken to lie directly in the sequence root and
    registered as the ``color`` channel. Frame patterns (mask and extension) are detected
    from the files on disk rather than assumed.

    :param root: The sequence root directory.

    :returns: A mapping of channel name to frame pattern relative to ``root``."""
    channels: dict[str, str] = {}
    for name in ("color", "depth", "ir"):
        channel_dir = root / name
        if not channel_dir.is_dir():
            continue
        pattern = detect_frame_pattern(channel_dir)
        if pattern is not None:
            channels[name] = os.path.join(name, pattern)

    if not channels:
        pattern = detect_frame_pattern(root)
        if pattern is not None:
            channels["color"] = pattern

    return channels


def _read_data(metadata: dict) -> SequenceData:
    """Reads the channels, objects, tags and values for a sequence from its metadata.

    :param metadata: The sequence metadata, including the ``root`` directory.

    :returns: The loaded sequence data."""

    channels: dict = {}
    tags: dict = {}
    values: dict = {}
    # ``length`` is optional in the sequence metadata: ``read_sequence_legacy``
    # leaves it ``None`` on purpose and older sequence files may omit it. When
    # absent it is inferred below from the channel file count.
    length = metadata.get("length", None)

    root_value = metadata.get("root")
    if root_value is None:
        raise DatasetException("Sequence metadata is missing the 'root' directory")
    root = Path(root_value)

    for c in ["color", "depth", "ir"]:
        channel_path = metadata.get("channels.%s" % c, None)
        if channel_path is not None:
            channels[c] = _load_channel(root / localize_path(channel_path), length)

    # Load default channel if no explicit channel data available
    if len(channels) == 0:
        channels["color"] = _load_channel(root / "color" / DEFAULT_FRAME_MASK, length=length)
    else:
        metadata["channel.default"] = next(iter(channels.keys()))

    if metadata.get("width", None) is None or metadata.get("height", None) is None:
        metadata["width"], metadata["height"] = next(iter(channels.values())).size

    lengths = [len(t) for t in channels.values()]
    if not all(x == lengths[0] for x in lengths):
        raise DatasetException(
            "Sequence '{}' has channels of mismatched lengths: {}".format(
                root, {name: len(c) for name, c in channels.items()}))
    length = lengths[0]
    if length == 0:
        raise DatasetException(
            "Sequence '{}' resolved to zero frames; check that the channel images exist".format(root))

    objectsfiles = sorted(root.glob("groundtruth_*.txt"))
    objects: dict = {}
    if len(objectsfiles) > 0:
        for objectfile in objectsfiles:
            groundtruth = read_trajectory(str(objectfile))
            groundtruth = pad_to_length(groundtruth, length, Special(SpecialCode.UNKNOWN))
            objectid = objectfile.stem[len("groundtruth_"):]
            objects[objectid] = groundtruth
    else:
        groundtruth_file = root / metadata.get("groundtruth", GROUNDTRUTH_FILE)
        groundtruth = read_trajectory(str(groundtruth_file))
        groundtruth = pad_to_length(groundtruth, length, Special(SpecialCode.UNKNOWN))
        objects["object"] = groundtruth

    metadata["length"] = length

    tagfiles = sorted(root.glob("*.tag")) + sorted(root.glob("*.label"))

    for tagfile in tagfiles:
        with open(tagfile, 'r') as filehandle:
            tagname = tagfile.stem
            tag = [line.strip() == "1" for line in filehandle.readlines()]
            tags[tagname] = pad_to_length(tag, length, False)

    valuefiles = sorted(root.glob("*.value"))

    for valuefile in valuefiles:
        with open(valuefile, 'r') as filehandle:
            valuename = valuefile.stem
            value = [float(line.strip()) for line in filehandle.readlines()]
            values[valuename] = pad_to_length(value, length, 0.0)

    for name, tag in tags.items():
        if not len(tag) == length:
            tag_tmp = length * [False]
            tag_tmp[:len(tag)] = tag
            tag = tag_tmp

    for name, value in values.items():
        if not len(value) == length:
            raise DatasetException("Length mismatch for value %s" % name)

    return SequenceData(channels, objects, tags, values, length)


def read_sequence(path: str | Path) -> Sequence | None:
    """Reads a sequence from the given path.

    :param path: The path to read the sequence from.

    :returns: The sequence, or ``None`` if the path does not contain a recognised sequence."""
    path = Path(path)
    if not (path / METADATA_FILE).is_file():
        return None

    return BasedSequence(path.name, _read_data, read_metadata(path))


def read_sequence_legacy(path: str | Path) -> Sequence | None:
    """Reads a legacy sequence (one that has only a ``groundtruth.txt``, no metadata file).

    :param path: The path to read the sequence from.

    :returns: The sequence, or ``None`` if no groundtruth is present."""
    path = Path(path)
    if not (path / GROUNDTRUTH_FILE).is_file():
        return None

    # The legacy reader leaves ``length`` to be inferred by ``_read_data`` from the
    # channel file count.
    metadata: dict = dict(fps=30, format="default")
    metadata["channel.default"] = "color"
    metadata["root"] = str(path)
    metadata["length"] = None

    # Legacy sequences carry no channel metadata: discover the channel directories
    # (color/depth/ir) and frame masks from disk, falling back to frames stored
    # directly in the sequence root.
    for name, pattern in _discover_channels(path).items():
        metadata["channels.%s" % name] = pattern

    return BasedSequence(path.name, _read_data, metadata=metadata)


def list_sequences(path: str | Path) -> list[str] | None:
    """Indexes the sequences in the given path. Only works if there is a ``list.txt`` file
    in the given path or the path is itself a list file.

    :param path: The path to index sequences in.

    :returns: List of sequence names, or ``None`` if no list could be found."""
    path = Path(path)

    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    if path.is_dir():
        return SequenceList(path).read()

    return None


def download_dataset_meta(url: str, path: str) -> None:
    """Downloads the metadata of a dataset from a given URL and stores it in the given
    path.

    :param url: The URL to download the metadata from.
    :param path: The path to store the metadata in.
    """
    from vot.utilities.net import download_uncompress, download_json, get_base_url, join_url, NetworkException
    from vot.utilities import format_size

    base_path = Path(path)
    meta = download_json(url)

    total_size = 0
    for sequence in meta["sequences"]:
        total_size += sequence["annotations"]["uncompressed"]
        for channel in sequence["channels"].values():
            total_size += channel["uncompressed"]

    logger.info('Downloading sequence dataset "%s" with %s sequences (total %s).',
                meta["name"], len(meta["sequences"]), format_size(total_size))

    base_url = get_base_url(url) + "/"

    failed = []

    with Progress("Downloading", len(meta["sequences"])) as progress:
        for sequence in meta["sequences"]:
            sequence_directory = base_path / sequence["name"]
            sequence_directory.mkdir(parents=True, exist_ok=True)

            metadata_path = sequence_directory / METADATA_FILE
            if metadata_path.is_file():
                refdata = read_properties(str(metadata_path))
                if "uid" in refdata and refdata["uid"] == sequence["annotations"]["uid"]:
                    logger.info('Sequence "%s" already downloaded.', sequence["name"])
                    progress.relative(1)
                    continue

            data = {'name': sequence["name"], 'fps': sequence["fps"], 'format': 'default'}

            if "metadata" in sequence and isinstance(sequence["metadata"], dict):
                # Only update metadata fields that are not already included in the metadata dictionary
                for key, value in sequence["metadata"].items():
                    if not key.startswith("channels.") and not key in data:
                        data[key] = value

            annotations_url = join_url(base_url, sequence["annotations"]["url"])

            data["uid"] = sequence["annotations"]["uid"]

            try:
                download_uncompress(annotations_url, str(sequence_directory))
            except NetworkException as e:
                logger.exception(e)
                failed.append(sequence["name"])
                continue
            except IOError as e:
                logger.exception(e)
                failed.append(sequence["name"])
                continue

            failure = False

            for cname, channel in sequence["channels"].items():
                channel_directory = sequence_directory / cname
                channel_directory.mkdir(parents=True, exist_ok=True)

                channel_urls = []
                for cbase in (base_url, url + "/"):
                    candidate = join_url(cbase, channel["url"])
                    if candidate not in channel_urls:
                        channel_urls.append(candidate)

                last_error: Exception | None = None
                for channel_url in channel_urls:
                    try:
                        download_uncompress(channel_url, str(channel_directory))
                        last_error = None
                        break
                    except (NetworkException, IOError) as e:
                        last_error = e

                if last_error is not None:
                    logger.exception(last_error)
                    failed.append(sequence["name"])
                    failure = True
                    break

                if "pattern" in channel:
                    data["channels." + cname] = cname + os.sep + channel["pattern"]
                else:
                    data["channels." + cname] = cname + os.sep

            # Only register the sequence as available once every channel has been
            # downloaded; a partial sequence must not get a valid 'sequence' file,
            # otherwise it is treated as complete and never re-downloaded.
            if not failure:
                write_metadata(sequence_directory, data)

            progress.relative(1)

    if len(failed) > 0:
        logger.error('Failed to download %d sequences.', len(failed))
        logger.error('Failed sequences: %s', ', '.join(failed))
    else:
        logger.info('Successfully downloaded all sequences.')
        SequenceList(base_path).write(sequence["name"] for sequence in meta["sequences"])


def _copy_frame_file(channel: Channel | None, index: int, destination: Path) -> bool:
    """Copies a file-backed JPEG frame byte-for-byte to ``destination``.

    Used by :func:`write_sequence` to avoid re-encoding (and quality loss) when the source
    frame is already a JPEG file on disk.

    :param channel: The source channel, or ``None``.
    :param index: The frame index in the channel.
    :param destination: The destination file path.

    :returns: True if the frame file was copied, False if it must be re-encoded instead."""
    if channel is None:
        return False
    try:
        source_file = channel.filename(index)
    except Exception:
        return False
    if not source_file:
        return False
    source_path = Path(source_file)
    if source_path.suffix.lower() not in (".jpg", ".jpeg") or not source_path.is_file():
        return False
    shutil.copy2(source_path, destination)
    return True


def _source_channel_layout(source_channel: Channel | None, channel_name: str,
                           root: str | None) -> tuple[str, str]:
    """Returns the ``(relative_directory, filename_pattern)`` of a source channel on disk.

    Used by :func:`write_sequence` to reproduce a channel's exact layout (subdirectory and printf
    filename mask, e.g. ``color/%05dv.jpg``) instead of normalizing it to ``<channel>/%08d.jpg``.
    Falls back to the channel name and the default mask when the channel is not file-backed or its
    base lies outside the sequence root. Proxy channels report their wrapped channel's layout via
    :attr:`Channel.disk_layout`.

    :param source_channel: The source channel object.
    :param channel_name: The channel name (used as the fallback subdirectory).
    :param root: The source sequence root directory, or ``None`` if unknown.

    :returns: ``(relative_directory, filename_pattern)``; ``relative_directory`` is empty for a
        flat (root-level) channel."""
    layout = source_channel.disk_layout if source_channel is not None else None
    if layout is not None and root:
        base, pattern = layout
        try:
            rel = os.path.relpath(base, root)
        except ValueError:
            return channel_name, DEFAULT_FRAME_MASK
        return ("" if rel in (".", "") else rel), pattern
    return channel_name, DEFAULT_FRAME_MASK


def write_sequence(directory: str | Path, sequence: Sequence, preserve_layout: bool = False) -> None:
    """Writes a sequence to a directory in the VOT format.

    Each channel is written as a directory of images; the groundtruth, tags, values and the
    ``sequence`` metadata file are written into the root. File-backed JPEG frames are copied
    byte-for-byte (no re-encoding); other frames are encoded with OpenCV.

    :param directory: The directory to write the sequence to.
    :param sequence: The sequence to write.
    :param preserve_layout: When True, reproduce each source channel's exact subdirectory and
        filename pattern (e.g. ``color/%05dv.jpg``) instead of normalizing to
        ``<channel>/%08d.jpg``. Defaults to False.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    channels = sequence.channels()

    # Preserve passthrough metadata (name, uid, custom fields); the structural fields
    # below (channels, length, ...) are recomputed for the written sequence.
    metadata: dict = {}
    source_metadata = sequence.metadata()
    if isinstance(source_metadata, Mapping):
        metadata.update({k: v for k, v in source_metadata.items() if k != "root"})
    metadata["channel.default"] = sequence.metadata("channel.default", "color")
    metadata["fps"] = sequence.metadata("fps", 30)
    metadata["length"] = len(sequence)

    root = source_metadata.get("root") if isinstance(source_metadata, Mapping) else None

    for channel in channels:
        source_channel = sequence.channel(channel)
        if preserve_layout:
            rel_dir, file_pattern = _source_channel_layout(source_channel, channel, root)
        else:
            rel_dir, file_pattern = channel, DEFAULT_FRAME_MASK

        cdir = directory / rel_dir if rel_dir else directory
        cdir.mkdir(parents=True, exist_ok=True)

        metadata["channels.%s" % channel] = os.path.join(rel_dir, file_pattern) if rel_dir else file_pattern

        for i in range(len(sequence)):
            destination = cdir / (file_pattern % (i + 1))
            if _copy_frame_file(source_channel, i, destination):
                continue
            image = sequence.frame(i).image(channel)
            if image is None:
                raise DatasetException(
                    "Cannot write sequence '{}': channel '{}' is missing the image for frame {}".format(
                        sequence.name, channel, i))
            cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    for tag in sequence.tags():
        data = "\n".join(["1" if tag in sequence.tags(i) else "0" for i in range(len(sequence))])
        with open(directory / f"{tag}.tag", "w", encoding="utf-8") as fp:
            fp.write(data)

    for value in sequence.values():
        data = "\n".join([str(sequence.values(i).get(value, "")) for i in range(len(sequence))])
        with open(directory / f"{value}.value", "w", encoding="utf-8") as fp:
            fp.write(data)

    # Write groundtruth. ``Frame.groundtruth()`` / ``Frame.object(oid)`` may return
    # ``None`` when the object is not visible in that frame; the on-disk trajectory
    # format encodes "unknown" via ``Special(SpecialCode.UNKNOWN)``, matching how
    # loaders pad missing groundtruth on read.
    unknown_region = Special(SpecialCode.UNKNOWN)
    if len(sequence.objects()) == 1:
        groundtruth_data = [f.groundtruth() or unknown_region for f in sequence]
        write_trajectory(str(directory / GROUNDTRUTH_FILE), groundtruth_data)
    else:
        for oid in sequence.objects():
            per_object_data = [f.object(oid) or unknown_region for f in sequence]
            write_trajectory(str(directory / f"groundtruth_{oid}.txt"), per_object_data)

    write_metadata(directory, metadata)
