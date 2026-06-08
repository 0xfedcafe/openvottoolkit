"""Dataset adapter for the TrackingNet dataset.

Note that the dataset is organized a different way than the VOT datasets, annotated
frames are stored in a separate directory. The dataset also contains train and test
splits. The loader assumes that only one of the splits is used at a time and that the
path is given to this part of the dataset.
"""

from __future__ import annotations

import os
import glob
from typing import Any, TYPE_CHECKING

import six

from vot import get_logger
from vot.region.io import read_trajectory

if TYPE_CHECKING:
    from vot.dataset import PatternFileListChannel, Sequence, SequenceData

logger = get_logger()


def load_channel(source: str) -> "PatternFileListChannel":
    """Load channel from the given source.

    :param source: Path to the source. If the source is a directory, it is assumed to be a pattern file list. If the source is a file, it is assumed to be a video file.

    :returns: Channel object."""
    from vot.dataset import PatternFileListChannel

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%d.jpg')
    return PatternFileListChannel(source)


def _read_data(metadata: dict[str, Any]) -> "SequenceData":
    """Internal function for reading data from the given metadata for a TrackingNet
    sequence.

    :param metadata: Metadata dictionary.

    :returns: Sequence data object."""
    from vot.dataset import SequenceData, pad_single_frame_groundtruth

    tags: dict[str, Any] = {}
    values: dict[str, Any] = {}

    name = metadata["name"]
    root = metadata["root"]

    channels = {"color": load_channel(os.path.join(root, 'frames', name))}
    metadata["channel.default"] = "color"
    metadata["width"], metadata["height"] = six.next(six.itervalues(channels)).size

    groundtruth = read_trajectory(root)

    groundtruth = pad_single_frame_groundtruth(groundtruth, len(channels["color"]))

    metadata["length"] = len(groundtruth)

    objects = {"object": groundtruth}

    return SequenceData(channels, objects, tags, values, len(groundtruth))


def read_sequence(path: str) -> Sequence | None:
    """Read sequence from the given path. Different to VOT datasets, the sequence is not
    a directory, but a file. From the file name the sequence name is extracted and the
    path to image frames is inferred based on standard TrackingNet directory structure.

    :param path: Path to the sequence groundtruth.

    :returns: Loaded sequence, or ``None`` if ``path`` is not a TrackingNet groundtruth file."""
    from vot.dataset import BasedSequence

    if not os.path.isfile(path):
        return None

    name, ext = os.path.splitext(os.path.basename(path))

    if ext != '.txt':
        return None

    root = os.path.dirname(os.path.dirname(os.path.dirname(path)))

    if not os.path.isfile(path) and os.path.isdir(os.path.join(root, 'frames', name)):
        return None

    metadata: dict[str, Any] = dict(fps=30)
    metadata["channel.default"] = "color"
    metadata["name"] = name
    metadata["root"] = root

    return BasedSequence(name, _read_data, metadata)


def list_sequences(path: str) -> list[str] | None:
    """List sequences in the given path. The path is expected to be the root of the
    TrackingNet dataset split.

    :param path: Path to the dataset root.

    :returns: List of sequence groundtruth files, or ``None`` if the expected layout is missing."""
    for dirname in ["anno", "frames"]:
        if not os.path.isdir(os.path.join(path, dirname)):
            return None

    return list(glob.glob(os.path.join(path, "anno", "*.txt")))




