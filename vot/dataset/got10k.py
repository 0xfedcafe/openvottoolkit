"""GOT-10k dataset adapter module.

The format of GOT-10k dataset is very similar to a subset of VOT, so there is a lot of
code duplication.
"""

import os
import glob
import configparser
from typing import Any

import six

from vot import get_logger
from vot.dataset import DatasetException, BasedSequence, \
     PatternFileListChannel, SequenceData, Sequence, pad_to_length, pad_single_frame_groundtruth
from vot.region.io import read_trajectory

logger = get_logger()


def load_channel(source: str) -> PatternFileListChannel:
    """Load channel from the given source.

    :param source: Path to the source. If the source is a directory, it is assumed to be a pattern file list. If the source is a file, it is assumed to be a video file.

    :returns: Channel object."""
    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    return PatternFileListChannel(source)


def _read_data(metadata: dict[str, Any]) -> SequenceData:
    """Read data from the given metadata.

    :param metadata: Metadata dictionary.
    """
    channels: dict[str, PatternFileListChannel] = {}
    tags: dict[str, Any] = {}
    values: dict[str, Any] = {}

    base = metadata["root"]

    channels["color"] = load_channel(os.path.join(base, "%08d.jpg"))
    metadata["channel.default"] = "color"
    metadata["width"], metadata["height"] = six.next(six.itervalues(channels)).size

    groundtruth_file = os.path.join(base, metadata.get("groundtruth", "groundtruth.txt"))
    groundtruth = read_trajectory(groundtruth_file)

    groundtruth = pad_single_frame_groundtruth(groundtruth, len(channels["color"]))

    metadata["length"] = len(groundtruth)

    tagfiles = glob.glob(os.path.join(base, '*.label'))

    for tagfile in tagfiles:
        with open(tagfile, 'r') as filehandle:
            tagname = os.path.splitext(os.path.basename(tagfile))[0]
            tag = [line.strip() == "1" for line in filehandle.readlines()]
            tags[tagname] = pad_to_length(tag, len(groundtruth), False)

    valuefiles = glob.glob(os.path.join(base, '*.value'))

    for valuefile in valuefiles:
        with open(valuefile, 'r') as filehandle:
            valuename = os.path.splitext(os.path.basename(valuefile))[0]
            value = [float(line.strip()) for line in filehandle.readlines()]
            values[valuename] = pad_to_length(value, len(groundtruth), 0.0)

    for name, channel in channels.items():
        if len(channel) != len(groundtruth):
            raise DatasetException("Length mismatch for channel %s" % name)

    for name, tag in tags.items():
        if len(tag) != len(groundtruth):
            tag_tmp = len(groundtruth) * [False]
            tag_tmp[:len(tag)] = tag
            tag = tag_tmp

    for name, value in values.items():
        if len(value) != len(groundtruth):
            raise DatasetException("Length mismatch for value %s" % name)

    objects = {"object": groundtruth}

    return SequenceData(channels, objects, tags, values, len(groundtruth))


from vot.dataset import sequence_reader


@sequence_reader.register("GOT-10k")
def read_sequence(path: str) -> Sequence | None:
    """Read GOT-10k sequence from the given path.

    :param path: Path to the sequence.

    :returns: Loaded sequence, or ``None`` if ``path`` does not look like a GOT-10k sequence."""

    if not (os.path.isfile(os.path.join(path, 'groundtruth.txt')) and os.path.isfile(os.path.join(path, 'meta_info.ini'))):
        return None

    # Heterogeneous metadata values (ints, strings, raw config sections) — typed
    # as ``dict[str, Any]`` so pyright doesn't pick a narrow value type from the
    # first ``dict(...)`` literal and then reject later assignments.
    metadata: dict[str, Any] = dict(fps=30, format="default")

    if os.path.isfile(os.path.join(path, 'meta_info.ini')):
        config = configparser.ConfigParser()
        config.read(os.path.join(path, 'meta_info.ini'))
        metadata.update(config["METAINFO"])
        # ``anno_fps`` is stored as e.g. ``"30fps"`` — strip the trailing unit and parse.
        metadata["fps"] = int(str(metadata["anno_fps"])[:-3])

    metadata["root"] = path
    metadata["name"] = os.path.basename(path)
    metadata["channel.default"] = "color"

    return BasedSequence(metadata["name"], _read_data, metadata)


