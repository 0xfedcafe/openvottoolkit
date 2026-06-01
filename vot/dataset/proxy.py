"""Proxy sequence classes that allow to modify the behaviour of a sequence without
changing the underlying data."""
from typing import Mapping, Set, overload

import numpy.typing as npt

from vot.region import Region

from vot.dataset import Channel, Sequence, Frame


class ProxySequence(Sequence):
    """A proxy sequence base that forwards requests to undelying source sequence.

    Meant as a base class.
    """

    def __init__(self, source: Sequence, name: str | None = None) -> None:
        """Creates a proxy sequence.

        :param source: Source sequence object
        :param name: Override the proxy sequence name. Defaults to the source name.
        """
        if name is None:
            name = source.name
        super().__init__(name)
        self._source = source

    def __len__(self) -> int:
        """Returns the length of the sequence."""
        return len(self._source)

    def frame(self, index: int) -> Frame:
        """Returns a frame object for the given index."""
        return Frame(self, index)

    def metadata(self, name: str | None = None, default: object | None = None) -> object:
        """Returns a metadata value for the given name. Forwards to the source sequence."""
        return self._source.metadata(name, default)

    def channel(self, channel: str | None = None) -> Channel | None:
        """Returns a channel object for the given name. Forwards to the source sequence."""
        return self._source.channel(channel)

    def channels(self) -> list[str]:
        """Returns a list of channel names. Forwards to the source sequence."""
        return self._source.channels()

    def objects(self, index: int | None = None) -> list[str]:
        """Returns a list of object ids. Forwards to the source sequence."""
        return self._source.objects(index)

    @overload
    def object(self, oid: str, index: None = None) -> list[Region] | None: ...
    @overload
    def object(self, oid: str, index: int) -> Region | None: ...
    def object(self, oid: str, index: int | None = None) -> Region | list[Region] | None:
        """Returns an object for the given id. Forwards to the source sequence."""
        return self._source.object(oid, index)

    @overload
    def groundtruth(self, index: None = None) -> list[Region] | None: ...
    @overload
    def groundtruth(self, index: int) -> Region | None: ...
    def groundtruth(self, index: int | None = None) -> Region | list[Region] | None:
        """Returns the groundtruth for the given index, or the full list when no
        index is supplied. Forwards to the source sequence."""
        return self._source.groundtruth(index)

    def tags(self, index: int | None = None) -> list[str]:
        """Returns a list of tags for the given index. Forwards to the source sequence."""
        return self._source.tags(index)

    @overload
    def values(self, index: None = None) -> list[str]: ...
    @overload
    def values(self, index: int) -> Mapping[str, float]: ...
    def values(self, index: int | None = None) -> list[str] | Mapping[str, float]:
        """Returns the values for the given index, or the list of value names when no
        index is supplied. Forwards to the source sequence."""
        return self._source.values(index)

    @property
    def size(self) -> tuple[int, int]:
        """Returns the size of the sequence. Forwards the request to the source
        sequence.

        :returns: Size of the sequence.
        :rtype: tuple[int, int]"""
        return self._source.size

    @property
    def width(self) -> int:
        """Width of the underlying source sequence."""
        return self._source.width

    @property
    def height(self) -> int:
        """Height of the underlying source sequence."""
        return self._source.height


class FrameMapChannel(Channel):
    """A proxy channel that maps frames to a different order."""

    def __init__(self, source: Channel, frame_map: list[int]) -> None:
        """Creates a frame mapping proxy channel.

        :param source: Source channel object
        :param frame_map: A list of frame indices in the source channel that will form the proxy. The list is filtered so that all indices that are out of bounds are removed.
        """
        super().__init__()
        self._source = source
        self._map = frame_map

    def __len__(self) -> int:
        """Returns the length of the channel."""
        return len(self._map)

    def frame(self, index: int) -> npt.NDArray | None:
        """Returns a frame object for the given index."""
        return self._source.frame(self._map[index])

    def filename(self, index: int) -> str:
        """Returns the filename of the frame for the given index, mapped through the frame map."""
        return self._source.filename(self._map[index])

    @property
    def size(self) -> tuple:
        """Returns the size of the channel."""
        return self._source.size

    @property
    def disk_layout(self) -> tuple[str, str] | None:
        """Delegates to the wrapped source channel so the original layout is preserved."""
        return self._source.disk_layout


class FrameMapSequence(ProxySequence):
    """A proxy sequence that maps frames from a source sequence in another order."""

    def __init__(self, source: Sequence, frame_map: list[int]) -> None:
        """Creates a frame mapping proxy sequence.

        :param source: Source sequence object
        :param frame_map: A list of frame indices in the source sequence that will form the proxy. The list is filtered so that all indices that are out of bounds are removed.
        """
        super().__init__(source)
        self._map = [i for i in frame_map if 0 <= i < len(source)]

    def channel(self, channel: str | None = None) -> Channel | None:
        """Returns a channel object for the given channel name."""
        sourcechannel = self._source.channel(channel)

        if sourcechannel is None:
            return None

        return FrameMapChannel(sourcechannel, self._map)

    def channels(self) -> list[str]:
        """Returns a list of channel names."""
        return self._source.channels()

    def frame(self, index: int) -> Frame:
        """Returns a frame object for the given index. Forwards the request to the
        source sequence with the mapped index."""
        return self._source.frame(self._map[index])

    @overload
    def groundtruth(self, index: None = None) -> list[Region] | None: ...
    @overload
    def groundtruth(self, index: int) -> Region | None: ...
    def groundtruth(self, index: int | None = None) -> Region | list[Region] | None:
        """Returns the groundtruth, mapped through the frame map."""
        if index is None:
            mapped: list[Region | None] = [None] * len(self)
            for i, m in enumerate(self._map):
                mapped[i] = self._source.groundtruth(m)
            return mapped  # type: ignore[return-value]
        return self._source.groundtruth(self._map[index])

    @overload
    def object(self, oid: str, index: None = None) -> list[Region] | None: ...
    @overload
    def object(self, oid: str, index: int) -> Region | None: ...
    def object(self, oid: str, index: int | None = None) -> Region | list[Region] | None:
        """Returns the per-frame object regions, mapped through the frame map."""
        if index is None:
            mapped: list[Region | None] = [None] * len(self)
            for i, m in enumerate(self._map):
                mapped[i] = self._source.object(oid, m)
            return mapped  # type: ignore[return-value]
        return super().object(oid, self._map[index])

    def tags(self, index: int | None = None) -> list[str]:
        """Returns a list of tags for the given index. Forwards to the source sequence
        with the mapped index."""
        if index is None:
            # TODO: this is probably not correct
            return self._source.tags()
        return self._source.tags(self._map[index])

    @overload
    def values(self, index: None = None) -> list[str]: ...
    @overload
    def values(self, index: int) -> Mapping[str, float]: ...
    def values(self, index: int | None = None) -> list[str] | Mapping[str, float]:
        """Returns the values for the given index, or the list of value names when no
        index is supplied. Forwards to the source sequence with the mapped index."""
        if index is None:
            # TODO: this is probably not correct
            return self._source.values()
        return self._source.values(self._map[index])

    def __len__(self) -> int:
        """Returns the length of the sequence. The length is the same as the length of
        the frame map."""
        return len(self._map)

class ChannelFilterSequence(ProxySequence):
    """A proxy sequence that only makes specific channels visible."""

    def __init__(self, source: Sequence, channels: Set[str]) -> None:
        """Creates a channel filter proxy sequence.

        :param source: Source sequence object
        :param channels: A set of channel names that will be visible in the proxy sequence. The set is filtered so that all channel names that are not in the source sequence are removed.
        """
        super().__init__(source)
        self._filter: list[str] = [i for i in channels if i in source.channels()]

    def channel(self, channel: str | None = None) -> Channel | None:
        """Returns a channel object for the given channel name. If the channel is not in
        the filter, None is returned."""
        if channel not in self._filter:
            return None
        return self._source.channel(channel)

    def channels(self) -> list[str]:
        """Returns the list of visible channel names."""
        return list(self._filter)


class ObjectFilterSequence(ProxySequence):
    """A proxy sequence that only makes specific object visible."""

    def __init__(self, source: Sequence, id: str, trim: bool = False) -> None:
        """Creates an object filter proxy sequence.

        :param source: Source sequence object
        :param id: ID of the object that will be visible in the proxy sequence.
        :param trim: If true, the sequence will be trimmed to the first and last frame where the object is visible.
        """
        super().__init__(source, "%s_%s" % (source.name, id))
        self._id = id
        # TODO: implement trim
        self._trim = trim

    def objects(self, index: int | None = None) -> list[str]:
        """Returns the list of visible object ids — only ``self._id`` is exposed.

        The base contract is ``list[str]``; the previous implementation here
        returned ``{self._id: objects[id]}`` (using the builtin ``id`` as a list
        index), which is both a Liskov violation and a runtime ``TypeError``.
        """
        del index  # ``self._id`` is visible across all frames.
        return [self._id]

    @overload
    def object(self, oid: str, index: None = None) -> list[Region] | None: ...
    @overload
    def object(self, oid: str, index: int) -> Region | None: ...
    def object(self, oid: str, index: int | None = None) -> Region | list[Region] | None:
        """Returns an object for the given id (only ``self._id`` is visible)."""
        if oid != self._id:
            return None
        return self._source.object(oid, index)

    @overload
    def groundtruth(self, index: None = None) -> list[Region] | None: ...
    @overload
    def groundtruth(self, index: int) -> Region | None: ...
    def groundtruth(self, index: int | None = None) -> Region | list[Region] | None:
        """Groundtruth for the visible object."""
        return self._source.object(self._id, index)


class ObjectsHideFilterSequence(ProxySequence):
    """A proxy sequence that virtually removes specified objects from the sequence.

    Note that the object is not removed from the sequence, but only hidden when listing
    them.
    """

    def __init__(self, source: Sequence, ids: Set[str]) -> None:
        """Creates an object hide filter proxy sequence.

        :param source: Source sequence object
        :param ids: IDs of the objects that will be hidden in the proxy sequence.
        """
        super().__init__(source)
        self._ids: Set[str] = ids

    def objects(self, index: int | None = None) -> list[str]:
        """Returns the list of visible object ids — the hidden ids are filtered out."""
        return [oid for oid in self._source.objects(index) if oid not in self._ids]


def IgnoreSpecialObjects(sequence: Sequence) -> Sequence:
    """Creates a proxy sequence that ignores special objects. Special objects are
    denoted by a leading underscore in the object name. Usually, those objects are
    used for storing additional information about the sequence.

    :param sequence: Source sequence object.

    :returns: Proxy sequence object (or the original sequence if no special objects exist)."""

    def is_special(id: str) -> bool:
        """Checks if the object id is special (starts with underscore)."""
        return id.startswith("_")

    ids: Set[str] = {oid for oid in sequence.objects() if is_special(oid)}

    if len(ids) == 0:
        return sequence

    return ObjectsHideFilterSequence(sequence, ids)