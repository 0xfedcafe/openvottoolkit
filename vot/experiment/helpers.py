"""Helper classes for experiments."""


from vot.dataset import Sequence
from vot.region import RegionException, is_special


def _objectstart(sequence: Sequence, id: str) -> int:
    """Returns the first frame where the object appears in the sequence.

    Raises :class:`RegionException` if the object is unknown or never visible
    (rather than silently returning ``-1`` via ``list.index`` ValueError)."""
    trajectory = sequence.object(id)
    if trajectory is None:
        raise RegionException(f"Unknown object id {id!r} in sequence {sequence.name}")
    visibility = [x is None or is_special(x) for x in trajectory]
    if False not in visibility:
        raise RegionException(f"Object {id!r} is never visible in sequence {sequence.name}")
    return visibility.index(False)


class MultiObjectHelper(object):
    """Helper class for multi-object sequences.

    It provides methods for querying active objects at a given frame.
    """

    def __init__(self, sequence: Sequence) -> None:
        """Initialize the helper class.

        :param sequence: The sequence to be used.
        """
        self._sequence = sequence
        ids = list(sequence.objects())
        starts: list[int] = [_objectstart(sequence, oid) for oid in ids]
        # ``self._ids`` holds ``(first_visible_frame, object_id)`` pairs sorted
        # by frame so ``new``/``objects`` can do a single linear pass.
        self._ids: list[tuple[int, str]] = sorted(zip(starts, ids), key=lambda x: x[0])

    def new(self, position: int) -> list[str]:
        """Returns a list of objects that appear at the given frame.

        :param position: The frame number.

        :returns: A list of object ids."""
        return [oid for start, oid in self._ids if start == position]

    def objects(self, position: int) -> list[str]:
        """Returns a list of objects that are active at the given frame.

        :param position: The frame number.

        :returns: A list of object ids."""
        return [oid for start, oid in self._ids if start <= position]

    def all(self) -> list[str]:
        """Returns a list of all objects in the sequence.

        :returns: A list of object ids."""
        return [oid for _, oid in self._ids]