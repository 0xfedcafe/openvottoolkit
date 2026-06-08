"""This module contains classes for region representation and manipulation.

Regions are also used to represent results of trackers as well as groundtruth
trajectories. The module also contains functions for calculating overlaps between
regions and for converting between different region types.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from enum import Enum, IntEnum

from vot import ToolkitException
from vot.utilities.draw import DrawHandle

class RegionException(ToolkitException):
    """General region exception."""

class ConversionException(RegionException):
    """Region conversion exception, the conversion cannot be performed."""
    def __init__(self, *args, source: Region | None = None) -> None:
        """Constructor.

        :param *args: Arguments for the base exception

        :param source: Source region (default: {None})
        :type source: Region
        """
        super().__init__(*args)
        self._source = source

class Region(ABC):
    """Base class for all region containers."""
    def __init__(self) -> None:
        """Base constructor."""
        super().__init__()

    @abstractmethod
    def copy(self) -> "Region":
        """Copy region to another object.

        :returns: Region -- Copy of the region"""
        raise NotImplementedError
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if region is empty.

        :returns: bool -- True if the region is empty, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Encode the region as its single-line text representation.

        This is the on-disk encoding used by :func:`vot.region.io.write_trajectory`
        for text trajectory files, so every concrete region type must implement it.

        :returns: str -- Text encoding of the region"""
        raise NotImplementedError

    @abstractmethod
    def draw(self, handle: DrawHandle) -> None:
        """Draw the region using the given draw handle.

        Every concrete region is drawable; :meth:`vot.utilities.draw.DrawHandle.region`
        and the report/notebook renderers rely on this.

        :param handle: The draw handle to render the region with."""
        raise NotImplementedError

class SpecialCode(IntEnum):
    """Semantic code carried by a :class:`Special` region.

    A ``Special`` region marks a frame that holds no shape; the code records why.
    The integer value is also the on-disk encoding -- trajectory files store a
    special region as this bare number -- so members must keep their values stable.
    """

    UNKNOWN = 0        #: Object state is unknown for this frame (e.g. padded groundtruth).
    INITIALIZATION = 1 #: Frame where the tracker is (re)initialized.
    FAILURE = 2        #: Tracker self-reported lost target (low overlap); reinitialized afterwards under supervision.
    CRASH = 3          #: Tracker process failed (exception/timeout); no output produced for this frame.

class Special(Region):
    """Special region: a frame with no shape, tagged with a :class:`SpecialCode`.

    :var code: The :class:`SpecialCode` for this region. Unrecognized integer codes
        read from legacy trajectory files are preserved as a plain ``int``.
    """

    def __init__(self, code: "SpecialCode | int") -> None:
        """Constructor.

        :param code: A :class:`SpecialCode` member, or an integer code. Known integer
            values are normalized to the matching :class:`SpecialCode` member.
        """
        super().__init__()
        try:
            self._code: SpecialCode | int = SpecialCode(int(code))
        except ValueError:
            self._code = int(code)

    def __str__(self) -> str:
        """Create string from class."""
        return str(int(self._code))

    @staticmethod
    def convert(region: Region) -> "Special":
        """Convert region to special region. Note that some conversions degrade
        information.

        :param region: Region to convert
        :type region: Region

        :raises ConversionException: Unable to convert region to special region
        :returns: Special -- Converted region"""
        if isinstance(region, Special):
            return region.copy()
        raise ConversionException(
            "Unable to convert {} region to special region".format(type(region).__name__),
            source=region,
        )

    @property
    def code(self) -> "SpecialCode | int":
        """Returns the special code for this region.

        :returns: The :class:`SpecialCode` member, or a plain ``int`` for an
            unrecognized legacy code."""
        return self._code

    def draw(self, handle: DrawHandle) -> None:
        """Draw region to the image using the provided handle.

        :param handle: Draw handle
        :type handle: DrawHandle
        """
        pass

    def is_empty(self) -> bool:
        """Check if region is empty.

        Special regions are always empty by definition.
        """
        return True

    def copy(self) -> "Special":
        """Create a copy of the special region."""
        return Special(self._code)

class Point(Region):
    """Point region — a single (x, y) coordinate without extent."""

    def __init__(self, x: float, y: float) -> None:
        """Constructor.

        :param x: X coordinate
        :param y: Y coordinate
        """
        super().__init__()
        self._x = float(x)
        self._y = float(y)

    def __str__(self) -> str:
        """Create string from class."""
        return '{},{}'.format(self._x, self._y)

    def copy(self) -> "Point":
        """Copy region to another object."""
        return Point(self._x, self._y)

    @staticmethod
    def convert(region: Region) -> "Point":
        """Convert region to point region. Note that some conversions degrade
        information.

        :param region: Region to convert

        :raises ConversionException: Unable to convert region to point region
        :returns: Converted region"""
        if isinstance(region, Point):
            return region.copy()
        raise ConversionException(
            "Unable to convert {} region to point region".format(type(region).__name__),
            source=region,
        )

    @property
    def x(self) -> float:
        """Returns X coordinate of the point."""
        return self._x

    @property
    def y(self) -> float:
        """Returns Y coordinate of the point."""
        return self._y

    def draw(self, handle: DrawHandle) -> None:
        """Draw region to the image using the provided handle.

        :param handle: Draw handle
        """
        handle.points([(self._x, self._y)])

    def resize(self, scale: float) -> "Point":
        """Resize region by the provided scale factor.

        :param scale: Scale factor
        """
        return Point(self._x * scale, self._y * scale)

    def is_empty(self) -> bool:
        """Check if region is empty.

        Point regions are never empty by definition.
        """
        return False


from .raster import calculate_overlap, calculate_overlaps
from .shapes import Rectangle, Polygon, Mask

def is_special(region: Region | None, code: "SpecialCode | int | None" = None) -> bool:
    """Check whether ``region`` is a :class:`Special` region.

    :param region: Region to check
    :param code: If given, additionally require the region's code to equal this value
    :returns: True if the region is special (and matches ``code`` when provided),
        False otherwise"""
    if not isinstance(region, Special):
        return False
    if code is None:
        return True
    return region.code == code

def is_shape(region: Region | None) -> bool:
    """Check if the region is a shape region.

    :param region: Region to check
    :returns: True if the region is a shape region, False otherwise"""
    return isinstance(region, (Rectangle, Polygon, Mask))
