"""Module for region shapes."""

from copy import copy
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import cv2

from vot.region import Region, ConversionException
from vot.utilities.draw import DrawHandle

class Shape(Region, ABC):
    """Base class for all shape regions."""

    @abstractmethod
    def draw(self, handle: DrawHandle) -> None:
        """Draw the region to the given handle."""
        raise NotImplementedError

    @abstractmethod
    def resize(self, factor=1) -> "Shape":
        """Resize the region by the given factor."""
        raise NotImplementedError

    @abstractmethod
    def move(self, dx: float = 0.0, dy: float = 0.0) -> "Shape":
        """Move the region by the given offset.

        :param dx: X offset. Defaults to 0.
        :type dx: float, optional
        :param dy: Y offset. Defaults to 0.
        :type dy: float, optional

        :returns: Moved region.
        :rtype: Shape"""
        raise NotImplementedError

    @abstractmethod
    def rasterize(self, bounds: tuple[int, int, int, int]) -> npt.NDArray:
        """Rasterize the region to a binary mask.

        :param bounds: Bounds of the mask.
        :type bounds: tuple[int, int, int, int]

        :returns: Binary mask.
        :rtype: npt.NDArray"""
        raise NotImplementedError

    @abstractmethod
    def bounds(self) -> tuple[int, int, int, int]:
        """Get the bounding box of the region.

        :returns: Bounding box (left, top, right, bottom).
        :rtype: tuple[int, int, int, int]"""
        raise NotImplementedError

class Rectangle(Shape):
    """Rectangle region class for representing rectangular regions."""
    def __init__(self, x: float = 0.0, y: float = 0.0, width: float = 0.0, height: float = 0.0):
        """Constructor for rectangle region.

        :param x: X coordinate of the top left corner. Defaults to 0.
        :type x: float, optional
        :param y: Y coordinate of the top left corner. Defaults to 0.
        :type y: float, optional
        :param width: Width of the rectangle. Defaults to 0.
        :type width: float, optional
        :param height: Height of the rectangle. Defaults to 0.
        :type height: float, optional
        """
        super().__init__()
        self._data = np.array([[x], [y], [width], [height]], dtype=np.float32)
    
    @staticmethod
    def from_2points(x1: float, y1: float, x2: float, y2: float) -> "Rectangle":
        """Create a rectangle from two points.

        :param x1: X left coordinate 
        :type x1: float
        :param y1: Y top coordinate
        :type y1: float
        :param x2: X right coordinate
        :type x2: float
        :param y2: Y bottom coordinate
        :type y2: float

        :returns: Rectangle created from the two points.
        :rtype: Rectangle"""
        return Rectangle(x1, y1, x2 - x1, y2 - y1)
    
    def to_2points(self) -> tuple[float, float, float, float]:
        """Get the rectangle as two points.

        :returns: Two points (x1, y1, x2, y2) where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
        :rtype: tuple[float, float, float, float]"""
        return self.x, self.y, self.x + self.width, self.y + self.height
    
    @staticmethod
    def populate_points(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
        """Populate the rectangle points from two points.

        :param x1: X left coordinate 
        :type x1: float
        :param y1: Y top coordinate
        :type y1: float
        :param x2: X right coordinate
        :type x2: float
        :param y2: Y bottom coordinate
        :type y2: float

        :returns: List of points [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        :rtype: list"""
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def __str__(self) -> str:
        """Create string from class."""
        return '{},{},{},{}'.format(self.x, self.y, self.width, self.height)

    @property
    def x(self) -> float:
        """X coordinate of the top left corner."""
        return float(self._data[0, 0])

    @property
    def y(self) -> float:
        """Y coordinate of the top left corner."""
        return float(self._data[1, 0])

    @property
    def width(self) -> float:
        """Width of the rectangle."""
        return float(self._data[2, 0])

    @property
    def height(self) -> float:
        """Height of the rectangle."""
        return float(self._data[3, 0])

    @staticmethod
    def convert(region: Region) -> "Rectangle":
        """Convert region to rectangle region. Note that some conversions degrade
        information.

        :param region: Region to convert
        :type region: Region

        :raises ConversionException: Unable to convert region to rectangle region
        :returns: Rectangle -- Converted region"""
        if isinstance(region, Rectangle):
            return region.copy()
        elif isinstance(region, Polygon):
            from vot.region.raster import _bounds_polygon
            bounds = _bounds_polygon(region._points)

            return Rectangle.from_2points(*bounds)
        elif isinstance(region, Mask):
            return Rectangle.from_2points(*region.bounds()) 
        else:
            raise ConversionException("Unable to convert {} region to rectangle region".format(type(region)), source=region)

    def copy(self) -> "Rectangle":
        """Copy region to another object."""
        return copy(self)

    def is_empty(self) -> bool:
        """Check if the region is empty.

        :returns: True if the region is empty, False otherwise.
        :rtype: bool"""
        if self.width > 0 and self.height > 0:
            return False
        else:
            return True

    def draw(self, handle: DrawHandle) -> None:
        """Draw the region to the given handle.

        :param handle: Handle to draw to.
        :type handle: DrawHandle
        """
        handle.rectangle(*self.to_2points())

    def resize(self, factor: float = 1.0) -> "Rectangle":
        """Resize the region by the given factor.

        :param factor: Resize factor. Defaults to 1.
        :type factor: float, optional

        :returns: Resized region.
        :rtype: Rectangle"""
        return Rectangle(self.x * factor, self.y * factor,
                         self.width * factor, self.height * factor)

    def center(self) -> tuple[float, float]:
        """Get the center of the region.

        :returns: Center coordinates (x,y).
        :rtype: tuple"""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def move(self, dx: float = 0.0, dy: float = 0.0) -> "Rectangle":
        """Move the region by the given offset.

        :param dx: X offset. Defaults to 0.
        :type dx: float, optional
        :param dy: Y offset. Defaults to 0.
        :type dy: float, optional

        :returns: Moved region.
        :rtype: Rectangle"""
        return Rectangle(self.x + dx, self.y + dy, self.width, self.height)

    def rasterize(self, bounds: tuple[int, int, int, int]) -> npt.NDArray:
        """Rasterize the region to a binary mask.

        :param bounds: Bounds of the mask (x1,y1,x2,y2).
        :type bounds: tuple
        """
        from vot.region.raster import rasterize_rectangle
        return rasterize_rectangle(self._data, bounds)

    def bounds(self) -> tuple[int, int, int, int]:
        """Get the bounding box of the region.

        :returns: Bounding box (x1,y1,x2,y2).
        :rtype: tuple"""
        return int(round(self.x)), int(round(self.y)), int(round(self.width + self.x)), int(round(self.height + self.y))

class Polygon(Shape):
    """Polygon region defined by a list of points.

    The polygon is closed, i.e. the first and last point are connected.
    """
    def __init__(self, points: list[tuple[float, float]]) -> None:
        """Constructor.

        :param points: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
        :type points: list
        """
        super().__init__()
        assert(points)
        self._points = np.array(points, dtype=np.float32)
        assert(self._points.shape[0] >= 3 and self._points.shape[1] == 2)  # pylint: disable=E1136


    def __str__(self) -> str:
        """Create string from class."""
        return ','.join(['{},{}'.format(p[0], p[1]) for p in self._points])


    @staticmethod
    def convert(region: Region) -> "Polygon":
        """Convert region to polygon region. Note that some conversions degrade
        information.

        :param region: Region to convert
        :type region: Region

        :raises ConversionException: Unable to convert region to polygon region
        :returns: Polygon -- Converted region"""
        if isinstance(region, Polygon):
            return region.copy()
        elif isinstance(region, Rectangle):
            region_bounds = region.bounds()
            return Polygon(Rectangle.populate_points(*region_bounds))
        elif isinstance(region, Mask):
            from vot.region.raster import _bounds_mask

            mask_bounds = _bounds_mask(region.mask, region.offset)
            return Polygon(Rectangle.populate_points(*mask_bounds))
        else:
            raise ConversionException("Unable to convert {} region to polygon region".format(type(region)), source=region)

    @property
    def size(self) -> int:
        """Get the number of points."""
        return self._points.shape[0] # pylint: disable=E1136

    def __getitem__(self, i: int) -> tuple[float, float]:
        """Get the i-th point."""
        return self._points[i, 0], self._points[i, 1]

    def points(self) -> list[tuple[float, float]]:
        """Get the list of points.

        :returns: List of points as tuples [(x1,y1), (x2,y2),...,(xN,yN)]
        :rtype: list"""
        return [self[i] for i in range(self.size)]

    def copy(self) -> "Polygon":
        """Create a copy of the polygon."""
        return copy(self)

    def draw(self, handle: DrawHandle) -> None:
        """Draw the polygon on the given handle.

        :param handle: Handle to draw on.
        :type handle: DrawHandle
        """
        handle.polygon(self._points.tolist())

    def resize(self, factor: float = 1.0) -> "Polygon":
        """Resize the polygon by a factor.

        :param factor: Resize factor.
        :type factor: float

        :returns: Resized polygon.
        :rtype: Polygon"""
        # Same as in draw, tolist returns list[tuple[float, float]]
        return Polygon((self._points * factor).tolist())

    def move(self, dx: float = 0.0, dy: float = 0.0) -> "Polygon":
        """Move the polygon by a given offset.

        :param dx: X offset.
        :type dx: float
        :param dy: Y offset.
        :type dy: float

        :returns: Moved polygon.
        :rtype: Polygon"""
        offset = np.array([dx, dy], dtype=np.float32)
        return Polygon((self._points + offset).tolist())

    def is_empty(self) -> bool:
        """Check if the polygon is empty.

        :returns: True if the polygon is empty, False otherwise.
        :rtype: bool"""
        mins = np.min(self._points, axis=0)
        maxs = np.max(self._points, axis=0)
        return mins[0] == maxs[0] or mins[1] == maxs[1]

    def rasterize(self, bounds: tuple[int, int, int, int]) -> npt.NDArray:
        """Rasterize the polygon into a binary mask.

        :param bounds: Bounding box of the mask as (left, top, right, bottom).
        :type bounds: tuple

        :returns: Binary mask.
        :rtype: npt.NDArray"""
        from vot.region.raster import rasterize_polygon
        return rasterize_polygon(self._points, bounds)

    def bounds(self) -> tuple[int, int, int, int]:
        """Get the bounding box of the polygon.

        :returns: Bounding box as (left, top, right, bottom).
        :rtype: tuple"""
        from vot.region.raster import _bounds_polygon 
        return _bounds_polygon(self._points)

from vot.region.raster import mask_bounds
from vot.region.io import mask_to_rle

class Mask(Shape):
    """Mask region defined by a binary mask.

    The mask is defined by a binary image and an offset.
    """

    def __init__(self, mask: npt.NDArray, offset: tuple[int, int] = (0, 0), optimize: bool = False) -> None:
        """Constructor.

        :param mask: Binary mask.
        :type mask: npt.NDArray
        :param offset: Offset of the mask as (x, y).
        :type offset: tuple
        :param optimize: Optimize the mask by removing empty rows and columns.
        :type optimize: bool
        """
        super().__init__()
        self._mask = mask.astype(np.uint8)
        self._mask[self._mask > 0] = 1
        self._offset = offset
        if optimize:  # optimize is used when mask without an offset is given (e.g. full-image mask)
            self._optimize()
            
    def __str__(self) -> str:
        """Create string from class."""
        offset_str = '%d,%d' % self.offset
        region_sz_str = '%d,%d' % (self.mask.shape[1], self.mask.shape[0])
        rle_str = ','.join([str(el) for el in mask_to_rle(self.mask)])
        return 'm%s,%s,%s' % (offset_str, region_sz_str, rle_str)

    def _optimize(self) -> None:
        """Optimize the mask by removing empty rows and columns.

        If the mask is empty, the mask is set to zero size. Do not call this method
        directly, it is called from the constructor.
        """
        bounds = mask_bounds(self.mask)
        if bounds[2] == 0:
            # mask is empty
            self._mask = np.zeros((0, 0), dtype=np.uint8)
            self._offset = (0, 0)
        else:
            self._mask = np.copy(self.mask[bounds[1]:bounds[3]+1, bounds[0]:bounds[2]+1])
            self._offset = (bounds[0] + self.offset[0], bounds[1] + self.offset[1])

    @property
    def mask(self) -> npt.NDArray:
        """Get the mask.

        Note that you should not modify the mask directly. Also make sure to take into
        account the offset when using the mask.
        """
        return self._mask

    @property
    def offset(self) -> tuple[int, int]:
        """Get the offset of the mask in pixels."""
        return self._offset

    def copy(self) -> "Mask":
        """Create a copy of the mask."""
        return copy(self)

    @staticmethod
    def convert(region: Region) -> "Mask":
        """Convert region to mask region. Note that some conversions degrade
        information.

        :param region: Region to convert
        :type region: Region

        :raises ConversionException: Unable to convert region to mask region
        :returns: Mask -- Converted region"""
        if isinstance(region, Mask):
            return region.copy()
        elif isinstance(region, Rectangle):
            # The rectangle is rasterized at its absolute coordinates (raster origin
            # (0, 0)), so the resulting mask is already in image coordinates and its
            # offset must be (0, 0). The previous (int(x), int(y)) offset double-shifted
            # the mask, leaving zero overlap with the source rectangle.
            return Mask(region.rasterize((0, 0, int(region.x + region.width), int(region.y + region.height))), (0, 0), optimize=False)
        elif isinstance(region, Polygon):
            bounds = region.bounds()
            return Mask(region.rasterize(bounds), (bounds[0], bounds[1]), optimize=False)
        else:
            raise ConversionException("Unable to convert {} region to mask region".format(type(region)), source=region)

    def draw(self, handle: DrawHandle) -> None:
        """Draw the mask into an image.

        :param handle: Handle to the image.
        :type handle: DrawHandle
        """
        handle.mask(self._mask, self.offset)

    def rasterize(self, bounds: tuple[int, int, int, int]) -> npt.NDArray:
        """Rasterize the mask into a binary mask. The mask is cropped to the given
        bounds.

        :param bounds: Bounding box of the mask as (left, top, right, bottom).
        :type bounds: tuple

        :returns: Binary mask. The mask is a copy of the original mask.
        :rtype: npt.NDArray"""
        from vot.region.raster import copy_mask
        return copy_mask(self._mask, self._offset, bounds)

    def is_empty(self) -> bool:
        """Check if the mask is empty.

        :returns: True if the mask is empty, False otherwise.
        :rtype: bool"""
        bounds = mask_bounds(self.mask)
        return bounds[2] == 0 or bounds[3] == 0

    def resize(self, factor: float = 1.0) -> "Mask":
        """Resize the mask by a given factor. The mask is resized using nearest neighbor
        interpolation.

        :param factor: Resize factor.
        :type factor: float

        :returns: Resized mask.
        :rtype: Mask"""

        offset = (int(self.offset[0] * factor), int(self.offset[1] * factor))
        height = max(1, int(self.mask.shape[0] * factor))
        width = max(1, int(self.mask.shape[1] * factor))

        if self.mask.size == 0:
            mask = np.zeros((0, 0), dtype=np.uint8)
        else:
            mask = cv2.resize(self.mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

        return Mask(mask, offset, False)

    def move(self, dx: float = 0.0, dy: float = 0.0) -> "Mask":
        """Move the mask by a given offset.

        Mask offsets are integer pixel coordinates, but the base ``Shape.move``
        signature accepts floats — incoming values are rounded to the nearest
        integer here to preserve mask pixel alignment.

        :param dx: Horizontal offset.
        :param dy: Vertical offset.

        :returns: Moved mask."""
        return Mask(
            self._mask,
            (self.offset[0] + int(round(dx)), self.offset[1] + int(round(dy))),
            False,
        )

    def bounds(self) -> tuple[int, int, int, int]:
        """Get the bounding box of the mask.

        :returns: Bounding box of the mask as (left, top, right, bottom).
        :rtype: tuple"""
        from vot.region.raster import _bounds_mask
        return _bounds_mask(self.mask, self.offset) 
