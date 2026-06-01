"""Drawing utilities for visualizing results."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from io import BytesIO
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import numpy as np
import numpy.typing as npt
import cv2

if TYPE_CHECKING:
    from vot.region import Region

def show_image(a: npt.NDArray[np.uint8]) -> None:
    """Shows an image in the IPython notebook."""
    try:
        from IPython.display import display, Image as IPImage # type: ignore
    except ImportError:
        return

    a = a.astype(np.uint8)
    f = BytesIO()
    Image.fromarray(a).save(f, "png")
    display(IPImage(data=f.getvalue()))

_PALETTE = {
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
    "gray": (0.5, 0.5, 0.5),
}

@dataclass(frozen=True)
class Color:
    """An RGBA colour with components in the ``[0, 1]`` range.

    Replaces the loose ``tuple[float, ...]`` colour representation: it makes the number of
    components and their range explicit, and converts to the forms the various backends
    expect (Matplotlib RGBA tuples, 8-bit Pillow/OpenCV tuples).
    """

    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def resolve(cls, color: "ColorLike") -> "Color":
        """Resolves a :class:`Color`, palette name, or float RGB/RGBA tuple to a ``Color``.

        Unknown palette names resolve to black. Tuple components are clamped to ``[0, 1]``.
        """
        if isinstance(color, Color):
            return color
        if isinstance(color, str):
            rgb = _PALETTE.get(color)
            return cls(*rgb) if rgb is not None else cls(0.0, 0.0, 0.0)
        values = [float(np.clip(v, 0.0, 1.0)) for v in tuple(color)[:4]]
        return cls(*values)

    def rgb(self) -> tuple[float, float, float]:
        """Returns the colour as an ``(r, g, b)`` float tuple."""
        return (self.r, self.g, self.b)

    def rgba(self) -> tuple[float, float, float, float]:
        """Returns the colour as an ``(r, g, b, a)`` float tuple (e.g. for Matplotlib)."""
        return (self.r, self.g, self.b, self.a)

    def with_alpha(self, alpha: float) -> "Color":
        """Returns a copy of the colour with the given alpha."""
        return Color(self.r, self.g, self.b, alpha)

    def to_int(self, alpha: int = 255) -> tuple[int, int, int, int]:
        """Returns the colour as an 8-bit ``(r, g, b, alpha)`` tuple (e.g. for Pillow/OpenCV)."""
        return (int(self.r * 255), int(self.g * 255), int(self.b * 255), alpha)


ColorLike = Color | str | tuple[float, ...]


def resolve_color(color: ColorLike) -> tuple[float, float, float]:
    """Resolves a colour to an ``(r, g, b)`` float tuple.

    Backwards-compatible helper; new code should use :meth:`Color.resolve`.
    """
    return Color.resolve(color).rgb()

class DrawHandle(object):
    """Base class for drawing handles."""

    def __init__(self, color: ColorLike = (1, 0, 0), width: int = 1, fill: bool = False) -> None:
        """Initializes the drawing handle.

        :param color: Color of the drawing handle.
        :type color: Color, tuple or str
        :param width: Width of the drawing handle.
        :type width: int
        :param fill: Whether to fill the drawing handle.
        :type fill: bool
        """
        self._color: Color = Color.resolve(color)
        self._width = width
        self._fill: Color | None = self._color.with_alpha(0.4) if fill else None

    def style(self, color: ColorLike = (1, 0, 0), width: int = 1, fill: bool = False) -> 'DrawHandle':
        """Sets the style of the drawing handle. Returns self for chaining.

        :param color: Color of the drawing handle.
        :type color: Color, tuple or str
        :param width: Width of the drawing handle.
        :type width: int
        :param fill: Whether to fill the drawing handle.
        :type fill: bool

        :returns: self"""
        self._color = Color.resolve(color)
        self._width = width
        self._fill = self._color.with_alpha(0.4) if fill else None
        return self

    def region(self, region: "Region") -> "DrawHandle":
        """Draws a region."""
        region.draw(self)
        return self

    def image(self, image: npt.NDArray | Image.Image, offset: tuple[int, int] | None = None) -> "DrawHandle":
        """Draws an image at the given offset."""
        return self

    def line(self, p1: tuple[float, float], p2: tuple[float, float]) -> "DrawHandle":
        """Draws a line between two points."""
        return self

    def lines(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws a line between multiple points."""
        return self

    def polygon(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws a polygon."""
        return self

    def points(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws points."""
        return self

    def rectangle(self, left: float, top: float, right: float, bottom: float) -> "DrawHandle":
        """Draws a rectangle.

        The rectangle is defined by the top-left and bottom-right corners.
        """
        self.polygon([(left, top), (right, top), (right, bottom), (left, bottom)])
        return self

    def mask(self, mask: npt.NDArray, offset: tuple[int, int] = (0, 0)) -> "DrawHandle":
        """Draws a mask."""
        return self

class MatplotlibDrawHandle(DrawHandle):
    """Draw handle for Matplotlib.

    This handle is used for drawing to a Matplotlib axis.
    """

    def __init__(self, axis: Axes, color: ColorLike = (1, 0, 0), width: int = 1, fill: bool = False, size: tuple[int, int] | None = None) -> None:
        """Initializes a new instance of the MatplotlibDrawHandle class."""
        super().__init__(color, width, fill)
        self._axis = axis
        self._size = size
        if not self._size is None:
            self._axis.set_xlim(left=0, right=self._size[0])
            self._axis.set_ylim(top=0, bottom=self._size[1])


    def image(self, image: npt.NDArray | Image.Image, offset: tuple[int, int] | None = None) -> "DrawHandle":
        """Draws an image at the given offset."""

        if offset is None:
            offset = (0, 0)

        if isinstance(image, np.ndarray):
            width = image.shape[1]
            height = image.shape[0]
        elif isinstance(image, Image.Image):
            width = image.size[0]
            height = image.size[1]
        else:
            raise TypeError("Unsupported image type: {}".format(type(image)))

        self._axis.imshow(image, extent=(offset[0],
                offset[0] + width, offset[1] + height, offset[1]))

        return self

    def line(self, p1: tuple[float, float], p2: tuple[float, float]) -> "DrawHandle":
        """Draws a line between two points."""
        self._axis.plot((p1[0], p2[0]), (p1[1], p2[1]), linewidth=self._width, color=self._color.rgba())
        return self

    def lines(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws a line between multiple points."""
        x = [x for x, _ in points]
        y = [y for _, y in points]
        self._axis.plot(x, y, linewidth=self._width, color=self._color.rgba())
        return self

    def polygon(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws a polygon."""
        if self._fill:
            poly = Polygon(points, edgecolor=self._color.rgba(), linewidth=self._width, fill=True, color=self._fill.rgba())
        else:
            poly = Polygon(points, edgecolor=self._color.rgba(), linewidth=self._width, fill=False)
        self._axis.add_patch(poly)
        return self

    def points(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws points."""
        x, y = zip(*points)
        self._axis.plot(x, y, markeredgecolor=self._color.rgba(), markeredgewidth=self._width, linewidth=0)
        return self

    def mask(self, mask: npt.NDArray, offset: tuple[int, int] = (0, 0)) -> "DrawHandle":
        """Draws a mask."""
        # TODO: segmentation should also have option of non-filled
        mask[mask != 0] = 1
        if self._fill:
            mask = 2 * mask - cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=self._width, borderValue=0)
            cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._fill.rgba(), self._color.rgba()]))
            self._axis.imshow(mask, cmap=cmap, interpolation='none', extent=(offset[0],
                offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]))
        else:
            mask = mask - cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=self._width, borderValue=0)
            cmap = colors.ListedColormap(np.array([[0, 0, 0, 0], self._color.rgba()]))
            self._axis.imshow(mask, cmap=cmap, interpolation='none', extent=(offset[0],
                offset[0] + mask.shape[1], offset[1] + mask.shape[0], offset[1]))

        if not self._size is None:
            self._axis.set_xlim(left=0, right=self._size[0])
            self._axis.set_ylim(top=0, bottom=self._size[1])
        return self


class ImageDrawHandle(DrawHandle):
    """Draw handle for Pillow.

    This handle is used for drawing to a Pillow image.
    """

    def __init__(self, image: npt.NDArray | Image.Image, color: ColorLike = (1, 0, 0), width: int = 1, fill: bool = False) -> None:
        """Initializes a new instance of the ImageDrawHandle class."""
        super().__init__(color, width, fill)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        self._image = image
        self._handle = ImageDraw.Draw(self._image, 'RGBA')

    @property
    def array(self) -> npt.NDArray:
        """Returns the image as a numpy array."""
        return np.asarray(self._image)

    @property
    def snapshot(self) -> Image.Image:
        """Returns a snapshot of the current image."""
        return self._image.copy()

    def image(self, image: npt.NDArray | Image.Image, offset: tuple[int, int] | None = None) -> "DrawHandle":
        """Draws an image at the given offset."""
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        if offset is None:
            offset = (0, 0)
        self._image.paste(image, offset)
        return self

    def line(self, p1: tuple[float, float], p2: tuple[float, float]) -> "DrawHandle":
        """Draws a line between two points."""
        color = self._color.to_int()
        self._handle.line([p1, p2], fill=color, width=self._width)
        return self

    def lines(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws a line between multiple points."""
        if len(points) == 0:
            return self
        color = self._color.to_int()
        self._handle.line(points, fill=color, width=self._width)
        return self

    def polygon(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws a polygon."""
        if len(points) == 0:
            return self

        if self._fill:
            color = self._color.to_int(alpha=128)
            self._handle.polygon(points, fill=color)

        color = self._color.to_int()
        self._handle.line(points + [points[0]], fill=color, width=self._width)
        return self

    def points(self, points: list[tuple[float, float]]) -> "DrawHandle":
        """Draws points."""
        color = self._color.to_int()
        for (x, y) in points:
            self._handle.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=self._width)
        return self

    def mask(self, mask: npt.NDArray, offset: tuple[int, int] = (0, 0)) -> "DrawHandle":
        """Draws a mask."""
        if mask.size == 0:
            return self

        if self._fill:
            image = Image.fromarray(mask * 128, mode="L")
            color = self._color.to_int(128)
            self._image.paste(color, offset, mask=image)

        image = Image.fromarray((mask - cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=self._width, borderValue=0)) * 255, mode="L")
        color = self._color.to_int()
        self._handle.bitmap(offset, image, fill=color)

        return self