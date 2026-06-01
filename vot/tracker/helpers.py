"""Helpers for trackers integrated through the native ``python`` runtime protocol.

A ``python``-protocol tracker (see :mod:`vot.tracker.python`) runs in an isolated
worker process and exchanges data with the toolkit over a queue, so it only ever
sees plain serializable values: regions arrive *encoded* (tuples / dicts) and
frames arrive as image file paths. These helpers convert between that wire form
and the toolkit's :class:`~vot.region.Region` objects, normalise the per-frame
object argument and load frame images -- so every tracker wrapper does not have
to re-implement the same protocol glue.

:func:`encode_region`, :func:`decode_region` and :func:`convert_region` also back
:class:`vot.tracker.python.PythonRuntime` itself, so a tracker and the runtime
are guaranteed to agree on the encoding.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from vot.region import Region, Mask, Rectangle, Point, Polygon
from vot.utilities import normalize_path

if TYPE_CHECKING:
    from vot.tracker import Tracker

#: Wire form of a region: a 4-tuple (rectangle), a 2-tuple (point), a list of
#: point pairs (polygon) or a ``{"mask", "offset"}`` dict (mask).
EncodedRegion = dict | tuple | list


def encode_region(region: Region) -> EncodedRegion:
    """Encode a :class:`~vot.region.Region` into its plain serializable wire form."""
    if isinstance(region, Mask):
        # Masks carry both a bitmap and an offset; serialize both so the region
        # survives the trip to the worker process intact.
        return {"mask": region.mask.tolist(), "offset": list(region.offset)}
    if isinstance(region, Rectangle):
        return (region.x, region.y, region.width, region.height)
    if isinstance(region, Point):
        return (region.x, region.y)
    if isinstance(region, Polygon):
        return [(float(x), float(y)) for x, y in region.points()]
    raise ValueError("Unknown region type: {}".format(type(region)))


def decode_region(data: Any) -> Region:
    """Decode a region wire payload back into a :class:`~vot.region.Region`.

    The concrete type mirrors what :func:`encode_region` produced -- mask,
    polygon, rectangle or point.

    :raises ValueError: if ``data`` is not a recognised region payload.
    """
    if isinstance(data, dict) and "mask" in data:
        offset = data["offset"]
        return Mask(np.asarray(data["mask"], dtype=np.uint8),
                    (int(offset[0]), int(offset[1])))
    if isinstance(data, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
        return Polygon([(float(px), float(py)) for px, py in data])
    if isinstance(data, (list, tuple)) and len(data) == 4:
        return Rectangle(*(float(v) for v in data))
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return Point(*(float(v) for v in data))
    raise ValueError("Cannot decode region from payload: {!r}".format(data))


def convert_region(region: Region, target: str | None) -> Region:
    """Convert ``region`` to a target shape.

    :param target: One of ``"mask"``, ``"rectangle"``, ``"point"`` or
        ``"polygon"``; ``None`` returns ``region`` unchanged.
    """
    if target is None:
        return region
    target = target.lower()
    if target == "mask":
        return Mask.convert(region)
    if target == "rectangle":
        return Rectangle.convert(region)
    if target == "point":
        return Point.convert(region)
    if target == "polygon":
        return Polygon.convert(region)
    raise ValueError("Unknown target region type: {}".format(target))


def normalize_paths(paths: list[str], tracker: "Tracker") -> list[str]:
    """Normalizes a list of paths relative to the tracker source.

    :param paths: The paths to normalize.
    :param tracker: The tracker whose source directory is the normalization root.

    :returns: The normalized paths."""
    root = os.path.dirname(tracker.source)
    return [normalize_path(path, root) for path in paths]


def encode_rectangle(box: tuple) -> tuple:
    """Encode a raw ``(x, y, w, h)`` box into the rectangle wire form.

    Convenience for trackers that compute boxes directly; equivalent to
    ``encode_region(Rectangle(*box))``.
    """
    x, y, w, h = box
    return (float(x), float(y), float(w), float(h))


def normalize_new(new: Any) -> list:
    """Normalise the runtime's per-frame ``new`` argument into a list of pairs.

    Across the call styles (single-/multi-object, init/update) ``new`` arrives as
    ``None``, a single ``(encoded_region, properties)`` pair, or a list of such
    pairs. This collapses all of them to a list of ``(encoded_region, properties)``.
    """
    if new is None:
        return []
    # A single (encoded_region, properties) pair: ``properties`` is a dict, which
    # distinguishes it from a list of pairs.
    if isinstance(new, tuple) and len(new) == 2 and isinstance(new[1], dict):
        return [new]
    if isinstance(new, list):
        return list(new)
    return [new]


def read_frame(frame: Any, channel: str | None = None) -> npt.NDArray:
    """Load the image for a runtime ``frame`` payload as a BGR ``numpy`` array.

    :param frame: An image path for a single-channel sequence, or a
        ``{channel: path}`` mapping for a multi-channel one.
    :param channel: Channel to load from a multi-channel ``frame``; when omitted a
        colour channel is preferred, falling back to any available one.

    :raises IOError: if the image file cannot be read.
    """
    import cv2

    if isinstance(frame, dict):
        if channel is not None and channel in frame:
            source = frame[channel]
        else:
            source = next((frame[k] for k in ("color", "rgb") if k in frame),
                           next(iter(frame.values())))
    else:
        source = frame

    path = str(source)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("Unable to read image: {}".format(path))
    return image
