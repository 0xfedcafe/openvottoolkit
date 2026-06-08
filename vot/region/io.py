"""Utilities for reading and writing regions from and to files."""

from __future__ import annotations

import math
from typing import Any, IO
import io

import numpy as np
import numpy.typing as npt
import numba

@numba.njit(cache=True)
def mask_to_rle(m, maxstride=100000000):
    """Converts a binary mask to RLE encoding. This is a Numba decorated function that
    is compiled just-in-time for faster execution.

    :param m: 2-D binary mask
    :type m: npt.NDArray
    :param maxstride: Maximum number of consecutive 0s or 1s in the RLE encoding. If the number of consecutive 0s or 1s is larger than maxstride, it is split into multiple elements.
    :type maxstride: int

    :returns: RLE encoding of the mask
    :rtype: list[int]"""
    # reshape mask to vector
    v = m.reshape((m.shape[0] * m.shape[1]))

    if v.size == 0:
        return [0]

    # output is empty at the beginning
    rle = []
    # index of the last different element
    last_idx = 0
    # check if first element is 1, so first element in RLE (number of zeros) must be set to 0
    if v[0] > 0:
        rle.append(0)

    # go over all elements and check if two consecutive are the same
    for i in range(1, v.size):
        if v[i] != v[i - 1]:
            length = i - last_idx
            # if length is larger than maxstride, split it into multiple elements
            while length > maxstride:
                rle.append(maxstride)
                rle.append(0)
                length -= maxstride
            # add remaining length
            if length > 0:
                rle.append(length)
            last_idx = i

    if v.size > 0:
        # handle last element of rle
        if last_idx < v.size - 1:
            # last element is the same as one element before it - add number of these last elements
            length = v.size - last_idx
            while length > maxstride:
                rle.append(maxstride)
                rle.append(0)
                length -= maxstride
            if length > 0:
                rle.append(length)
        else:
            # last element is different than one element before - add 1
            rle.append(1)

    return rle

@numba.njit(cache=True)
def rle_to_mask(rle, width, height):
    """Converts RLE encoding to a binary mask. This is a Numba decorated function that
    is compiled just-in-time for faster execution.

    :param rle: RLE encoding of the mask
    :type rle: list[int]
    :param width: Width of the mask
    :type width: int
    :param height: Height of the mask
    :type height: int

    :returns: 2-D binary mask
    :rtype: npt.NDArray"""

    # allocate list of zeros
    v = np.zeros(width * height, dtype=np.uint8)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    #return np.array(v, dtype=np.uint8).reshape((height, width))
    return v.reshape((height, width))

def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)

from vot.region import Region
from vot.region.raster import mask_bounds

def encode_mask(mask: npt.NDArray) -> tuple[tuple[int, int, int, int], list[int]]:
    """ Encode a binary mask to a string in the following format: x0, y0, w, h, RLE.

    :param mask: 2-D binary mask

    :returns: ``((tl_x, tl_y, region_w, region_h), rle)`` describing the minimal
        bounding box of the foreground and its RLE encoding."""
    # handle the case when the mask is empty: ``mask_bounds`` returns (0, 0, 0, 0)
    # for an all-zero mask (it never returns None), which is indistinguishable from a
    # single foreground pixel at the origin, so test the pixels directly.
    if not mask.any():
        return (0, 0, 0, 0), [0]

    # calculate coordinates of the top-left corner and region width and height (minimal region containing all 1s)
    x_min, y_min, x_max, y_max = mask_bounds(mask)

    tl_x = x_min
    tl_y = y_min
    region_w = x_max - x_min + 1
    region_h = y_max - y_min + 1

    # extract target region from the full mask and calculate RLE
    # do not use full mask to optimize speed and space
    target_mask = mask[tl_y:tl_y + region_h, tl_x:tl_x + region_w]
    rle = mask_to_rle(np.array(target_mask))

    return (tl_x, tl_y, region_w, region_h), rle


def parse_region(string: str, separator: str = ",") -> "Region":
    """Parse input string to the appropriate region format and return Region object.

    :param string: comma separated list of values
    :param separator: separator of values in the input string

    :returns: parsed region
    :raises RegionException: if ``string`` does not match any known region format
    """
    from vot import config
    from vot.region import Special, SpecialCode, Point, RegionException
    from vot.region.shapes import Rectangle, Polygon, Mask

    if not string:
        raise RegionException("Cannot parse region from an empty string")

    if string[0] == 'm':
        # input is a mask - decode it
        m_, offset_ = create_mask_from_string(string[1:].split(separator))
        return Mask(m_, offset=offset_, optimize=config.mask_optimize_read)

    # input is not a mask - check if special, rectangle or polygon
    tokens = [float(t) for t in string.split(separator)]

    # A region line filled entirely with non-finite values (NaN/Inf) is the
    # trajectory-format sentinel for "object absent in this frame". A line that
    # is only partially non-finite is corrupt -- not a sanctioned encoding --
    # so fail loudly rather than silently guess a region.
    nonfinite = [not math.isfinite(t) for t in tokens]
    if all(nonfinite):
        return Special(SpecialCode.UNKNOWN)
    if any(nonfinite):
        raise RegionException(
            "Region line mixes finite and non-finite values: {!r}".format(string)
        )

    if len(tokens) == 1:
        return Special(int(tokens[0]))
    if len(tokens) == 2:
        return Point(tokens[0], tokens[1])
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    if len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([(x_, y_) for x_, y_ in zip(tokens[::2], tokens[1::2])])

    # ``parse_region`` is declared to return a ``Region`` — any unparseable input is
    # a caller error and must raise rather than silently return ``None``.
    raise RegionException(
        "Cannot parse region from {} token(s): {!r}".format(len(tokens), string)
    )

def read_trajectory_binary(fp: IO[bytes]) -> list["Region"]:
    """Reads a trajectory from a binary file and returns a list of regions.

    :param fp: Binary file handle. Any object that exposes ``.read()`` returning
        bytes works — ``io.RawIOBase``, ``io.BufferedIOBase``, and ``io.BytesIO``
        are all accepted.

    :returns: List of regions"""
    import struct
    from cachetools import LRUCache, cached
    from vot.region import Special, Point
    from vot.region.shapes import Rectangle, Polygon, Mask

    data_bytes: bytes = fp.read()
    cursor: int = 0

    @cached(cache=LRUCache(maxsize=32))
    def calcsize(format: str) -> int:
        """Calculate size of the struct format."""
        return struct.calcsize(format)

    def read(format: str) -> tuple[Any, ...]:
        """Read struct from the buffer and update offset."""
        nonlocal cursor
        unpacked = struct.unpack_from(format, data_bytes, cursor)
        cursor += calcsize(format)
        return unpacked

    _, length = read("<hI")

    trajectory: list["Region"] = []

    for _ in range(length):
        type_code, = read("<B")
        r: "Region"
        if type_code == 0:
            r = Special(*read("<I"))
        elif type_code == 1:
            r = Rectangle(*read("<ffff"))
        elif type_code == 2:
            n, = read("<H")
            values = read("<%df" % (2 * n))
            r = Polygon(list(zip(values[0::2], values[1::2])))
        elif type_code == 3:
            tl_x, tl_y, region_w, region_h, n = read("<hhHHH")
            rle = np.array(read("<%dH" % n), dtype=np.int32)
            r = Mask(rle_to_mask(rle, region_w, region_h), (tl_x, tl_y))
        elif type_code == 4:
            r = Point(*read("<ff"))
        else:
            raise IOError("Wrong region type")
        trajectory.append(r)
    return trajectory


def write_trajectory_binary(fp: IO[bytes], data: list["Region"]) -> None:
    """Writes a trajectory to a binary file.

    :param fp: Binary file handle accepting ``bytes`` writes.
    :param data: List of regions
    """
    import struct
    from vot.region import Special, Point
    from vot.region.shapes import Rectangle, Polygon, Mask

    fp.write(struct.pack("<hI", 1, len(data)))

    for r in data:
        if isinstance(r, Special):
            fp.write(struct.pack("<BI", 0, r.code))
        elif isinstance(r, Point):
            fp.write(struct.pack("<Bff", 4, r.x, r.y))
        elif isinstance(r, Rectangle):
            fp.write(struct.pack("<Bffff", 1, r.x, r.y, r.width, r.height))
        elif isinstance(r, Polygon):
            fp.write(struct.pack("<BH%df" % (2 * r.size), 2, r.size, *[item for sublist in r.points() for item in sublist]))
        elif isinstance(r, Mask):
            rle = mask_to_rle(r.mask, maxstride=255 * 255)
            fp.write(struct.pack("<BhhHHH%dH" % len(rle), 3, r.offset[0], r.offset[1], r.mask.shape[1], r.mask.shape[0], len(rle), *rle))
        else:
            raise IOError(f"Wrong region type {type(r).__name__}")


def _looks_binary(fp: Any) -> bool:
    """Best-effort detection of whether an opened file handle yields bytes.

    Returns ``True`` for raw/buffered binary streams (and ``io.BytesIO``), ``False``
    for text streams. Used by :func:`read_trajectory` and :func:`write_trajectory`
    to decide which branch to take when the caller passes an already-open handle.
    """
    if isinstance(fp, (io.RawIOBase, io.BufferedIOBase, io.BytesIO)):
        return True
    if isinstance(fp, io.TextIOBase):
        return False
    # Some handles (e.g. ``gzip.GzipFile``) don't inherit from ``io`` base classes;
    # fall back to inspecting the ``mode`` attribute when present.
    mode = getattr(fp, "mode", "")
    return "b" in mode if isinstance(mode, str) else False


def read_trajectory(fp: str | IO[Any], separator: str = ",") -> list["Region"]:
    """Reads a trajectory from a file and returns a list of regions.

    :param fp: File path or already-open file handle (text or binary).
    :param separator: Separator of values in the region, only used for text files.

    :returns: List of regions"""
    binary: bool
    handle: IO[Any]
    close: bool
    if isinstance(fp, str):
        try:
            import struct
            # Read-only ('rb'): the file is only peeked at to sniff the binary header,
            # so opening it read-write ('r+b') would needlessly fail on read-only files
            # and silently mis-parse a binary trajectory as text.
            with open(fp, "rb") as tfp:
                v, = struct.unpack("<h", tfp.read(struct.calcsize("<h")))
                binary = v == 1
        except Exception:
            binary = False
        handle = open(fp, "rb" if binary else "r")
        close = True
    else:
        handle = fp
        binary = _looks_binary(handle)
        close = False

    regions: list["Region"]
    if binary:
        regions = read_trajectory_binary(handle)
    else:
        regions = [parse_region(line.strip(), separator) for line in handle.readlines()]

    if close:
        handle.close()

    return regions


def write_trajectory(fp: str | IO[Any], data: list["Region"]) -> None:
    """Write a trajectory to a file handle or a file with a given name. Based on the
    suffix of a file or properties of a file handle, the output may be either text based
    or binary.

    :param fp: File handle or file name
    :param data: Trajectory, a list of region objects

    :raises IOError: If the file format is not supported, or the trajectory
        contains an element that is not a region"""

    for region in data:
        if not isinstance(region, Region):
            raise IOError("Trajectory contains a non-region element: {!r}".format(region))

    binary: bool
    handle: IO[Any]
    close: bool
    if isinstance(fp, str):
        binary = fp.endswith(".bin")
        handle = open(fp, "wb" if binary else "w")
        close = True
    else:
        handle = fp
        binary = _looks_binary(handle)
        close = False

    if binary:
        write_trajectory_binary(handle, data)
    else:
        for region in data:
            handle.write(str(region) + "\n")

    if close:
        handle.close()
