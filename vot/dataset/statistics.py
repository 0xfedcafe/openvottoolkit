"""Sequence statistics for VOT-format sequences on disk.

Numpy-backed analyses of a sequence's groundtruth: per-frame movement statistics and
fixed-length window search by object size. These operate directly on a sequence directory
(the ``groundtruth.txt`` annotations and the ``sequence`` metadata file) via the
:mod:`vot.dataset.layout` primitives.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence as TypingSequence

import numpy as np
import numpy.typing as npt

from vot.dataset.layout import GROUNDTRUTH_FILE, read_metadata
from vot.region.io import read_trajectory
from vot.region.shapes import Rectangle

logger = logging.getLogger("vot")


def _read_rectangles(sequence_dir: str | Path) -> list[Rectangle]:
    """Reads ``groundtruth.txt`` from a sequence directory as a list of :class:`Rectangle` instances.

    Regions that are not natively rectangles are converted via :meth:`Rectangle.convert`. Unconvertible
    regions (e.g. ``Special(SpecialCode.UNKNOWN)``) are represented by an empty rectangle so the index alignment with
    the frame sequence is preserved.

    :param sequence_dir: The sequence root directory.
    :type sequence_dir: str | Path

    :raises FileNotFoundError: If the groundtruth file is missing.
    :returns: One rectangle per groundtruth entry.
    :rtype: list[Rectangle]
    """
    gt_path = Path(sequence_dir) / GROUNDTRUTH_FILE
    if not gt_path.exists():
        raise FileNotFoundError("Groundtruth file not found: {}".format(gt_path))
    regions = read_trajectory(str(gt_path))
    rectangles: list[Rectangle] = []
    for region in regions:
        if isinstance(region, Rectangle):
            rectangles.append(region)
        else:
            try:
                rectangles.append(Rectangle.convert(region))
            except Exception:
                rectangles.append(Rectangle())
    return rectangles


@dataclass
class MovementStats:
    """Per-frame movement statistics for a sequence's groundtruth.

    :ivar time_seconds: Per-frame timestamps in seconds.
    :ivar centers_x: Bounding-box center X coordinates.
    :ivar centers_y: Bounding-box center Y coordinates.
    :ivar areas: Bounding-box areas in square pixels.
    :ivar widths: Bounding-box widths in pixels.
    :ivar heights: Bounding-box heights in pixels.
    :ivar velocity_x: Center X-velocity in pixels per second.
    :ivar velocity_y: Center Y-velocity in pixels per second.
    :ivar velocity_mag: Velocity magnitude in pixels per second.
    :ivar area_change_rate: Bounding-box area derivative in square pixels per second.
    """

    time_seconds: npt.NDArray
    centers_x: npt.NDArray
    centers_y: npt.NDArray
    areas: npt.NDArray
    widths: npt.NDArray
    heights: npt.NDArray
    velocity_x: npt.NDArray
    velocity_y: npt.NDArray
    velocity_mag: npt.NDArray
    area_change_rate: npt.NDArray

    def summary(self) -> dict[str, float]:
        """Returns high-level aggregates over the frames.

        :returns: A dict with ``duration_sec``, ``frames``, ``avg_velocity``, ``max_velocity``, ``avg_area`` and ``area_variance``.
        :rtype: dict[str, float]
        """
        if self.velocity_mag.size <= 1:
            avg_velocity = max_velocity = 0.0
        else:
            avg_velocity = float(np.mean(self.velocity_mag[1:]))
            max_velocity = float(np.max(self.velocity_mag[1:]))
        return {
            "duration_sec": float(self.time_seconds[-1] - self.time_seconds[0]),
            "frames": int(len(self.time_seconds)),
            "avg_velocity": avg_velocity,
            "max_velocity": max_velocity,
            "avg_area": float(np.mean(self.areas)),
            "area_variance": float(np.var(self.areas)),
        }


def compute_movement_stats(sequence_dir: str | Path,
                           frame_range: tuple[int, int] | None = None) -> MovementStats:
    """Computes per-frame velocity, area and center statistics for a sequence groundtruth.

    Velocities are computed as :func:`numpy.gradient` of the center coordinates with respect to time,
    using the sequence FPS.

    :param sequence_dir: The sequence root directory.
    :type sequence_dir: str | Path
    :param frame_range: Inclusive 0-based ``(start, end)`` bounds, or ``None`` for the entire sequence.
    :type frame_range: tuple[int, int] | None

    :raises RuntimeError: If no groundtruth rectangles are available.
    :returns: The per-frame statistics bundle.
    :rtype: MovementStats
    """
    sequence_dir = Path(sequence_dir)
    metadata = read_metadata(sequence_dir)
    fps = int(metadata.get("fps", 30))

    rectangles = _read_rectangles(sequence_dir)
    if frame_range is not None:
        start, end = frame_range
        rectangles = rectangles[start:end + 1]

    count = len(rectangles)
    if count == 0:
        raise RuntimeError("No groundtruth rectangles available in {}".format(sequence_dir))

    time = np.arange(count) / fps
    centers_x = np.array([r.x + r.width / 2 for r in rectangles])
    centers_y = np.array([r.y + r.height / 2 for r in rectangles])
    areas = np.array([r.width * r.height for r in rectangles])
    widths = np.array([r.width for r in rectangles])
    heights = np.array([r.height for r in rectangles])

    dt = 1.0 / fps
    if count >= 2:
        velocity_x = np.gradient(centers_x, dt)
        velocity_y = np.gradient(centers_y, dt)
        area_change_rate = np.gradient(areas, dt)
    else:
        velocity_x = np.zeros(count)
        velocity_y = np.zeros(count)
        area_change_rate = np.zeros(count)
    velocity_mag = np.sqrt(velocity_x ** 2 + velocity_y ** 2)

    return MovementStats(time_seconds=time, centers_x=centers_x, centers_y=centers_y,
                         areas=areas, widths=widths, heights=heights,
                         velocity_x=velocity_x, velocity_y=velocity_y,
                         velocity_mag=velocity_mag, area_change_rate=area_change_rate)


def find_size_range_windows(sequence_dir: str | Path, size_min: float, size_max: float,
                            target_frames: int, min_bbox_movements: float = 1.5,
                            check_initial_size_only: bool = False, stride: int = 5) -> list[dict]:
    """Finds fixed-length windows where the object diagonal lies in a given size range with good motion.

    Candidates are ranked so windows requiring no speed-up come first, then by descending quality score.
    Each candidate dict contains ``start_frame``, ``end_frame``, ``bbox_movements``, ``avg_velocity``,
    ``avg_size``, ``initial_size``, ``total_distance``, ``quality_score``, ``speed_up_needed`` and
    ``meets_criteria``.

    :param sequence_dir: The sequence root directory.
    :type sequence_dir: str | Path
    :param size_min: Minimum allowed bbox diagonal in pixels.
    :type size_min: float
    :param size_max: Maximum allowed bbox diagonal in pixels.
    :type size_max: float
    :param target_frames: Window length in frames.
    :type target_frames: int
    :param min_bbox_movements: Minimum total displacement expressed as multiples of the bbox size. Defaults to 1.5.
    :type min_bbox_movements: float
    :param check_initial_size_only: If True, only the first frame's size is required to be in range. Defaults to False.
    :type check_initial_size_only: bool
    :param stride: Step in frames between candidate window starts. Defaults to 5.
    :type stride: int

    :returns: Ranked candidate windows.
    :rtype: list[dict]
    """
    sequence_dir = Path(sequence_dir)
    rectangles = _read_rectangles(sequence_dir)
    sizes = np.array([np.sqrt(r.width ** 2 + r.height ** 2) for r in rectangles])
    count = len(rectangles)

    candidates: list[dict] = []
    for start_idx in range(0, count - target_frames + 1, stride):
        end_idx = start_idx + target_frames - 1
        if check_initial_size_only:
            if not (size_min <= sizes[start_idx] <= size_max):
                continue
        else:
            window_sizes = sizes[start_idx:end_idx + 1]
            if not np.all((window_sizes >= size_min) & (window_sizes <= size_max)):
                continue

        stats = compute_movement_stats(sequence_dir, frame_range=(start_idx, end_idx))
        total_distance = float(np.sum(np.sqrt(np.diff(stats.centers_x) ** 2 + np.diff(stats.centers_y) ** 2)))
        avg_bbox_diagonal = float(np.mean(np.sqrt(stats.widths ** 2 + stats.heights ** 2)))
        bbox_movements = total_distance / avg_bbox_diagonal if avg_bbox_diagonal > 0 else 0.0
        avg_velocity = float(np.mean(stats.velocity_mag[1:])) if stats.velocity_mag.size > 1 else 0.0

        speed_up = 1
        if 0 < bbox_movements < min_bbox_movements:
            speed_up = max(1, int(np.ceil(min_bbox_movements / bbox_movements)))

        candidates.append({
            "start_frame": start_idx,
            "end_frame": end_idx,
            "bbox_movements": bbox_movements,
            "avg_velocity": avg_velocity,
            "avg_size": avg_bbox_diagonal,
            "initial_size": float(sizes[start_idx]),
            "total_distance": total_distance,
            "quality_score": bbox_movements * avg_velocity / 10,
            "speed_up_needed": speed_up,
            "meets_criteria": bbox_movements >= min_bbox_movements,
        })

    candidates.sort(key=lambda x: (x["speed_up_needed"], -x["quality_score"]))
    return candidates


def analyze_size_availability(sequence_dir: str | Path,
                              size_ranges: TypingSequence[tuple[float, float, str]],
                              target_frames: int, min_bbox_movements: float = 1.5,
                              check_initial_size_only: bool = False) -> dict[str, dict]:
    """Summarizes how many candidate windows match each ``(min, max, label)`` size range.

    Useful for tuning size and movement thresholds before generating slices.

    :param sequence_dir: The sequence root directory.
    :type sequence_dir: str | Path
    :param size_ranges: Sequence of ``(size_min, size_max, label)`` tuples to check.
    :type size_ranges: TypingSequence[tuple[float, float, str]]
    :param target_frames: Window length in frames.
    :type target_frames: int
    :param min_bbox_movements: Minimum movement quality threshold. Defaults to 1.5.
    :type min_bbox_movements: float
    :param check_initial_size_only: Whether to check only the first frame's size. Defaults to False.
    :type check_initial_size_only: bool

    :returns: A per-label dict with candidate counts, best quality and minimum speed-up required.
    :rtype: dict[str, dict]
    """
    results: dict[str, dict] = {}
    for size_min, size_max, label in size_ranges:
        candidates = find_size_range_windows(sequence_dir, size_min, size_max, target_frames,
                                             min_bbox_movements, check_initial_size_only)
        good_no_speedup = [c for c in candidates if c["meets_criteria"] and c["speed_up_needed"] == 1]
        results[label] = {
            "total_candidates": len(candidates),
            "good_without_speedup": len(good_no_speedup),
            "best_quality": max((c["quality_score"] for c in candidates), default=0),
            "min_speedup_needed": min((c["speed_up_needed"] for c in candidates), default=None),
        }
    return results


def verify_slice(slice_dir: str | Path, size_min: float, size_max: float,
                 check_initial_size_only: bool) -> dict[str, float]:
    """Computes summary metrics for a generated slice and checks whether it stayed in the size range.

    :param slice_dir: The slice directory to inspect.
    :type slice_dir: str | Path
    :param size_min: Lower bound used to generate the slice.
    :type size_min: float
    :param size_max: Upper bound used to generate the slice.
    :type size_max: float
    :param check_initial_size_only: Match the policy used during slicing.
    :type check_initial_size_only: bool

    :returns: A dict of metrics including ``length``, ``size_min``, ``size_max``, ``size_avg``, ``size_at_init``, ``bbox_movements``, ``avg_velocity`` and ``size_in_range``.
    :rtype: dict[str, float]
    """
    rectangles = _read_rectangles(slice_dir)
    sizes = np.array([np.sqrt(r.width ** 2 + r.height ** 2) for r in rectangles])
    stats = compute_movement_stats(slice_dir)
    total_distance = float(np.sum(np.sqrt(np.diff(stats.centers_x) ** 2 + np.diff(stats.centers_y) ** 2)))
    avg = float(np.mean(sizes))
    bbox_movements = total_distance / avg if avg > 0 else 0.0
    avg_velocity = float(np.mean(stats.velocity_mag[1:])) if stats.velocity_mag.size > 1 else 0.0
    initial_size = float(sizes[0])
    if check_initial_size_only:
        size_ok = size_min <= initial_size <= size_max
    else:
        size_ok = bool(np.min(sizes) >= size_min and np.max(sizes) <= size_max)
    return {
        "length": int(len(rectangles)),
        "size_min": float(np.min(sizes)),
        "size_max": float(np.max(sizes)),
        "size_avg": avg,
        "size_at_init": initial_size,
        "bbox_movements": bbox_movements,
        "avg_velocity": avg_velocity,
        "size_in_range": size_ok,
    }
