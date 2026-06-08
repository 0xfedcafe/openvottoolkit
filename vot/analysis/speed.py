"""Speed / FPS analyses.

These analyses consume per-frame timing recorded by the experiments (the trajectory ``time``
property, written out as ``<results>_time.value`` on disk) and produce per-tracker, per-sequence and
aggregated speed metrics. The results plug into the standard report pipeline so a stack that includes
``- type: average_speed`` (or ``sequence_speed``) will show FPS measures in the overview table and a
per-frame FPS curve in the plots section without any further wiring.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from attributee import Integer

from vot.analysis import (
    Analysis,
    Measure,
    MissingResultsException,
    Plot,
    Result,
    SeparableAnalysis,
    SequenceAggregator,
    Sorting,
)
from vot.experiment import Experiment
from vot.experiment.multirun import MultiRunExperiment
from vot.experiment.multistart import MultiStartExperiment
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.utilities.data import Grid
from vot.tracker.results import Trajectory

# Experiment types that expose a ``gather`` method returning full-length trajectories with
# per-frame ``time`` properties. ``MultiStartExperiment`` remaps its anchor runs into
# sequence coordinates in ``gather``, so it can be analysed the same way as a multi-run.
_SPEED_EXPERIMENTS = (MultiRunExperiment, MultiStartExperiment)


@dataclass(frozen=True)
class SpeedMetrics:
    """Average frame rate and frame time for a tracker run.

    ``frames`` is the number of frames the averages cover; it weights the metrics when
    they are combined across sequences. A run with no usable timings is all zeros."""

    fps: float = 0.0
    time_ms: float = 0.0
    frames: int = 0

    @classmethod
    def from_sequence_result(cls, result: tuple[Any, ...]) -> "SpeedMetrics":
        """Reads a :meth:`SequenceSpeed.subcompute` result tuple
        ``(fps, time_ms, per_frame_fps, frames)``, discarding the per-frame curve."""
        fps, time_ms, _, frames = result
        return cls(fps, time_ms, frames)

    @staticmethod
    def averaged(metrics: list["SpeedMetrics"]) -> "SpeedMetrics":
        """Frame-count-weighted mean of several metrics, zeroed when no frames contribute."""
        weight = sum(metric.frames for metric in metrics)
        if weight <= 0 or math.isnan(weight):
            return SpeedMetrics()
        fps = sum(metric.fps * metric.frames for metric in metrics) / weight
        time_ms = sum(metric.time_ms * metric.frames for metric in metrics) / weight
        return SpeedMetrics(fps, time_ms, weight)


def _per_frame_times(trajectory: Trajectory) -> list[tuple[int, float]]:
    """Returns ``(frame_index, elapsed_seconds)`` pairs for every frame whose recorded
    ``time`` property is positive.

    The frame index is preserved (rather than collapsed to a position in a packed list)
    so downstream plots can place each sample at its true x position. For a realtime
    experiment most frames have ``time == 0`` (the tracker was skipped and the previous
    status replayed); those frames are dropped here but the surviving entries keep
    their absolute indices, so the FPS curve plots e.g. frames 0, 1, 2, 3, 4, 64, 129
    instead of collapsing to indices 0..6.

    :param trajectory: The tracker trajectory to inspect.
    :returns: A list of ``(frame_index, elapsed_seconds)`` for every frame with a usable timing.
    """
    pairs: list[tuple[int, float]] = []
    for index in range(len(trajectory)):
        # ``Trajectory.properties(int)`` returns a per-frame dict (the ``None``-arg overload returns a
        # tuple of property names; the runtime ``isinstance`` narrows the union for the type checker).
        properties = trajectory.properties(index)
        if not isinstance(properties, dict):
            continue
        value = properties.get("time")
        if value is None:
            continue
        try:
            elapsed = float(value)
        except (TypeError, ValueError):
            continue
        if elapsed > 0:
            pairs.append((index, elapsed))
    return pairs


def compute_speed(trajectory: Trajectory, skip_initial: int = 5) -> tuple[SpeedMetrics, list[tuple[int, float]]]:
    """Computes average FPS, average frame time and per-frame FPS for a single trajectory.

    ``skip_initial`` is applied to the *average* only — the per-frame FPS curve preserves
    every positive-time sample so the plot can show e.g. a sparse realtime run's real
    invocations at their true frame indices. The skip is counted in surviving samples
    (not raw frames) because warmup happens on the first few real invocations, and a
    realtime run can have huge gaps between them.

    :param trajectory: The tracker trajectory to inspect.
    :param skip_initial: Number of leading positive-time samples to discard before averaging. Defaults to 5.

    :returns: The averaged :class:`SpeedMetrics` and the per-frame FPS curve, a list of
        ``(frame_index, fps)`` pairs (empty when no usable timings exist).
    :rtype: tuple[SpeedMetrics, list[tuple[int, float]]]
    """
    pairs = _per_frame_times(trajectory)
    if not pairs:
        return SpeedMetrics(), []

    per_frame_fps = [(frame, 1.0 / elapsed) for frame, elapsed in pairs]

    tail = pairs[skip_initial:] if len(pairs) > skip_initial else pairs
    times_array = np.asarray([t for _, t in tail], dtype=float)
    fps = float(np.mean(1.0 / times_array))
    time_ms = float(np.mean(times_array) * 1000.0)
    return SpeedMetrics(fps, time_ms, len(tail)), per_frame_fps


class SequenceSpeed(SeparableAnalysis):
    """Per-tracker, per-sequence speed analysis.

    For each (tracker, sequence) pair this analysis reads the per-frame ``time`` property from the
    trajectory and emits the tracker's average FPS, average frame time (in milliseconds) and a
    per-frame FPS curve suitable for plotting via :class:`vot.report.LinePlot`.
    """

    skip_initial = Integer(default=5, val_min=0,
                           description="Number of leading frames excluded from the average (init / warmup).")

    @property
    def _title_default(self) -> str:
        """Returns the title of the analysis used in reports."""
        return "Speed"

    def describe(self) -> tuple[Result | None, ...]:
        """Returns the result descriptors emitted by :meth:`subcompute`."""
        return Measure("Average FPS", "FPS", minimal=0, direction=Sorting.DESCENDING), \
            Measure("Average frame time (ms)", "ms", minimal=0, direction=Sorting.ASCENDING), \
            Plot("Per-frame FPS", "FPS", wrt="frame", minimal=0, trait="fps"), \
            None

    def compatible(self, experiment: Experiment) -> bool:
        """Returns True for any experiment that records per-frame timings.

        ``MultiRunExperiment`` covers supervised, unsupervised and realtime runs and
        ``MultiStartExperiment`` covers anchor-based runs; each of them writes
        ``Trajectory.properties[\"time\"]``.
        """
        return isinstance(experiment, _SPEED_EXPERIMENTS)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence,
                   dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes speed metrics for a single tracker / sequence pair.

        :param experiment: The experiment context (a :class:`MultiRunExperiment` or :class:`MultiStartExperiment`).
        :type experiment: Experiment
        :param tracker: Tracker being analysed.
        :type tracker: Tracker
        :param sequence: Sequence being analysed.
        :type sequence: Sequence
        :param dependencies: Dependencies of the analysis (unused).
        :type dependencies: list[Grid]

        :raises MissingResultsException: When no trajectory files exist for the pair.
        :returns: ``(average_fps, average_time_ms, per_frame_fps, frame_count)``.
        :rtype: tuple[Any, ...]
        """
        assert isinstance(experiment, _SPEED_EXPERIMENTS)

        # ``gather`` may return ``None`` placeholders when called with ``pad=True``; speed
        # analysis is unpadded but the return type permits it, so drop any nones here.
        trajectories = [t for t in experiment.gather(tracker, sequence) if t is not None]
        if len(trajectories) == 0:
            raise MissingResultsException()

        # Merge per-frame samples by absolute frame index across trajectories. Multistart
        # anchor runs cover disjoint slices of the sequence — averaging them along the
        # packed-array index (slot 0 vs slot 0) compared frame 0 against the anchor's
        # frame, which was meaningless. Indexing by absolute frame keeps each sample at
        # its real position; if two trajectories happen to cover the same frame (e.g.
        # multirun repetitions), their FPS values are averaged for that frame.
        merged: dict[int, list[float]] = {}
        runs: list[SpeedMetrics] = []

        for trajectory in trajectories:
            metrics, curve = compute_speed(trajectory, self.skip_initial)
            runs.append(metrics)
            for frame, fps in curve:
                merged.setdefault(frame, []).append(fps)

        # Average across repetitions, weighting each run equally.
        average_fps = sum(run.fps for run in runs) / len(runs)
        average_time_ms = sum(run.time_ms for run in runs) / len(runs)

        per_frame_fps = sorted((frame, sum(values) / len(values)) for frame, values in merged.items())

        return average_fps, average_time_ms, per_frame_fps, len(sequence)


class AverageSpeed(SequenceAggregator):
    """Aggregates :class:`SequenceSpeed` results into a single per-tracker FPS / time pair.

    The averages are weighted by sequence length so longer sequences contribute proportionally more.
    """

    analysis = SequenceSpeed()

    @property
    def _title_default(self) -> str:
        """Returns the title of the analysis used in reports."""
        return "Average Speed"

    def dependencies(self) -> tuple[Analysis, ...]:
        """Declares the per-sequence dependency analysis."""
        return (self.analysis,)

    def describe(self) -> tuple[Result | None, ...]:
        """Returns the aggregated result descriptors."""
        return Measure("Average FPS", "FPS", minimal=0, direction=Sorting.DESCENDING), \
            Measure("Average frame time (ms)", "ms", minimal=0, direction=Sorting.ASCENDING), \
            None

    def compatible(self, experiment: Experiment) -> bool:
        """Mirrors :meth:`SequenceSpeed.compatible`."""
        return isinstance(experiment, _SPEED_EXPERIMENTS)

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregates per-sequence speed measurements into a single tracker-level pair.

        :param tracker: Tracker whose results are being aggregated.
        :type tracker: Tracker
        :param sequences: List of sequences in the aggregation.
        :type sequences: list[Sequence]
        :param results: A grid row holding one :class:`SequenceSpeed` tuple per sequence.
        :type results: Grid

        :returns: ``(average_fps, average_time_ms, None)``. The third slot is reserved by the descriptor tuple.
        :rtype: tuple[Any, ...]
        """
        metrics = [SpeedMetrics.from_sequence_result(entry) for entry in results if entry is not None]
        combined = SpeedMetrics.averaged(metrics)
        return combined.fps, combined.time_ms, None
