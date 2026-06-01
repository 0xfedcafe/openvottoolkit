"""Failure- and crash-count analyses for supervised experiments.

``FailureCount`` reports tracker-reported tracking failures (low-overlap loss);
``CrashCount`` reports tracker process failures (exceptions / timeouts that the
runtime recovered from); ``SequenceFailureCurve`` reports the running cumulative
crash / robustness / total counts per frame. All consume the FAILURE / CRASH
:class:`vot.region.Special` markers written by
:class:`vot.experiment.multirun.SupervisedExperiment`, via the
:class:`vot.tracker.results.Trajectory` marker metrics.
"""

from typing import Sequence as TSequence, Any

import numpy as np

from attributee import Include

from vot.analysis import (Analysis, Measure, Plot, Result,
                          MissingResultsException,
                          SequenceAggregator, Sorting,
                          SeparableAnalysis)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (SupervisedExperiment)
from vot.tracker import Tracker
from vot.utilities.data import Grid


class FailureCount(SeparableAnalysis):
    """Count the number of failures in a sequence.

    A failure is a region annotated with ``SpecialCode.FAILURE`` by the experiment.
    """

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    @property
    def _title_default(self) -> str:
        """Default title for the analysis."""
        return "Number of failures"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single sequence."""

        assert isinstance(experiment, SupervisedExperiment)

        objects = sequence.objects()
        objects_failures: float = 0.0

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            failures = 0
            for trajectory in trajectories:
                failures = failures + len(trajectory.failures())
            objects_failures += failures / len(trajectories)

        return objects_failures / len(objects), len(sequence)

class CumulativeFailureCount(SequenceAggregator):
    """Count the number of failures over all sequences.

    A failure is a region annotated with ``SpecialCode.FAILURE`` by the experiment.
    """

    analysis = Include(FailureCount)

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Return the dependencies of the analysis."""
        return (self.analysis,)

    @property
    def _title_default(self) -> str:
        """Default title for the analysis."""
        return "Number of failures"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING),

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregate the analysis for a list of sequences. The aggregation is done by
        summing the number of failures for each sequence.

        :param tracker: The tracker the results belong to.
        :type tracker: Tracker
        :param sequences: The list of sequences to aggregate.
        :type sequences: list[Sequence]
        :param results: The results of the analysis for each sequence.
        :type results: Grid

        :returns: The aggregated failure count.
        :rtype: tuple[Any, ...]"""

        failures = 0

        for a in results:
            failures = failures + a[0]

        return failures,


class CrashCount(SeparableAnalysis):
    """Count the number of tracker process crashes in a sequence.

    A crash is a region annotated with ``SpecialCode.CRASH`` by
    :class:`vot.experiment.multirun.SupervisedExperiment` when ``runtime``
    raised a :class:`TrackerException` during ``initialize`` or ``update``.
    """

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    @property
    def _title_default(self) -> str:
        """Default title for the analysis."""
        return "Number of crashes"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure("Crashes", "C", 0, None, Sorting.ASCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single sequence."""

        assert isinstance(experiment, SupervisedExperiment)

        objects = sequence.objects()
        objects_crashes: float = 0.0

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            crashes = 0
            for trajectory in trajectories:
                crashes = crashes + len(trajectory.crashes())
            objects_crashes += crashes / len(trajectories)

        return objects_crashes / len(objects), len(sequence)


class CumulativeCrashCount(SequenceAggregator):
    """Count the number of crashes over all sequences.

    A crash is a region annotated with ``SpecialCode.CRASH`` by the experiment.
    """

    analysis = Include(CrashCount)

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Return the dependencies of the analysis."""
        return (self.analysis,)

    @property
    def _title_default(self) -> str:
        """Default title for the analysis."""
        return "Number of crashes"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure("Crashes", "C", 0, None, Sorting.ASCENDING),

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Sum the per-sequence crash counts.

        :returns: ``(total_crashes,)``."""

        crashes = 0

        for a in results:
            crashes = crashes + a[0]

        return crashes,


class SequenceFailureCurve(SeparableAnalysis):
    """Per-frame cumulative crash / robustness / total curves for a sequence.

    For each (tracker, sequence) pair this emits three curves indexed by frame: the
    running count of crashes, the running count of tracking failures (robustness) and
    their sum (total). The curve endpoints match :class:`CrashCount` / :class:`FailureCount`,
    and ``total`` is the combined incident count that drives the AR exp-decay. The
    incident frames come from the :class:`vot.tracker.results.Trajectory` marker metrics,
    averaged over repetitions and objects like :class:`FailureCount`.
    """

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, SupervisedExperiment)

    @property
    def _title_default(self) -> str:
        """Default title for the analysis."""
        return "Cumulative failure curves"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Plot("Cumulative crashes", "C", wrt="frame", minimal=0), \
            Plot("Cumulative robustness", "R", wrt="frame", minimal=0), \
            Plot("Cumulative total", "T", wrt="frame", minimal=0), \
            None

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the cumulative crash / robustness / total curves for a single sequence."""

        assert isinstance(experiment, SupervisedExperiment)

        objects = sequence.objects()
        length = len(sequence)
        failure_increments = np.zeros(length, dtype=float)
        crash_increments = np.zeros(length, dtype=float)

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_failures = np.zeros(length, dtype=float)
            object_crashes = np.zeros(length, dtype=float)
            for trajectory in trajectories:
                for frame in trajectory.failures():
                    if frame < length:
                        object_failures[frame] += 1
                for frame in trajectory.crashes():
                    if frame < length:
                        object_crashes[frame] += 1
            failure_increments += object_failures / len(trajectories)
            crash_increments += object_crashes / len(trajectories)

        failure_increments /= len(objects)
        crash_increments /= len(objects)

        cumulative_crashes = np.cumsum(crash_increments)
        cumulative_robustness = np.cumsum(failure_increments)
        cumulative_total = cumulative_crashes + cumulative_robustness

        return cumulative_crashes.tolist(), cumulative_robustness.tolist(), cumulative_total.tolist(), length
