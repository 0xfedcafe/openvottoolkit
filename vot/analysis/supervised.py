"""This module constans common analysis routines for supervised experiment, e.g.
Accuracy-Robustness and EAO as defined in VOT papers."""

import math
from typing import Sequence as TSequence, Any

import numpy as np
import numpy.typing as npt

from attributee import Integer, Boolean, Float, Include

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import SupervisedExperiment
from vot.region import Region, SpecialCode, calculate_overlaps
from vot.analysis import MissingResultsException, Measure, Result, Point, is_special, Plot, Analysis, \
    Sorting, SeparableAnalysis, SequenceAggregator, TrackerSeparableAnalysis, Axes
from vot.utilities.data import Grid

def compute_accuracy(trajectory: list[Region], sequence: Sequence, burnin: int = 10,
    ignore_unknown: bool = True, bounded: bool = True) -> tuple[float, int]:
    """Computes accuracy of a tracker on a given sequence. Accuracy is defined as mean
    overlap of the tracker region with the groundtruth region. The overlap is computed
    only for frames where the tracker is not in initialization or failure state. The
    overlap is computed only for frames after the burnin period.

    :param trajectory: Tracker trajectory.
    :type trajectory: list[Region]
    :param sequence: Sequence to compute accuracy on.
    :type sequence: Sequence
    :param burnin: Burnin period. Defaults to 10.
    :type burnin: int, optional
    :param ignore_unknown: Ignore unknown regions. Defaults to True.
    :type ignore_unknown: bool, optional
    :param bounded: Clip overlaps to the image bounds (``sequence.size``). Defaults to True.
    :type bounded: bool, optional

    :returns: Mean overlap (accuracy) and the number of frames it was averaged over.
    :rtype: tuple[float, int]"""

    groundtruth = sequence.groundtruth()
    if groundtruth is None:
        return 0.0, 0
    overlaps = np.asarray(calculate_overlaps(trajectory, groundtruth, sequence.size if bounded else None), dtype=np.float64)
    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, SpecialCode.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, SpecialCode.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, SpecialCode.FAILURE) or is_special(region, SpecialCode.CRASH):
            mask[i] = False

    if any(mask):
        return float(np.mean(overlaps[mask])), int(np.sum(mask))
    return 0.0, 0

def compute_eao_curve(overlaps: list[list[float]], weights: list[float], success: list[bool]) -> npt.NDArray:
    """Computes EAO curve from a list of overlaps, weights and success flags.

    :param overlaps: Per-run overlap sequences.
    :type overlaps: list[list[float]]
    :param weights: Weight of each run.
    :type weights: list[float]
    :param success: Per-run flag, False if the tracker failed during the run.
    :type success: list[bool]

    :returns: Expected average overlap at each frame.
    :rtype: npt.NDArray"""
    max_length = max([len(el) for el in overlaps])
    total_runs = len(overlaps)
    
    overlaps_array = np.zeros((total_runs, max_length), dtype=np.float32)
    mask_array = np.zeros((total_runs, max_length), dtype=np.float32)  # mask out frames which are not considered in EAO calculation
    weights_vector = np.reshape(np.array(weights, dtype=np.float32), (len(weights), 1))  # weight of each run

    for i, (o, succeeded) in enumerate(zip(overlaps, success)):
        overlaps_array[i, :len(o)] = np.array(o)
        if not succeeded:
            # tracker has failed during this run - fill zeros until the end of the run
            mask_array[i, :] = 1
        else:
            # tracker has successfully tracked to the end - consider only this part of the sequence
            mask_array[i, :len(o)] = 1

    overlaps_array_sum = overlaps_array.copy()
    for j in range(1, overlaps_array_sum.shape[1]):
        overlaps_array_sum[:, j] = np.mean(overlaps_array[:, 1:j+1], axis=1)
    
    return np.sum(weights_vector * overlaps_array_sum * mask_array, axis=0) / np.sum(mask_array * weights_vector, axis=0)
    
class AccuracyRobustness(SeparableAnalysis):
    """Accuracy-Robustness analysis.

    Computes accuracy and robustness of a tracker on a given sequence. Accuracy is
    defined as mean overlap of the tracker region with the groundtruth region. The
    overlap is computed only for frames where the tracker is not in initialization or
    failure state. The overlap is computed only for frames after the burnin period.
    Robustness is defined as a number of failures divided by the total number of frames.
    """

    sensitivity = Float(default=30, val_min=1)
    burnin = Integer(default=10, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    @property
    def _title_default(self) -> str:
        """Returns title of the analysis."""
        return "AR analysis"

    def describe(self) -> tuple[Result | None, ...]:
        """Returns description of the analysis."""
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.ASCENDING), \
             Measure("Crashes", "C", minimal=0, direction=Sorting.ASCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None

    def compatible(self, experiment: Experiment) -> bool:
        """Returns True if the analysis is compatible with the experiment.

        Only SupervisedExperiment is compatible.
        """
        return isinstance(experiment, SupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes accuracy, tracking-failure count and crash count for a tracker
        on a given sequence. ``Robustness`` reports the tracking-failure count
        (VOT semantics); ``Crashes`` reports the count of tracker process failures.
        The AR exp-decay penalizes the *combined* incident rate so that crashing
        trackers don't hide behind unchanged Robustness.

        :returns: ``(accuracy, failures, crashes, ar, frames)``.
        """
        assert isinstance(experiment, SupervisedExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        accuracy: float = 0.0
        failures: float = 0.0
        crashes: float = 0.0
        for trajectory in trajectories:
            failures += len(trajectory.failures())
            crashes += len(trajectory.crashes())
            accuracy += compute_accuracy(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)[0]

        failures /= len(trajectories)
        crashes /= len(trajectories)
        accuracy /= len(trajectories)

        ar = (math.exp(- ((failures + crashes) / len(sequence)) * self.sensitivity), accuracy)

        return accuracy, failures, crashes, ar, len(sequence)

class AverageAccuracyRobustness(SequenceAggregator):
    """Average accuracy-robustness analysis. Computes average accuracy and robustness of
    a tracker on a given sequence.

    Accuracy is defined as mean overlap of the tracker region with the groundtruth
    region. The overlap is computed only for frames where the tracker is not in
    initialization or failure state. The overlap is computed only for frames after the
    burnin period. Robustness is defined as a number of failures divided by the total
    number of frames. The analysis is computed as an average of accuracy and robustness
    over all sequences.
    """

    analysis = Include(AccuracyRobustness)

    @property
    def _title_default(self) -> str:
        """Returns title of the analysis."""
        return "AR Analysis"

    def dependencies(self) -> TSequence[Analysis]:
        """Returns dependencies of the analysis."""
        return (self.analysis,)

    def describe(self) -> tuple[Result | None, ...]:
        """Returns description of the analysis."""
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.ASCENDING), \
             Measure("Crashes", "C", minimal=0, direction=Sorting.ASCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar"), \
             None

    def compatible(self, experiment: Experiment) -> bool:
        """Returns True if the analysis is compatible with the experiment.

        Only SupervisedExperiment is compatible.
        """
        return isinstance(experiment, SupervisedExperiment)

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregates per-sequence accuracy / failure / crash counts into the
        weighted mean across sequences (frames as the natural weight).

        :returns: ``(accuracy, failures, crashes, ar, length)``.
        """

        failures: float = 0.0
        crashes: float = 0.0
        accuracy: float = 0.0
        weight_total: float = 0.0

        for a, f, c, _, w in results:
            failures += f * w
            crashes += c * w
            accuracy += a * w
            weight_total += w

        failures /= weight_total
        crashes /= weight_total
        accuracy /= weight_total
        length = weight_total / len(results)

        ar = (math.exp(- ((failures + crashes) / length) * self.analysis.sensitivity), accuracy)

        return accuracy, failures, crashes, ar, length

class EAOCurve(TrackerSeparableAnalysis):
    """Expected Average Overlap curve analysis.

    Computes expected average overlap of a tracker on a given sequence. The overlap is
    computed only for frames where the tracker is not in initialization or failure
    state. The overlap is computed only for frames after the burnin period. The analysis
    is computed as an average of accuracy and robustness over all sequences.
    """

    burnin = Integer(default=10, val_min=0)
    bounded = Boolean(default=True)

    @property
    def _title_default(self) -> str:
        """Returns title of the analysis."""
        return "EAO Curve"

    def describe(self) -> tuple[Result | None, ...]:
        """Returns description of the analysis."""
        return Plot("Expected Average Overlap", "EAO", minimal=0, maximal=1, trait="eao"),

    def compatible(self, experiment: Experiment) -> bool:
        """Returns True if the analysis is compatible with the experiment.

        Only SupervisedExperiment is compatible.
        """
        return isinstance(experiment, SupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: list[Sequence], dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes expected average overlap of a tracker over the supplied sequences.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param tracker: Tracker.
        :type tracker: Tracker
        :param sequence: List of sequences (this analysis is separable on tracker only,
            so each part receives the entire sequence list).
        :type sequence: list[Sequence]
        :param dependencies: Dependencies.
        :type dependencies: list[Grid]

        :returns: Expected average overlap curve.
        :rtype: tuple[Any, ...]"""

        assert isinstance(experiment, SupervisedExperiment)

        overlaps_all: list[list[float]] = []
        weights_all: list[float] = []
        success_all: list[bool] = []

        for one_sequence in sequence:

            trajectories = experiment.gather(tracker, one_sequence)

            if len(trajectories) == 0:
                raise MissingResultsException()

            groundtruth = one_sequence.groundtruth()
            if groundtruth is None:
                raise MissingResultsException(f"Missing groundtruth for sequence {one_sequence.name}")

            for trajectory in trajectories:

                overlaps = calculate_overlaps(trajectory.regions(), groundtruth, one_sequence.size if self.bounded else None)
                init_idxs, fail_idxs, crash_idxs = trajectory.markers()
                # Both FAILURE and CRASH terminate a tracking run for EAO
                # purposes: FAILURE is a tracker-reported loss of target,
                # CRASH is a tracker process failure — either way the run
                # ended before the tracker produced more output. An orphan
                # marker (no preceding open init) becomes a length-1 failed
                # run so the incident contributes to the curve exactly the
                # way an ``INIT@F + FAILURE@F+1`` pair would.
                terminator_idxs = sorted(fail_idxs + crash_idxs)

                if not init_idxs:
                    # Nothing was ever initialized successfully; treat the trajectory
                    # as a single failed run so it still contributes to the curve.
                    overlaps_all.append(overlaps)
                    success_all.append(False)
                    weights_all.append(1.0)
                    continue

                term_pos = 0
                for start in init_idxs:
                    # Orphan terminators before this init each become a length-1
                    # failed run capturing the incident.
                    while term_pos < len(terminator_idxs) and terminator_idxs[term_pos] < start:
                        t = terminator_idxs[term_pos]
                        overlaps_all.append(overlaps[t:t + 1])
                        success_all.append(False)
                        weights_all.append(1.0)
                        term_pos += 1
                    if term_pos < len(terminator_idxs):
                        overlaps_all.append(overlaps[start:terminator_idxs[term_pos]])
                        success_all.append(False)
                        weights_all.append(1.0)
                        term_pos += 1
                    else:
                        # tracker was initialized, but it has not failed until the end of the sequence
                        overlaps_all.append(overlaps[start:])
                        success_all.append(True)
                        weights_all.append(1.0)

                # Trailing orphan terminators after the last init also contribute
                # one length-1 failed run each.
                while term_pos < len(terminator_idxs):
                    t = terminator_idxs[term_pos]
                    overlaps_all.append(overlaps[t:t + 1])
                    success_all.append(False)
                    weights_all.append(1.0)
                    term_pos += 1

        return compute_eao_curve(overlaps_all, weights_all, success_all),

class EAOScore(Analysis):
    """Expected Average Overlap score analysis.

    The analysis is computed as an average of EAO scores over multiple sequences.
    """

    eaocurve = Include(EAOCurve)
    low = Integer()
    high = Integer()

    @property
    def _title_default(self) -> str:
        """Returns title of the analysis."""
        return "EAO analysis"

    def describe(self) -> tuple[Result | None, ...]:
        """Returns description of the analysis."""
        return Measure("Expected average overlap", "EAO", 0, 1, Sorting.DESCENDING),

    def compatible(self, experiment: Experiment) -> bool:
        """Returns True if the analysis is compatible with the experiment.

        Only SupervisedExperiment is compatible.
        """
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Returns dependencies of the analysis."""
        return (self.eaocurve,)

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """Computes expected average overlap of a tracker on a given sequence.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param trackers: List of trackers.
        :type trackers: list[Tracker]
        :param sequences: List of sequences.
        :type sequences: list[Sequence]
        :param dependencies: Dependencies.
        :type dependencies: list[Grid]

        :returns: Expected average overlap.
        :rtype: Grid"""
        return dependencies[0].foreach(lambda x, i, j: (float(np.mean(x[0][self.low:self.high + 1])), ))

    @property
    def axes(self) -> Axes:
        """Returns axes of the analysis."""
        return Axes.TRACKERS