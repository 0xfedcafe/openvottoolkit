"""This module contains the implementation of the long term tracking performance
measures."""
import math
import numpy as np
from typing import Iterable, Sequence as TSequence, Any
from dataclasses import dataclass
import itertools

from attributee import Float, Integer, Boolean, Include, String

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.region import Region, SpecialCode, calculate_overlaps
from vot.region.raster import Bounds
from vot.experiment import Experiment
from vot.experiment.multirun import UnsupervisedExperiment, MultiRunExperiment
from vot.analysis import SequenceAggregator, Analysis, SeparableAnalysis, \
    MissingResultsException, Measure, Result, Sorting, Curve, Plot, \
    Axes, Point, is_special, filter_by_mask
from vot.utilities.data import Grid

def determine_thresholds(scores: Iterable[float], resolution: int) -> list[float]:
    """Determine thresholds for a given set of scores and a resolution. The thresholds
    are determined by sorting the scores and selecting the thresholds that divide the
    sorted scores into equal sized bins.

    :param scores: Scores to determine thresholds for.
    :type scores: Iterable[float]
    :param resolution: Number of thresholds to determine.
    :type resolution: int

    :returns: List of thresholds.
    :rtype: list[float]"""
    scores = [score for score in scores if not math.isnan(score)] #and not score is None]
    scores = sorted(scores, reverse=True)

    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(np.linspace(delta, len(scores) - delta, num=resolution - 2)).astype(int)
        thresholds = [scores[idx] for idx in idxs]
    else:
        thresholds = scores

    thresholds.insert(0, math.inf)
    thresholds.insert(len(thresholds), -math.inf)

    return thresholds

def compute_tpr_curves(trajectory: list[Region], confidence: list[float], sequence: Sequence, thresholds: list[float],
    ignore_unknown: bool = True, bounded: bool = True,
    ignore_masks: list[Region] | None = None) -> tuple[list[float], list[float]]:
    """Compute the TPR curves for a given trajectory and confidence scores.

    :param trajectory: Trajectory to compute the TPR curves for.
    :type trajectory: list[Region]
    :param confidence: Confidence scores for the trajectory.
    :type confidence: list[float]
    :param sequence: Sequence to compute the TPR curves for.
    :type sequence: Sequence
    :param thresholds: Thresholds to compute the TPR curves for.
    :type thresholds: list[float]
    :param ignore_unknown: Ignore unknown regions. Defaults to True.
    :type ignore_unknown: bool, optional
    :param bounded: Bounded evaluation. Defaults to True.
    :type bounded: bool, optional
    :param ignore_masks: Ignore masks. Defaults to None.
    :type ignore_masks: list[Region], optional

    :returns: Precision and recall arrays, one entry per threshold.
    :rtype: tuple[list[float], list[float]]"""
    assert len(trajectory) == len(confidence), "Trajectory and confidence must have the same length"
    if ignore_masks is not None:
        assert len(trajectory) == len(ignore_masks), "Trajectory and ignore masks must have the same length"

    groundtruth = sequence.groundtruth()
    if groundtruth is None:
        return len(thresholds) * [1.0], len(thresholds) * [0.0]

    overlaps = np.asarray(
        calculate_overlaps(trajectory, groundtruth, sequence.size if bounded else None, ignore=ignore_masks),
        dtype=np.float64,
    )
    confidence_arr = np.array(confidence)

    n_visible = len([region for region in groundtruth if not is_special(region)])

    precision = len(thresholds) * [0.0]
    recall = len(thresholds) * [0.0]

    for i, threshold in enumerate(thresholds):

        subset = confidence_arr >= threshold

        if np.sum(subset) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = float(np.mean(overlaps[subset]))
            recall[i] = float(np.sum(overlaps[subset]) / n_visible) if n_visible > 0 else 0.0

    return precision, recall

class _ConfidenceScores(SeparableAnalysis):
    """Computes the confidence scores for a tracker for given sequences.

    This is internal analysis and should not be used directly.
    """

    @property
    def _title_default(self) -> str:
        """Title of the analysis."""
        return "Aggregate confidence scores"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return (None,)

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis."""
        return isinstance(experiment, UnsupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes the confidence scores for a tracker for given sequences.

        :param experiment: Experiment to compute the confidence scores for.
        :type experiment: Experiment
        :param tracker: Tracker to compute the confidence scores for.
        :type tracker: Tracker
        :param sequence: Sequence to compute the confidence scores for.
        :type sequence: Sequence
        :param dependencies: Dependencies of the analysis.
        :type dependencies: list[Grid]

        :returns: Confidence scores for the given sequence.
        :rtype: tuple[Any, ...]"""

        assert isinstance(experiment, UnsupervisedExperiment)

        scores_all = []
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException("Missing results for sequence {}".format(sequence.name))

        for trajectory in trajectories:
            confidence = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
            scores_all.extend(confidence)

        return scores_all,

class _Thresholds(SequenceAggregator):
    """Computes the thresholds for a tracker for given sequences.

    This is internal analysis and should not be used directly.
    """

    resolution = Integer(default=100)

    @property
    def _title_default(self) -> str:
        """Title of the analysis."""
        return "Thresholds for tracking precision/recall"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return (None,)

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis."""
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Dependencies of the analysis."""
        return (_ConfidenceScores(),)

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Computes the thresholds for a tracker for given sequences.

        :param tracker: Tracker to compute the thresholds for.
        :type tracker: Tracker
        :param sequences: Sequences to compute the thresholds for.
        :type sequences: list[Sequence]
        :param results: Results of the dependencies.
        :type results: Grid

        :returns: Thresholds for the given sequences.
        :rtype: tuple[Any, ...]"""

        thresholds = determine_thresholds(itertools.chain(*[result[0] for result in results]), self.resolution),

        return thresholds,

class PrecisionRecallCurves(SeparableAnalysis):
    """Computes the precision/recall curves for a tracker for given sequences."""

    thresholds = Include(_Thresholds)
    ignore_unknown = Boolean(default=True, description="Ignore unknown regions")
    bounded = Boolean(default=True, description="Bounded evaluation")
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")

    @property
    def _title_default(self) -> str:
        """Title of the analysis."""
        return "Tracking precision/recall"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return Curve("Precision Recall curve", dimensions=2, abbreviation="PR", minimal=(0, 0), maximal=(1, 1), labels=("Recall", "Precision")), None

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis."""
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Dependencies of the analysis."""
        return (self.thresholds,)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes the precision/recall curves for a tracker for given sequences.

        :param experiment: Experiment to compute the precision/recall curves for.
        :type experiment: Experiment
        :param tracker: Tracker to compute the precision/recall curves for.
        :type tracker: Tracker
        :param sequence: Sequence to compute the precision/recall curves for.
        :type sequence: Sequence
        :param dependencies: Dependencies of the analysis.
        :type dependencies: list[Grid]

        :returns: Precision/recall curves for the given sequence.
        :rtype: tuple[Any, ...]"""

        assert isinstance(experiment, UnsupervisedExperiment)

        thresholds = dependencies[0][0, 0][0][0]

        trajectories = experiment.gather(tracker, sequence)

        ignore_masks = sequence.object(self.ignore_masks)

        if len(trajectories) == 0:
            raise MissingResultsException()

        precision = len(thresholds) * [0.0]
        recall = len(thresholds) * [0.0]
        for trajectory in trajectories:
            confidence = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
            pr, re = compute_tpr_curves(trajectory.regions(), confidence, sequence, thresholds, self.ignore_unknown, self.bounded, ignore_masks=ignore_masks)
            for i in range(len(thresholds)):
                precision[i] += pr[i]
                recall[i] += re[i]

        return [(pr / len(trajectories), re / len(trajectories)) for pr, re in zip(precision, recall)], thresholds

class PrecisionRecallCurve(SequenceAggregator):
    """Computes the average precision/recall curve for a tracker."""

    curves = Include(PrecisionRecallCurves)

    @property
    def _title_default(self) -> str:
        """Title of the analysis."""
        return "Tracking precision/recall average curve"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return self.curves.describe()

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with unsupervised experiments.
        """
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Dependencies of the analysis."""
        return (self.curves,)

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Computes the average precision/recall curve for a tracker.

        :param tracker: Tracker to compute the average precision/recall curve for.
        :type tracker: Tracker
        :param sequences: Sequences to compute the average precision/recall curve for.
        :type sequences: list[Sequence]
        :param results: Results of the dependencies.
        :type results: Grid

        :returns: Average precision/recall curve for the given sequences.
        :rtype: tuple[Any, ...]"""

        curve: list[tuple[float, float]] | None = None
        thresholds: list[float] | None = None
        divisor = len(results)

        for cell in results:
            if cell is None:
                continue
            partial, thresholds = cell
            if curve is None:
                curve = list(partial)
                continue
            curve = [(pr1 + pr2, re1 + re2) for (pr1, re1), (pr2, re2) in zip(curve, partial)]

        if curve is None or divisor == 0:
            return [], thresholds

        # Emit points as (recall, precision) to match the curve's ("Recall", "Precision") axis labels.
        curve = [(re / divisor, pr / divisor) for pr, re in curve]

        return curve, thresholds

class FScoreCurve(Analysis):
    """Computes the F-score curve for a tracker."""

    beta = Float(default=1, description="Beta value for the F-score")
    prcurve = Include(PrecisionRecallCurve)

    @property
    def _title_default(self) -> str:
        """Title of the analysis."""
        return "Tracking F-score curve"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return Plot("Tracking F-score curve", "F", wrt="normalized threshold", minimal=0, maximal=1), None

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with unsupervised experiments.
        """
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Dependencies of the analysis."""
        return (self.prcurve,)

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """Computes the F-score curve for a tracker.

        :param experiment: Experiment to compute the F-score curve for.
        :type experiment: Experiment
        :param trackers: Trackers to compute the F-score curve for.
        :type trackers: list[Tracker]
        :param sequences: Sequences to compute the F-score curve for.
        :type sequences: list[Sequence]
        :param dependencies: Dependencies of the analysis.
        :type dependencies: list[Grid]

        :returns: F-score curve for the given sequences.
        :rtype: Grid"""

        processed_results = Grid(len(trackers), 1)

        for i, result in enumerate(dependencies[0]):
            beta2 = (self.beta * self.beta)
            f_curve = [((1 + beta2) * pr_ * re_) / (beta2 * pr_ + re_) for pr_, re_ in result[0]]

            processed_results[i, 0] = (f_curve, result[1])

        return processed_results

    @property
    def axes(self) -> Axes:
        """Axes of the analysis."""
        return Axes.TRACKERS

class PrecisionRecall(Analysis):
    """Computes the average precision/recall for a tracker."""

    prcurve = Include(PrecisionRecallCurve)
    fcurve = Include(FScoreCurve)

    @property
    def _title_default(self) -> str:
        """Title of the analysis."""
        return "Tracking precision/recall"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return Measure("Precision", "Pr", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Recall", "Re", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("F Score", "F", minimal=0, maximal=1, direction=Sorting.DESCENDING)

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with unsupervised experiments.
        """
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self) -> TSequence[Analysis]:
        """Dependencies of the analysis."""
        return (self.prcurve, self.fcurve)

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """Computes the average precision/recall for a tracker.

        :param experiment: Experiment to compute the average precision/recall for.
        :type experiment: Experiment
        :param trackers: Trackers to compute the average precision/recall for.
        :type trackers: list[Tracker]
        :param sequences: Sequences to compute the average precision/recall for.
        :type sequences: list[Sequence]
        :param dependencies: Dependencies of the analysis.
        :type dependencies: list[Grid]

        :returns: Average precision/recall for the given sequences.
        :rtype: Grid"""

        f_curves = dependencies[1]
        pr_curves = dependencies[0]

        joined = Grid(len(trackers), 1)

        for i, (f_curve, pr_curve) in enumerate(zip(f_curves, pr_curves)):
            # get optimal F-score and Pr and Re at this threshold
            f_score = max(f_curve[0])
            best_i = f_curve[0].index(f_score)
            re_score = pr_curve[0][best_i][0]
            pr_score = pr_curve[0][best_i][1]
            joined[i, 0] = (pr_score, re_score, f_score)

        return joined

    @property
    def axes(self) -> Axes:
        """Axes of the analysis."""
        return Axes.TRACKERS


@dataclass(frozen=True)
class FrameCounts:
    """Per-frame outcome tally for a long-term trajectory.

    Each frame where the target is present is classified as ``tracking`` (overlap
    above threshold), ``failure`` (predicted but off-target) or ``miss`` (predicted
    empty); each frame where the target is absent as ``notice`` (correctly empty) or
    ``hallucination`` (predicted present). Fields are floats so averaged tallies are
    representable."""

    tracking: float = 0.0
    failure: float = 0.0
    miss: float = 0.0
    hallucination: float = 0.0
    notice: float = 0.0

    def __add__(self, other: "FrameCounts") -> "FrameCounts":
        return FrameCounts(self.tracking + other.tracking, self.failure + other.failure,
            self.miss + other.miss, self.hallucination + other.hallucination,
            self.notice + other.notice)

    def __truediv__(self, divisor: float) -> "FrameCounts":
        return FrameCounts(self.tracking / divisor, self.failure / divisor,
            self.miss / divisor, self.hallucination / divisor, self.notice / divisor)

    @staticmethod
    def mean(counts: TSequence["FrameCounts"]) -> "FrameCounts":
        """Element-wise mean of a non-empty sequence of tallies."""
        return sum(counts, FrameCounts()) / len(counts)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Tally as a (tracking, failure, miss, hallucination, notice) tuple."""
        return (self.tracking, self.failure, self.miss, self.hallucination, self.notice)

    @property
    def present(self) -> float:
        """Frames where the target is present (tracking + failure + miss)."""
        return self.tracking + self.failure + self.miss

    @property
    def absent(self) -> float:
        """Frames where the target is absent (notice + hallucination)."""
        return self.notice + self.hallucination

    @property
    def non_reported_error(self) -> float | None:
        """Fraction of present frames reported empty, or None if never present."""
        return self.miss / self.present if self.present > 0 else None

    @property
    def drift_rate_error(self) -> float | None:
        """Fraction of present frames tracked off-target, or None if never present."""
        return self.failure / self.present if self.present > 0 else None

    def absence_detection_quality(self, threshold: float) -> float | None:
        """Fraction of absent frames correctly reported empty, or None if there are
        at most ``threshold`` absent frames."""
        return self.notice / self.absent if self.absent > threshold else None


def count_frames(trajectory: list[Region], groundtruth: list[Region],
    bounds: Bounds | None = None, threshold: float | None = 0,
    ignore_masks: list[Region] | None = None) -> FrameCounts:
    """Classifies every frame of a trajectory against the groundtruth.

    :param trajectory: Trajectory of the tracker.
    :type trajectory: list[Region]
    :param groundtruth: Groundtruth trajectory.
    :type groundtruth: list[Region]
    :param bounds: Bounds of the sequence.
    :type bounds: Bounds | None
    :param threshold: Threshold for the overlap.
    :type threshold: float | None
    :param ignore_masks: Ignore masks.
    :type ignore_masks: list[Region] | None

    :returns: Per-frame outcome tally.
    :rtype: FrameCounts"""

    assert len(trajectory) == len(groundtruth), "Trajectory and groundtruth must have the same length"

    if ignore_masks is not None:
        assert len(trajectory) == len(ignore_masks), "Trajectory and ignore masks must have the same length"

    overlaps = np.array(calculate_overlaps(trajectory, groundtruth, bounds, ignore=ignore_masks))
    if threshold is None: threshold = -1.0

    tracking = failure = miss = hallucination = notice = 0

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        if (is_special(region_gt, SpecialCode.UNKNOWN)):
            continue
        if region_gt.is_empty():
            if region_tr.is_empty():
                notice += 1
            else:
                hallucination += 1
        else:
            if overlaps[i] > threshold:
                tracking += 1
            else:
                if region_tr.is_empty():
                    miss += 1
                else:
                    failure += 1

    return FrameCounts(tracking, failure, miss, hallucination, notice)


def mean_frame_counts(trajectories: TSequence[Iterable[Region]], groundtruth: list[Region],
    bounds: Bounds | None, ignore_masks: list[Region] | None,
    frame_mask: list[bool] | None) -> FrameCounts:
    """Averages :func:`count_frames` over a set of runs for one object, applying the
    optional per-frame filter to every trajectory."""
    gt_regions = filter_by_mask(groundtruth, frame_mask)
    masks = filter_by_mask(ignore_masks, frame_mask)
    assert gt_regions is not None

    counts = []
    for trajectory in trajectories:
        traj_regions = filter_by_mask(trajectory, frame_mask)
        assert traj_regions is not None
        counts.append(count_frames(traj_regions, gt_regions, bounds=bounds, ignore_masks=masks))
    return FrameCounts.mean(counts)

class SafeAverage(object):
    """Running average that ignores ``None`` contributions and reports ``None``
    when no values were added (avoiding division-by-zero)."""

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._count: int = 0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self._sum += value
        self._count += 1

    def average(self) -> float | None:
        if self._count == 0:
            return None
        return self._sum / self._count

    def empty(self) -> bool:
        return self._count == 0

class CountFrames(SeparableAnalysis):
    """Counts the number of frames where the tracker is correct, fails, misses,
    hallucinates or notices an object."""

    threshold = Float(default=0.0, val_min=0, val_max=1)
    bounded = Boolean(default=True)
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with multi-run experiments.
        """
        return isinstance(experiment, MultiRunExperiment)

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return (None,)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes the number of frames where the tracker is correct, fails, misses,
        hallucinates or notices an object."""

        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        distribution: list[tuple[float, float, float, float, float]] = []
        bounds = sequence.size if self.bounded else None

        ignore_masks = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask: list[bool] | None = [self.filter_tag in sequence.tags(i) for i in range(len(sequence))]
        else:
            frame_mask = None

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_groundtruth = sequence.object(o)
            assert object_groundtruth is not None, f"Missing groundtruth for object {o}"

            counts = mean_frame_counts(trajectories, object_groundtruth, bounds, ignore_masks, frame_mask)
            distribution.append(counts.as_tuple())

        return distribution,

class QualityAuxiliary(SeparableAnalysis):
    """Computes the non-reported error, drift-rate error and absence-detection
    quality."""

    threshold = Float(default=0.0, val_min=0, val_max=1)
    bounded = Boolean(default=True)
    absence_threshold = Integer(default=10, val_min=0)
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with multi-run experiments.
        """
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Quality Auxiliary"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return Measure("Non-reported Error", "NRE", 0, 1, Sorting.ASCENDING), \
            Measure("Drift-rate Error", "DRE", 0, 1, Sorting.ASCENDING), \
            Measure("Absence-detection Quality", "ADQ", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Computes the non-reported error, drift-rate error and absence-detection
        quality.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param tracker: Tracker.
        :type tracker: Tracker
        :param sequence: Sequence.
        :type sequence: Sequence
        :param dependencies: Dependencies.
        :type dependencies: list[Grid]

        :returns: Non-reported error, drift-rate error and absence-detection quality.
        :rtype: tuple[Any, ...]"""

        assert isinstance(experiment, MultiRunExperiment)

        not_reported_error = SafeAverage()
        drift_rate_error = SafeAverage()
        absence_detection = SafeAverage()

        objects = sequence.objects()
        bounds = sequence.size if self.bounded else None

        ignore_masks = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask: list[bool] | None = [self.filter_tag in sequence.tags(i) for i in range(len(sequence))]
        else:
            frame_mask = None

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_groundtruth = sequence.object(o)
            assert object_groundtruth is not None, f"Missing groundtruth for object {o}"

            counts = mean_frame_counts(trajectories, object_groundtruth, bounds, ignore_masks, frame_mask)

            not_reported_error.add(counts.non_reported_error)
            drift_rate_error.add(counts.drift_rate_error)
            absence_detection.add(counts.absence_detection_quality(self.absence_threshold))

        return not_reported_error.average(), drift_rate_error.average(), absence_detection.average(),

class AverageQualityAuxiliary(SequenceAggregator):
    """Computes the average non-reported error, drift-rate error and absence-detection
    quality."""

    analysis = Include(QualityAuxiliary)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Quality Auxiliary"

    def dependencies(self) -> TSequence[Analysis]:
        """Returns the dependencies of the analysis."""
        return (self.analysis,)

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return Measure("Non-reported Error", "NRE", 0, 1, Sorting.ASCENDING), \
            Measure("Drift-rate Error", "DRE", 0, 1, Sorting.ASCENDING), \
            Measure("Absence-detection Quality", "ADQ", 0, 1, Sorting.DESCENDING),

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with multi-run experiments.
        """
        return isinstance(experiment, MultiRunExperiment)

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregates the non-reported error, drift-rate error and absence-detection
        quality.

        :param tracker: Tracker.
        :type tracker: Tracker
        :param sequences: Sequences.
        :type sequences: list[Sequence]
        :param results: Results.
        :type results: Grid

        :returns: Non-reported error, drift-rate error and absence-detection quality.
        :rtype: tuple[Any, ...]"""

        not_reported_error = SafeAverage()
        drift_rate_error = SafeAverage()
        absence_detection = SafeAverage()

        for nre, dre, ad in results:
            not_reported_error.add(nre)
            drift_rate_error.add(dre)
            absence_detection.add(ad)

        return not_reported_error.average(), drift_rate_error.average(), absence_detection.average(),

from vot.analysis.accuracy import SequenceAccuracy

class AccuracyRobustness(Analysis):
    """Longterm multi-object accuracy-robustness measure."""

    threshold = Float(default=0.0, val_min=0, val_max=1)
    bounded = Boolean(default=True)
    counts = Include(CountFrames)
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def dependencies(self) -> TSequence[Analysis]:
        """Returns the dependencies of the analysis."""
        return (
            self.counts,
            SequenceAccuracy(burnin=0, threshold=self.threshold, bounded=self.bounded,
                ignore_invisible=True, ignore_unknown=False,
                ignore_masks=self.ignore_masks, filter_tag=self.filter_tag),
        )

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the experiment is compatible with the analysis.

        This analysis is compatible with multi-run experiments.
        """
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Accuracy-robustness"

    def describe(self) -> tuple[Result | None, ...]:
        """Describes the analysis."""
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar")

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """Aggregate results from multiple sequences into a single value.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param trackers: Trackers.
        :type trackers: list[Tracker]
        :param sequences: Sequences.
        :type sequences: list[Sequence]
        :param dependencies: Dependencies.
        :type dependencies: list[Grid]

        :returns: Aggregated results.
        :rtype: Grid"""

        frame_counts = dependencies[0]
        accuracy_analysis = dependencies[1]

        results = Grid(len(trackers), 1)

        for j in range(len(trackers)):
            accuracy = SafeAverage()
            robustness = SafeAverage()

            for i in range(len(sequences)):
                if accuracy_analysis[j, i] is None:
                    continue

                accuracy.add(accuracy_analysis[j, i][0])
                frame_counts_sequence = frame_counts[j, i][0]
                objects = len(frame_counts_sequence)
                
                sequence_robustness = SafeAverage()
                
                for o in range(objects):
                    
                    n = (frame_counts_sequence[o][0] + frame_counts_sequence[o][1] + frame_counts_sequence[o][2])
                    if n > 0: sequence_robustness.add(frame_counts_sequence[o][0] / n)

                if not sequence_robustness.empty():
                    robustness.add(sequence_robustness.average())
                
            results[j, 0] = (accuracy.average(), robustness.average(), (robustness.average(), accuracy.average()))

        return results

    @property
    def axes(self) -> Axes:
        """Returns the axes of the analysis."""
        return Axes.TRACKERS