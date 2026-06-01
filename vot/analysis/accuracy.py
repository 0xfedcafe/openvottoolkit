"""Accuracy analysis.

Computes average overlap between predicted and groundtruth regions.
"""

from typing import Sequence as TSequence, Any

import numpy as np
import numpy.typing as npt

from attributee import Boolean, Integer, Include, Float, String

from vot.analysis import (Analysis, Measure, Result,
                          MissingResultsException,
                          SequenceAggregator, Sorting,
                          is_special, SeparableAnalysis,
                          Curve, filter_by_mask)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (MultiRunExperiment)
from vot.region import Region, SpecialCode, calculate_overlaps
from vot.region.raster import Bounds
from vot.tracker import Tracker
from vot.utilities.data import Grid

def gather_overlaps(trajectory: list[Region], groundtruth: list[Region], burnin: int = 10,
    ignore_unknown: bool = True, ignore_invisible: bool = False,
    bounds: Bounds | None = None,
    threshold: float | None = None,
    ignore_masks: list[Region] | None = None) -> tuple[npt.NDArray, list[int]]:
    """Gather overlaps between trajectory and groundtruth regions.

    :param trajectory: List of regions predicted by the tracker.
    :type trajectory: list[Region]
    :param groundtruth: List of groundtruth regions.
    :type groundtruth: list[Region]
    :param burnin: Number of frames to skip at the beginning of the sequence. Defaults to 10.
    :type burnin: int, optional
    :param ignore_unknown: Ignore unknown regions in the groundtruth. Defaults to True.
    :type ignore_unknown: bool, optional
    :param ignore_invisible: Ignore invisible regions in the groundtruth. Defaults to False.
    :type ignore_invisible: bool, optional
    :param bounds: Bounds of the sequence. Defaults to None.
    :type bounds: Bounds | None, optional
    :param threshold: Minimum overlap to consider. Defaults to None.
    :type threshold: float, optional
    :param ignore_masks: List of regions to ignore. Defaults to None.
    :type ignore_masks: list[Region], optional

    :returns: Overlaps for the considered frames and the indices of those frames.
    :rtype: tuple[npt.NDArray, list[int]]"""

    assert len(trajectory) == len(groundtruth), "Trajectory and groundtruth must have the same length."

    if ignore_masks is not None:
        assert len(trajectory) == len(ignore_masks), "Trajectory and ignore mask must have the same length."

    overlaps = np.array(calculate_overlaps(trajectory, groundtruth, bounds, ignore=ignore_masks))
    mask = np.ones(len(overlaps), dtype=bool)

    if threshold is None: threshold = -1

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        # Skip if groundtruth is unknown
        if is_special(region_gt, SpecialCode.UNKNOWN):
            mask[i] = False
        elif ignore_invisible and region_gt.is_empty():
            mask[i] = False
        # Skip if predicted is unknown
        elif is_special(region_tr, SpecialCode.UNKNOWN) and ignore_unknown:
            mask[i] = False
        # Skip if predicted is initialization frame
        elif is_special(region_tr, SpecialCode.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region_tr, SpecialCode.FAILURE) or is_special(region_tr, SpecialCode.CRASH):
            mask[i] = False
        elif overlaps[i] <= threshold:
            mask[i] = False

    return overlaps[mask], [i for i in range(len(overlaps)) if mask[i]]

class Overlaps(SeparableAnalysis):
    """Overlaps analysis.

    Computes overlaps between predicted and groundtruth regions.
    """

    burnin = Integer(default=10, val_min=0, description="Number of frames to skip after the initialization.")
    ignore_unknown = Boolean(default=True, description="Ignore unknown regions in the groundtruth.")
    ignore_invisible = Boolean(default=False, description="Ignore invisible regions in the groundtruth.")
    bounded = Boolean(default=True, description="Consider only the bounded region of the sequence.")
    threshold = Float(default=None, val_min=0, val_max=1, description="Minimum overlap to consider.")
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Overlaps"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single sequence.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param tracker: Tracker.
        :type tracker: Tracker
        :param sequence: Sequence.
        :type sequence: Sequence
        :param dependencies: List of dependencies.
        :type dependencies: list[Grid]

        :returns: Tuple of results.
        :rtype: tuple[Any, ...]"""
        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        bounds = sequence.size if self.bounded else None

        ignore_masks = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask: list[bool] | None = [self.filter_tag in sequence.tags(i) for i in range(len(sequence))]
        else:
            frame_mask = None

        results = []

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_groundtruth = sequence.object(o)
            assert object_groundtruth is not None, f"Missing groundtruth for object {o}"

            for trajectory in trajectories:
                traj_regions = filter_by_mask(trajectory, frame_mask)
                gt_regions = filter_by_mask(object_groundtruth, frame_mask)
                masks = filter_by_mask(ignore_masks, frame_mask)
                assert traj_regions is not None and gt_regions is not None

                overlaps, frames = gather_overlaps(traj_regions, gt_regions, self.burnin,
                                        ignore_unknown=self.ignore_unknown, ignore_invisible=self.ignore_invisible,
                                        bounds=bounds, threshold=self.threshold, ignore_masks=masks)

                results.append((o, overlaps, frames))

        return results,

class SequenceAccuracy(SeparableAnalysis):
    """Sequence accuracy analysis.

    Computes average overlap between predicted and groundtruth regions.
    """

    burnin = Integer(default=10, val_min=0, description="Number of frames to skip after the initialization.")
    ignore_unknown = Boolean(default=True, description="Ignore unknown regions in the groundtruth.")
    ignore_invisible = Boolean(default=False, description="Ignore invisible regions in the groundtruth.")
    bounded = Boolean(default=True, description="Consider only the bounded region of the sequence.")
    threshold = Float(default=None, val_min=0, val_max=1, description="Minimum overlap to consider.")
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Sequence accuracy"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single sequence.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param tracker: Tracker.
        :type tracker: Tracker
        :param sequence: Sequence.
        :type sequence: Sequence
        :param dependencies: List of dependencies.
        :type dependencies: list[Grid]

        :returns: Tuple of results.
        :rtype: tuple[Any, ...]"""
        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        objects_accuracy: float = 0.0
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

            cummulative: float = 0.0

            for trajectory in trajectories:
                traj_regions = filter_by_mask(trajectory, frame_mask)
                gt_regions = filter_by_mask(object_groundtruth, frame_mask)
                masks = filter_by_mask(ignore_masks, frame_mask)
                assert traj_regions is not None and gt_regions is not None

                overlaps, _ = gather_overlaps(traj_regions, gt_regions, self.burnin,
                                        ignore_unknown=self.ignore_unknown, ignore_invisible=self.ignore_invisible,
                                        bounds=bounds, threshold=self.threshold, ignore_masks=masks)

                if overlaps.size > 0:
                    cummulative += float(np.mean(overlaps))

            objects_accuracy += cummulative / len(trajectories)

        return objects_accuracy / len(objects),

class AverageAccuracy(SequenceAggregator):
    """Average accuracy analysis.

    Computes average overlap between predicted and groundtruth regions.
    """

    analysis = Include(SequenceAccuracy, description="Sequence accuracy analysis.")
    weighted = Boolean(default=True, description="Weight accuracy by the number of frames.")

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis.

        This analysis requires a multirun experiment.
        """
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Accuracy"

    def dependencies(self) -> TSequence[Analysis]:
        """List of dependencies."""
        return (self.analysis,)

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Measure(self.title, "", 0, 1, Sorting.DESCENDING),

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregate the results of the analysis.

        :param tracker: Tracker (unused — accuracy is averaged across sequences only).
        :type tracker: Tracker
        :param sequences: List of sequences.
        :type sequences: list[Sequence]
        :param results: Grid of results.
        :type results: Grid

        :returns: Tuple of results.
        :rtype: tuple[Any, ...]"""

        accuracy: float = 0.0
        frames: int = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1

        if frames == 0:
            return (0.0,)
        return (accuracy / frames,)

class SuccessPlot(SeparableAnalysis):
    """Success plot analysis.

    Computes the success plot of the tracker.
    """

    ignore_unknown = Boolean(default=True, description="Ignore unknown regions in the groundtruth.")
    ignore_invisible = Boolean(default=False, description="Ignore invisible regions in the groundtruth.")
    burnin = Integer(default=0, val_min=0, description="Number of frames to skip after the initialization.")
    bounded = Boolean(default=True, description="Consider only the bounded region of the sequence.")
    threshold = Float(default=None, val_min=0, val_max=1, description="Minimum overlap to consider.")
    resolution = Integer(default=100, val_min=2, description="Number of points in the plot.")
    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")
    filter_tag = String(default=None, description="Filter tag for the analysis.")

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis.

        This analysis is only compatible with multi-run experiments.
        """
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Sequence success plot"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Curve("Plot", 2, "S", minimal=(0, 0), maximal=(1, 1), labels=("Threshold", "Success"), trait="success"),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single sequence.

        :param experiment: Experiment.
        :type experiment: Experiment
        :param tracker: Tracker.
        :type tracker: Tracker
        :param sequence: Sequence.
        :type sequence: Sequence
        :param dependencies: List of dependencies.
        :type dependencies: list[Grid]

        :returns: Tuple of results.
        :rtype: tuple[Any, ...]"""

        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        bounds = sequence.size if self.bounded else None

        axis_x = np.linspace(0, 1, self.resolution)
        axis_y = np.zeros_like(axis_x)

        ignore_masks = sequence.object(self.ignore_masks)

        if self.filter_tag is not None:
            frame_mask: list[bool] | None = [self.filter_tag in sequence.tags(i) for i in range(len(sequence))]
        else:
            frame_mask = None

        valid_objects = 0

        for o in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[o])
            if len(trajectories) == 0:
                raise MissingResultsException()

            object_groundtruth = sequence.object(o)
            assert object_groundtruth is not None, f"Missing groundtruth for object {o}"

            object_y = np.zeros_like(axis_x)
            valid_trajectories = 0

            for trajectory in trajectories:
                traj_regions = filter_by_mask(trajectory, frame_mask)
                gt_regions = filter_by_mask(object_groundtruth, frame_mask)
                masks = filter_by_mask(ignore_masks, frame_mask)
                assert traj_regions is not None and gt_regions is not None

                overlaps, _ = gather_overlaps(traj_regions, gt_regions, burnin=self.burnin,
                                            ignore_unknown=self.ignore_unknown,
                                            ignore_invisible=self.ignore_invisible,
                                            bounds=bounds, threshold=self.threshold, ignore_masks=masks)

                if len(overlaps) == 0:
                    continue

                valid_trajectories += 1

                for i, threshold in enumerate(axis_x):
                    if threshold == 1:
                        # Nicer handling of the edge case
                        object_y[i] += np.sum(overlaps >= threshold) / len(overlaps)
                    else:
                        object_y[i] += np.sum(overlaps > threshold) / len(overlaps)

            if valid_trajectories == 0:
                continue

            valid_objects += 1

            axis_y += object_y / valid_trajectories

        if valid_objects > 0:
            axis_y /= valid_objects

        return [(x, y) for x, y in zip(axis_x, axis_y)],

class AverageSuccessPlot(SequenceAggregator):
    """Average success plot analysis.

    Computes the average success plot of the tracker.
    """

    resolution = Integer(default=100, val_min=2)
    analysis = Include(SuccessPlot)

    def dependencies(self) -> TSequence[Analysis]:
        """List of dependencies."""
        return (self.analysis,)

    def compatible(self, experiment: Experiment) -> bool:
        """Check if the experiment is compatible with the analysis.

        This analysis is only compatible with multi-run experiments.
        """
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self) -> str:
        """Default title of the analysis."""
        return "Success plot"

    def describe(self) -> tuple[Result | None, ...]:
        """Describe the analysis."""
        return Curve("Plot", 2, "S", minimal=(0, 0), maximal=(1, 1), labels=("Threshold", "Success"), trait="success"),

    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregate the results of the analysis.

        :param tracker: Tracker (unused — success plot averages across sequences only).
        :type tracker: Tracker
        :param sequences: List of sequences.
        :type sequences: list[Sequence]
        :param results: Grid of results.
        :type results: Grid

        :returns: Tuple of results.
        :rtype: tuple[Any, ...]"""

        axis_x = np.linspace(0, 1, self.resolution)
        axis_y = np.zeros_like(axis_x)

        for i, _ in enumerate(sequences):
            if results[i, 0] is None:
                continue

            curve = results[i, 0][0]

            for j, (_, y) in enumerate(curve):
                axis_y[j] += y

        axis_y /= len(sequences)

        return [(x, y) for x, y in zip(axis_x, axis_y)],
