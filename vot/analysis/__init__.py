"""This module contains classes and functions for analysis of tracker performance.

The analysis is performed on the results of an experiment.
"""

from collections import namedtuple
from enum import Enum, auto
from typing import Generic, Iterable, Sequence as TSequence, Any, cast
from typing_extensions import TypeVar
from abc import ABC, abstractmethod

from attributee import Attributee, String

from vot import ToolkitException
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.region import is_special
from vot.utilities import class_fullname, arg_hash, Registry
from vot.utilities.data import Grid

_T = TypeVar("_T")


def filter_by_mask(items: Iterable[_T] | None, mask: list[bool] | None) -> list[_T] | None:
    """Apply a per-frame boolean mask to a sequence.

    Returns a materialized list. If ``items`` is ``None`` the result is ``None``;
    if ``mask`` is ``None`` the input is materialized but not filtered. Used by
    analyses that subset frames via a ``filter_tag``.
    """
    if items is None:
        return None
    if mask is None:
        return list(items)
    return [item for item, keep in zip(items, mask) if keep]

class MissingResultsException(ToolkitException):
    """Exception class that denotes missing results during analysis."""
    def __init__(self, *args: object) -> None:
        """Constructor."""
        if not args:
            args = ("Missing results",)
        super().__init__(*args)

class Sorting(Enum):
    """Sorting direction enumeration class."""
    UNSORTABLE = auto()
    DESCENDING = auto()
    ASCENDING = auto()

class Axes(Enum):
    """Semantic information for axis in analysis grid."""
    NONE = auto()
    TRACKERS = auto()
    SEQUENCES = auto()
    BOTH = auto()

class Result(ABC):
    """Abstract result object base.

    This is the base class for all result descriptions.
    """

    def __init__(self, name: str, abbreviation: str | None = None, description: str = "") -> None:
        """Constructor.

        :param name: Name of the result, used in reports.
        :type name: str
        :param abbreviation: Shorter text representation; falls back to ``name`` if None.
        :type abbreviation: str | None, optional
        :param description: Optional longer description.
        :type description: str, optional
        """
        self._name: str = name
        self._abbreviation: str = abbreviation if abbreviation is not None else name
        self._description: str = description if description is not None else ""

    @property
    def name(self) -> str:
        """Name of the result, used in reports."""
        return self._name

    @property
    def abbreviation(self) -> str:
        """Abbreviation, if empty, then name is used.

        Can be used to define a shorter text representation.
        """
        return self._abbreviation

    @property
    def description(self) -> str:
        """Description of the result, used in reports."""
        return self._description

class Label(Result):
    """Label describes a single categorical output of an analysis.

    Can have a set of possible values.
    """

    def __init__(self, *args, **kwargs):
        """Constructor."""
        super().__init__(*args, **kwargs)

class Measure(Result):
    """Measure describes a single value numerical output of an analysis.

    Can have minimum and maximum value as well as direction of sorting.
    """

    def __init__(self, name: str, abbreviation: str | None = None, minimal: float | None = None, \
        maximal: float | None = None, direction: Sorting = Sorting.UNSORTABLE) -> None:
        """Constructor for Measure class.

            name {str} -- Name of the measure, used in reports

            abbreviation {str | None} -- Abbreviation, if empty, then name is used.
            Can be used to define a shorter text representation. (default: {None})
            minimal {float | None} -- Minimal value of the measure. If None, then the measure is not bounded from below. (default: {None})
            maximal {float | None} -- Maximal value of the measure. If None, then the measure is not bounded from above. (default: {None})
            direction {Sorting} -- Direction of sorting. If Sorting.UNSORTABLE, then the measure is not sortable. (default: {Sorting.UNSORTABLE})
        """

        super().__init__(name, abbreviation)
        self._minimal: float | None = minimal
        self._maximal: float | None = maximal
        self._direction: Sorting = direction if direction is not None else Sorting.UNSORTABLE

    @property
    def minimal(self) -> float | None:
        """Minimal value of the measure.

        If None, then the measure is not bounded from below.
        """
        return self._minimal

    @property
    def maximal(self) -> float | None:
        """Maximal value of the measure.

        If None, then the measure is not bounded from above.
        """
        return self._maximal

    @property
    def direction(self) -> Sorting:
        """Direction of sorting.

        If Sorting.UNSORTABLE, then the measure is not sortable.
        """
        return self._direction

class Drawable(Result):
    """Base class for results that can be visualized in plots."""

    def __init__(self, name: str, abbreviation: str | None = None, trait: str | None = None):
        """Constructor.

        :param name: Name of the result, used in reports.
        :type name: str
        :param abbreviation: Shorter text representation; falls back to ``name`` if None.
        :type abbreviation: str | None, optional
        :param trait: Trait of the data, used for specification. Defaults to None.
        :type trait: str | None, optional
        """
        super().__init__(name, abbreviation)
        self._trait: str | None = trait

    @property
    def trait(self) -> str | None:
        """Trait of the data, used for specification."""
        return self._trait

class Multidimensional(Drawable):
    """Base class for multidimensional results.

    This class is used to describe results that can be visualized in a scatter plot.
    """

    def __init__(self, name: str, dimensions: int, abbreviation: str | None = None,
        minimal: TSequence[float] | None = None,
        maximal: TSequence[float] | None = None,
        labels: TSequence[str] | None = None,
        trait: str | None = None):
        """Constructor for Multidimensional class.

            name {str} -- Name of the measure, used in reports
            dimensions {int} -- Number of dimensions of the result

            abbreviation {str | None} -- Abbreviation, if empty, then name is used.
            Can be used to define a shorter text representation. (default: {None})
            minimal {Sequence[float] | None} -- Minimal value per dimension. If None, no lower bound. (default: {None})
            maximal {Sequence[float] | None} -- Maximal value per dimension. If None, no upper bound. (default: {None})
            labels {Sequence[str] | None} -- Labels for each dimension. (default: {None})
            trait {str | None} -- Trait of the data, used for specification . Defaults to None.
        """

        assert dimensions > 1
        super().__init__(name, abbreviation, trait)
        self._dimensions: int = dimensions
        self._minimal: TSequence[float] | None = minimal
        self._maximal: TSequence[float] | None = maximal
        self._labels: TSequence[str] | None = labels

    @property
    def dimensions(self) -> int:
        """Number of dimensions of the result."""
        return self._dimensions

    def minimal(self, i: int) -> float | None:
        """Minimal value of the i-th dimension.

        If None, then the measure is not bounded from below.
        """
        if self._minimal is None:
            return None
        return self._minimal[i]

    def maximal(self, i: int) -> float | None:
        """Maximal value of the i-th dimension.

        If None, then the measure is not bounded from above.
        """
        if self._maximal is None:
            return None
        return self._maximal[i]

    def label(self, i: int) -> str | None:
        """Label for the i-th dimension."""
        if self._labels is None:
            return None
        return self._labels[i]

class Point(Multidimensional):
    """Point is a two or more dimensional numerical output that can be visualized in a
    scatter plot."""

class Plot(Drawable):
    """Plot describes a result in form of a list of values with optional minimum and
    maximum with respect to some unit.

    The results of the same analysis for different trackers should have the same number
    of measurements (independent variable).
    """

    def __init__(self, name: str, abbreviation: str | None = None, wrt: str = "frames", minimal: float | None = None, \
        maximal: float | None = None, trait: str | None = None):
        """Constructor for Plot class.

            name {str} -- Name of the measure, used in reports

            abbreviation {str | None} -- Abbreviation, if empty, then name is used.
            Can be used to define a shorter text representation. (default: {None})
            wrt {str} -- Unit of the independent variable. (default: {"frames"})
            minimal {float | None} -- Minimal value of the measure. If None, then the measure is not bounded from below. (default: {None})
            maximal {float | None} -- Maximal value of the measure. If None, then the measure is not bounded from above. (default: {None})
            trait {str | None} -- Trait of the data, used for specification . Defaults to None.
        """
        super().__init__(name, abbreviation, trait)
        self._wrt: str = wrt
        self._minimal: float | None = minimal
        self._maximal: float | None = maximal

    @property
    def minimal(self) -> float | None:
        """Minimal value of the measure.

        If None, then the measure is not bounded from below.
        """
        return self._minimal

    @property
    def maximal(self) -> float | None:
        """Maximal value of the measure.

        If None, then the measure is not bounded from above.
        """
        return self._maximal

    @property
    def wrt(self) -> str:
        """Unit of the independent variable."""
        return self._wrt

class Curve(Multidimensional):
    """Curve is a list of 2+ dimensional results.

    The number of elements in a list can vary between samples.
    """

class Analysis(Attributee):
    """Base class for all analysis classes.

    Analysis is a class that describes computation of one or more performance metrics for
    a given experiment.
    """

    name = String(default=None, description="Name of the analysis")

    def __init__(self, **kwargs):
        """Constructor for Analysis class."""
        super().__init__(**kwargs)
        self._identifier_cache: str | None = None

    def compatible(self, experiment: Experiment) -> bool:
        """Checks if the analysis is compatible with the experiment type."""
        raise NotImplementedError()

    @property
    def title(self) -> str:
        """Returns the title of the analysis.

        If name is not set, then the default title is returned.
        """

        if self.name is None:
            return self._title_default
        return self.name

    @property
    def _title_default(self) -> str:
        """Returns the default title of the analysis.

        This is used when name is not set.
        """
        raise NotImplementedError()

    def dependencies(self) -> TSequence["Analysis"]:
        """Returns the dependencies of the analysis.

        This is used to determine the order of execution of the analysis.
        Subclasses may return either a list or a tuple — both are iterated
        and indexed by the processor.
        """
        return ()

    @property
    def identifier(self) -> str:
        """Returns a unique identifier of the analysis.

        This is used to determine if the analysis has been already computed.
        """

        if self._identifier_cache is not None:
            return self._identifier_cache

        params = self.dump()
        del params["name"]

        confighash = arg_hash(**params)

        identifier = class_fullname(self) + "@" + confighash
        self._identifier_cache = identifier
        return identifier

    def describe(self) -> tuple[Result | None, ...]:
        """Returns a tuple of descriptions of results of the analysis."""
        raise NotImplementedError()

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """Computes the analysis for the given experiment, trackers and sequences. The
        dependencies are the results of the dependent analyses. The result is a grid
        with the results of the analysis. The grid is indexed by trackers and sequences.
        The axes are described by the axes() method.

        :param experiment: Experiment to compute the analysis for.
        :type experiment: Experiment
        :param trackers: List of trackers to compute the analysis for.
        :type trackers: list[Tracker]
        :param sequences: List of sequences to compute the analysis for.
        :type sequences: list[Sequence]
        :param dependencies: List of dependencies of the analysis.
        :type dependencies: list[Grid]

        :returns: Grid with the results of the analysis.
        :rtype: Grid
        """
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        """Returns axes semantic description for the result grid."""
        raise NotImplementedError()

    @property
    def cached(self) -> bool:
        """Returns whether the analysis should be cached."""
        return True

    def commit(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence]):
        """Commits the analysis for execution on default processor."""
        return AnalysisProcessor.commit_default(self, experiment, trackers, sequences)

    def run(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence]):
        """Runs the analysis on default processor."""
        return AnalysisProcessor.run_default(self, experiment, trackers, sequences)

_T_Tracker = TypeVar("_T_Tracker", default=Tracker)
_T_Sequence = TypeVar("_T_Sequence", default=Sequence)


class SeparableAnalysis(Analysis, Generic[_T_Tracker, _T_Sequence]):
    """Analysis that is separable with respect to trackers and/or sequences, each part
    can be processed in parallel as a separate job.

    The class is parameterised by the **per-part** tracker and sequence types:

    - ``Axes.BOTH``      → ``SeparableAnalysis[Tracker, Sequence]`` (default).
    - ``Axes.TRACKERS``  → ``SeparableAnalysis[Tracker, list[Sequence]]``.
    - ``Axes.SEQUENCES`` → ``SeparableAnalysis[list[Tracker], Sequence]``.

    Concrete subclasses bind the type parameters via the helper bases
    :class:`TrackerSeparableAnalysis` and :class:`SequenceSeparableAnalysis`,
    which lets ``subcompute`` carry exact types instead of ``Any``.
    """

    SeparablePart = namedtuple("SeparablePart", ["trackers", "sequences", "tid", "sid"])

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: _T_Tracker, sequence: _T_Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single part.

        :param experiment: Experiment from which to take results.
        :param tracker: Either one ``Tracker`` or a ``list[Tracker]`` depending on the bound type parameter.
        :param sequence: Either one ``Sequence`` or a ``list[Sequence]`` depending on the bound type parameter.
        :param dependencies: Dependency grids, already sliced by :meth:`select` to the scope of this part.

        :returns: Tuple of results of the analysis."""
        raise NotImplementedError()

    def __init__(self, **kwargs):
        """Initializes the analysis.

        The axes semantic description is checked to be compatible with the dependencies.
        """
        super().__init__(**kwargs)

        # All dependencies should be mappable to individual parts. If parts contain
        # separation only across trackers or sequences then we are unable to properly
        # assign dependencies that contain individual
        if self.axes != Axes.BOTH:
            assert all([dependency.axes != Axes.BOTH for dependency in self.dependencies()])

    def separate(self, trackers: list[Tracker], sequences: list[Sequence]) -> list["SeparableAnalysis.SeparablePart"]:
        """Separates the analysis into parts that can be processed separately.

        :param trackers: List of trackers to compute the analysis for.
        :type trackers: list[Tracker]
        :param sequences: List of sequences to compute the analysis for.
        :type sequences: list[Sequence]

        :returns: List of parts of the analysis.
        :rtype: list["SeparableAnalysis.SeparablePart"]
        """
        parts: list[SeparableAnalysis.SeparablePart] = []
        axes = self.axes
        if axes == Axes.BOTH:
            for i, tracker in enumerate(trackers):
                for j, sequence in enumerate(sequences):
                    parts.append(SeparableAnalysis.SeparablePart([tracker], [sequence], i, j))
            return parts
        if axes == Axes.TRACKERS:
            for i, tracker in enumerate(trackers):
                parts.append(SeparableAnalysis.SeparablePart([tracker], sequences, i, None))
            return parts
        if axes == Axes.SEQUENCES:
            for j, sequence in enumerate(sequences):
                parts.append(SeparableAnalysis.SeparablePart(trackers, [sequence], None, j))
            return parts
        raise ValueError(f"Cannot separate analysis with axes={axes!r}")

    def join(self, trackers: list[Tracker], sequences: list[Sequence], results: list[Grid]) -> Grid:
        """Joins the results of the analysis into a single grid. The results are indexed
        by trackers and sequences.

        :param trackers: List of trackers to compute the analysis for.
        :type trackers: list[Tracker]
        :param sequences: List of sequences to compute the analysis for.
        :type sequences: list[Sequence]
        :param results: List of per-part result grids.
        :type results: list[Grid]

        :returns: Grid with the results of the analysis.
        :rtype: Grid"""

        axes = self.axes
        if axes == Axes.BOTH:
            transformed_results = Grid(len(trackers), len(sequences))
            k = 0
            for i, _ in enumerate(trackers):
                for j, _ in enumerate(sequences):
                    transformed_results[i, j] = results[k][0, 0]
                    k += 1
            return transformed_results
        if axes == Axes.TRACKERS:
            transformed_results = Grid(len(trackers), 1)
            for i, _ in enumerate(trackers):
                transformed_results[i, 0] = results[i][0, 0]
            return transformed_results
        if axes == Axes.SEQUENCES:
            transformed_results = Grid(1, len(sequences))
            for i, _ in enumerate(sequences):
                transformed_results[0, i] = results[i][0, 0]
            return transformed_results
        raise ValueError(f"Cannot join analysis with axes={axes!r}")

    @staticmethod
    def select(meta: Analysis, data: Grid, tracker: int | None, sequence: int | None) -> Grid:
        """Select appropriate subpart of dependency results for the part, used
        internally by sequential and parallel processor. This method handles propagation
        across "singleton" dimension.

        The idea is that a certain part of the analysis will only require the part of the result corresponding
        to the tracker and/or sequence that it is processing.

        :param meta: Description of the dependency analysis
        :type meta: Analysis
        :param data: Returned data of the dependency
        :type data: Grid
        :param tracker: Index of the tracker required by the part or None
        :type tracker: int | None
        :param sequence: Index of the sequence required by the part or None
        :type sequence: int | None

        :returns: Subsection of the result, still in Grid format.
        :rtype: Grid"""
        if meta.axes == Axes.BOTH:
            assert tracker is not None and sequence is not None
            return data.cell(tracker, sequence)
        if meta.axes == Axes.TRACKERS:
            assert tracker is not None
            return data.row(tracker)
        if meta.axes == Axes.SEQUENCES:
            assert sequence is not None
            return data.column(sequence)
        return data

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """The blocking non-parallel version of computation that can be called directly.
        Splits the job in parts and runs them sequentially. For parallel execution use
        the analysis processor.

        :param experiment: Experiment from which to take results
        :type experiment: Experiment
        :param trackers: Trackers to run analysis on
        :type trackers: list[Tracker]
        :param sequences: Sequences to run analysis on
        :type sequences: list[Sequence]
        :param dependencies: Results from depndencies, if you override the class and add dependencies, you also have to override this function and handle them.
        :type dependencies: list[Grid]

        :returns: Results in a data grid object
        :rtype: Grid"""

        # Runtime narrowing: `self.axes` selects which type the per-part subcompute
        # actually receives. The casts inform the type checker of that binding;
        # they are runtime no-ops.
        if self.axes == Axes.BOTH and len(trackers) == 1 and len(sequences) == 1:
            both = cast("SeparableAnalysis[Tracker, Sequence]", self)
            return Grid.scalar(both.subcompute(experiment, trackers[0], sequences[0], dependencies))
        elif self.axes == Axes.TRACKERS and len(trackers) == 1:
            per_tracker = cast("SeparableAnalysis[Tracker, list[Sequence]]", self)
            return Grid.scalar(per_tracker.subcompute(experiment, trackers[0], sequences, dependencies))
        elif self.axes == Axes.SEQUENCES and len(sequences) == 1:
            per_sequence = cast("SeparableAnalysis[list[Tracker], Sequence]", self)
            return Grid.scalar(per_sequence.subcompute(experiment, trackers, sequences[0], dependencies))
        else:
            parts = self.separate(trackers, sequences)
            results = []
            for part in parts:
                partdependencies = [SeparableAnalysis.select(meta, data, part.tid, part.sid)
                    for meta, data in zip(self.dependencies(), dependencies)]
                results.append(self.compute(experiment, part.trackers, part.sequences, partdependencies))

            return self.join(trackers, sequences, results)

    @property
    def axes(self) -> Axes:
        """Returns the axes of the analysis.

        This is used to determine how the analysis is split into parts.
        """
        return Axes.BOTH

class SequenceAggregator(Analysis): # pylint: disable=W0223
    """Base class for sequence aggregators.

    Sequence aggregators take the results of a tracker and aggregate them over
    sequences.
    """

    def __init__(self, **kwargs):
        """Base constructor."""
        super().__init__(**kwargs)
        # We only support one dependency in aggregator ...
        assert len(self.dependencies()) == 1
        # ... it should produce a grid of results that can be averaged over sequences
        assert self.dependencies()[0].axes == Axes.BOTH

    @abstractmethod
    def aggregate(self, tracker: Tracker, sequences: list[Sequence], results: Grid) -> tuple[Any, ...]:
        """Aggregate the results of the analysis over sequences for a single tracker.

        :param tracker: Tracker to aggregate the results for.
        :type tracker: Tracker
        :param sequences: List of sequences to aggregate the results for.
        :type sequences: list[Sequence]
        :param results: Results of the analysis for the tracker and sequences.
        :type results: Grid
        """
        raise NotImplementedError()

    def compute(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence], dependencies: list[Grid]) -> Grid:
        """Compute the analysis for a list of trackers and sequences.

        :param experiment: Experiment to compute the analysis for.
        :type experiment: Experiment
        :param trackers: List of trackers to compute the analysis for.
        :type trackers: list[Tracker]
        :param sequences: List of sequences to compute the analysis for.
        :type sequences: list[Sequence]
        :param dependencies: List of dependencies, should be one grid with results of the dependency analysis.
        :type dependencies: list[Grid]

        :returns: Grid with the results of the analysis.
        :rtype: Grid"""
        results = dependencies[0]
        transformed_results = Grid(len(trackers), 1)

        for i, tracker in enumerate(trackers):
            transformed_results[i, 0] = self.aggregate(tracker, sequences, results.row(i))

        return transformed_results

    @property
    def axes(self) -> Axes:
        """The analysis is separable in trackers."""
        return Axes.TRACKERS

class TrackerSeparableAnalysis(SeparableAnalysis[Tracker, list[Sequence]]):
    """Per-tracker analysis: each part receives one tracker and the full
    sequence list (``axes = TRACKERS``)."""

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: list[Sequence], dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for a single tracker over all sequences."""
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        """The analysis is separable in trackers."""
        return Axes.TRACKERS

class SequenceSeparableAnalysis(SeparableAnalysis[list[Tracker], Sequence]):
    """Per-sequence analysis: each part receives the full tracker list and one
    sequence (``axes = SEQUENCES``)."""

    @abstractmethod
    def subcompute(self, experiment: Experiment, tracker: list[Tracker], sequence: Sequence, dependencies: list[Grid]) -> tuple[Any, ...]:
        """Compute the analysis for all trackers over a single sequence."""
        raise NotImplementedError()

    @property
    def axes(self) -> Axes:
        """The analysis is separable in sequences."""
        return Axes.SEQUENCES

analysis_registry = Registry("analysis")

from .processor import process_stack_analyses, AnalysisProcessor, AnalysisError

