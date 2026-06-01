"""Experiments are the main building blocks of the toolkit.

They are used to evaluate trackers on sequences in various ways.
"""

import typing
from datetime import datetime
from abc import abstractmethod
from typing import Any, Callable, TYPE_CHECKING

from attributee import Attributee, Object, Integer, Float, Nested, List, Boolean

from vot.tracker import TrackerException, ObjectStatus
from vot.utilities import Progress, to_number, Registry
from vot.dataset.proxy import IgnoreSpecialObjects

if TYPE_CHECKING:
    from vot.dataset import Sequence
    from vot.workspace.storage import Storage, Results
    from vot.tracker import Tracker, TrackerRuntime

class RealtimeConfig(Attributee):
    """Config proxy for real-time experiment."""

    grace = Integer(val_min=0, default=0)
    fps = Float(val_min=0, default=20)

class NoiseConfig(Attributee):
    """Config proxy for noise modifiers in experiments."""
    # Not implemented yet
    placeholder = Integer(default=1)

class InjectConfig(Attributee):
    """Config proxy for parameter injection in experiments."""
    # Not implemented yet
    placeholder = Integer(default=1)

def transformer_resolver(typename: str, context: Any, **kwargs: Any) -> Any:
    """Resolve a transformer from a string. If the transformer is not registered, it is
    imported as a class and instantiated with the provided arguments.

    :param typename: Name of the transformer
    :type typename: str
    :param context: Context of the resolver
    :type context: Attributee

    :returns: Resolved transformer
    :rtype: Transformer"""
    from vot.utilities import import_class
    from vot.experiment.transformer import Transformer


    from vot.workspace.storage import FilesystemStorage

    parent_storage = context.parent.storage
    if isinstance(parent_storage, FilesystemStorage):
        storage = parent_storage.substorage("cache").substorage("transformer")
    else:
        storage = None

    if typename in transformer_registry:
        transformer = transformer_registry.get(typename, cache=storage, **kwargs)
        assert isinstance(transformer, Transformer)
        return transformer
    else:
        transformer_class = import_class(typename)
        assert issubclass(transformer_class, Transformer)
        return transformer_class(cache=storage, **kwargs)

def analysis_resolver(typename: str, context: Any, **kwargs: Any) -> Any:
    """Resolve an analysis from a string. If the analysis is not registered, it is
    imported as a class and instantiated with the provided arguments.

    :param typename: Name of the analysis
    :type typename: str
    :param context: Context of the resolver
    :type context: Attributee

    :returns: Resolved analysis
    :rtype: Analysis"""
    from vot.utilities import import_class
    from vot.analysis import Analysis, analysis_registry

    if typename in analysis_registry:
        analysis = analysis_registry.get(typename, **kwargs)
        assert isinstance(analysis, Analysis)
    else:
        analysis_class = import_class(typename)
        assert issubclass(analysis_class, Analysis)
        analysis = analysis_class(**kwargs)

    assert analysis.compatible(context.parent)

    return analysis

class Experiment(Attributee):
    """Experiment abstract base class.

    Each experiment is responsible for running a tracker on a sequence and storing
    results into dedicated storage.
    """

    realtime = Nested(RealtimeConfig, default=None, description="Realtime modifier config")
    noise = Nested(NoiseConfig, default=None)
    inject = Nested(InjectConfig, default=None)
    transformers = List(Object(transformer_resolver), default=[])
    analyses = List(Object(analysis_resolver), default=[])
    ignore_special = Boolean(default=True, description="Ignore special objects in experiment")

    def __init__(self, _identifier: str, _storage: "Storage", **kwargs):
        """Initialize an experiment.

        :param _identifier: Identifier of the experiment
        :type _identifier: str
        :param _storage: Storage to use for storing results
        :type _storage: Storage

        :param **kwargs: Additional arguments

        :raises ValueError: If the identifier is not valid"""
        self._identifier = _identifier
        self._storage = _storage
        super().__init__(**kwargs)
        # TODO: validate analysis names

    @property
    def storage(self) -> "Storage":
        """Storage to use for storing results. Can be None if the experiment is not
        supposed to store results.

        :returns: Storage to use for storing results
        :rtype: Storage"""
        return self._storage

    @property
    def identifier(self) -> str:
        """Identifier of the experiment.

        :returns: Identifier of the experiment
        :rtype: str"""
        return self._identifier

    @property
    def _multiobject(self) -> bool:
        """Whether the experiment is multi-object or not.

        :returns: Whether the experiment is multi-object or not
        :rtype: bool"""
        # TODO: at some point this may be a property for all experiments
        return False

    def _get_initialization(self, sequence: "Sequence", index: int, oid: str | None = None) -> ObjectStatus:
        """Get initialization for a given sequence, index and object id.

        :param sequence: Sequence to get initialization for
        :param index: Index of the frame to get initialization for
        :param oid: Object id to get initialization for. When ``None`` and the
            experiment is single-object, the sequence-level groundtruth is used.

        :returns: Initialization state for the given sequence, index and object id"""
        if not self._multiobject and oid is None:
            region = sequence.groundtruth(index)
            assert region is not None, "Missing groundtruth for sequence initialization"
            return ObjectStatus(region, {})
        # ``oid`` is non-None here because either the experiment is multi-object
        # (every object identified explicitly) or the caller passed an explicit id.
        assert oid is not None
        region = sequence.frame(index).object(oid)
        assert region is not None, "Missing groundtruth for object {}".format(oid)
        return ObjectStatus(region, {})

    def _get_runtime(self, tracker: "Tracker", sequence: "Sequence", multiobject: bool = False) -> "TrackerRuntime":
        """Get runtime for a given tracker and sequence. Can convert single-object
        runtimes to multi-object runtimes.

        :param tracker: Tracker to get runtime for
        :param sequence: Sequence to get runtime for
        :param multiobject: Whether the runtime should be multi-object or not

        :returns: Runtime for the given tracker and sequence
        :raises TrackerException: If the tracker does not support multi-object experiments"""
        from vot.tracker import OnlineTrackerRuntime, RealtimeTrackerRuntime

        runtime: "TrackerRuntime" = tracker.runtime()

        if self.realtime is not None:
            grace = to_number(self.realtime.grace, min_n=0)
            fps = to_number(self.realtime.fps, min_n=0, conversion=float)
            interval = 1 / float(typing.cast(float, sequence.metadata("fps", fps)))
            if not isinstance(runtime, OnlineTrackerRuntime):
                raise TrackerException(
                    "Realtime experiments require an online tracker runtime",
                    tracker=tracker,
                )
            runtime = RealtimeTrackerRuntime(runtime, grace, interval)

        return runtime

    @abstractmethod
    def execute(
        self,
        tracker: "Tracker",
        sequence: "Sequence",
        force: bool = False,
        callback: Callable[[float], None] | None = None,
    ) -> None:
        """Execute the experiment for a given tracker and sequence.

        :param tracker: Tracker to execute
        :param sequence: Sequence to execute
        :param force: Whether to force execution even if the results are already present
        :param callback: Optional callback called with progress in [0, 1] per repetition."""
        raise NotImplementedError

    @abstractmethod
    def scan(self, tracker: "Tracker", sequence: "Sequence") -> tuple:
        """Scan stored results for a given tracker and sequence.

        :param tracker: Tracker to scan results for
        :param sequence: Sequence to scan results for

        :returns: ``(complete, files, results)`` — a completeness flag, the list of
            present result files, and the results object."""
        raise NotImplementedError

    def results(self, tracker: "Tracker", sequence: "Sequence") -> "Results":
        """Get results for a given tracker and sequence.

        :param tracker: Tracker to get results for
        :type tracker: Tracker
        :param sequence: Sequence to get results for
        :type sequence: Sequence

        :returns: Results for the tracker and sequence
        :rtype: Results"""
        if tracker.storage is not None:
            return tracker.storage.results(tracker, self, sequence)
        return self._storage.results(tracker, self, sequence)

    def log(self, identifier: str):
        """Get a log file for the experiment.

        :param identifier: Identifier of the log

        :returns: Writable log file handle (from the workspace storage)."""
        return self._storage.substorage("logs").write("{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.log".format(identifier, datetime.now()))

    def transform(self, sequences: "Sequence | list[Sequence]") -> "list[Sequence]":
        """Transform a list of sequences using the experiment transformers.

        :param sequences: A single sequence or list of sequences to transform.

        :returns: List of transformed sequences. The number of sequences may be larger
            than the input because some transformers split sequences."""
        from vot.dataset import Sequence
        from vot.experiment.transformer import SingleObject
        from vot import get_logger

        if isinstance(sequences, Sequence):
            sequences = [sequences]

        transformers = list(self.transformers)

        if not self._multiobject:
            get_logger().debug("Adding single object transformer since experiment is not multi-object")
            transformers.insert(0, SingleObject(cache=None))

        # Process sequences one transformer at the time. The number of sequences may grow
        for transformer in transformers:
            transformed: list["Sequence"] = []
            for sequence in sequences:
                get_logger().debug(
                    "Transforming sequence %s with transformer %s.%s",
                    sequence.identifier,
                    transformer.__class__.__module__,
                    transformer.__class__.__name__,
                )
                transformed.extend(transformer(sequence))
            sequences = transformed

        if self.ignore_special:
            sequences = [IgnoreSpecialObjects(sequence) for sequence in sequences]

        return sequences

from .multirun import UnsupervisedExperiment, SupervisedExperiment
from .multistart import MultiStartExperiment

def run_experiment(
    experiment: Experiment,
    tracker: "Tracker",
    sequences: list["Sequence"],
    force: bool = False,
    persist: bool = False,
) -> None:
    """A helper function that performs a given experiment with a given tracker on a list
    of sequences.

    :param experiment: The experiment object
    :param tracker: The tracker object
    :param sequences: List of sequences.
    :param force: Ignore the cached results, rerun all the experiments. Defaults to False.
    :param persist: Continue running even if exceptions were raised. Defaults to False.

    :raises TrackerException: If the experiment is interrupted"""

    class EvaluationProgress(object):
        """A helper class that wraps a progress bar and updates it based on the number
        of finished sequences.

        Internally the bar is sized at ``total * _RESOLUTION`` integer ticks so the
        fractional per-sequence callbacks (``progress`` in [0, 1]) can advance the
        bar without dropping sub-sequence updates to zero. ``Progress.absolute``
        only accepts ``int`` — the resolution multiplier is how we keep its
        contract while still getting smooth visual progress.
        """

        _RESOLUTION: int = 1000

        def __init__(self, description: str, total: int) -> None:
            self._total: int = total
            self.bar = Progress(description, total * self._RESOLUTION)
            self._finished: int = 0

        def __call__(self, progress: float) -> None:
            """Update the progress bar. ``progress`` is in [0, 1] within the current sequence."""
            clamped = min(1.0, max(0.0, progress))
            self.bar.absolute(self._finished * self._RESOLUTION + int(clamped * self._RESOLUTION))

        def push(self) -> None:
            """Advance the progress bar to the next sequence."""
            self._finished = self._finished + 1
            self.bar.absolute(self._finished * self._RESOLUTION)

        def close(self) -> None:
            """Close the progress bar."""
            self.bar.close()

    from vot import get_logger

    logger = get_logger()

    transformed = []
    for sequence in sequences:
        transformed.extend(experiment.transform(sequence))
    sequences = transformed

    progress = EvaluationProgress("{}/{}".format(tracker.identifier, experiment.identifier), len(sequences))
    for sequence in sequences:
        try:
            experiment.execute(tracker, sequence, force=force, callback=progress)
        except TrackerException as te:
            logger.error("Tracker %s encountered an error at sequence %s: %s", te.tracker.identifier, sequence.name, te)
            logger.debug(te, exc_info=True)
            if te.log is not None:
                with experiment.log(te.tracker.identifier) as flog:
                    flog.write(te.log)
                    logger.error("Tracker output written to file: %s", flog.name)
            if not persist:
                raise TrackerException("Experiment interrupted", te, tracker=tracker)
        progress.push()

    progress.close()
    
experiment_registry = Registry("experiment")
transformer_registry = Registry("transformer")
