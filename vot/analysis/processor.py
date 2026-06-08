"""This module contains the implementation of the analysis processor.

The processor is responsible for executing the analysis tasks in parallel and caching
the results.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, namedtuple
from collections.abc import Iterable, Mapping
from functools import partial
from typing import (
    Any, Callable, TYPE_CHECKING,
)
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import RLock, Condition
from queue import Queue, Empty

from cachetools import Cache
from bidict import bidict

from vot import ToolkitException
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.experiment import Experiment
from vot.analysis import SeparableAnalysis, Analysis
from vot.utilities import arg_hash, class_fullname, Progress
from vot.utilities.data import Grid

if TYPE_CHECKING:
    from vot.workspace import Workspace

logger = logging.getLogger("vot")

# Type aliases for clarity. ``HashKey`` is the cache/promise dictionary key
# returned by :func:`hashkey`. ``TrackersArg``/``SequencesArg`` represent the
# polymorphic input shape that ``AnalysisProcessor.commit`` accepts.
HashKey = tuple[Any, ...]
TrackersArg = Tracker | list[Tracker]
SequencesArg = Sequence | list[Sequence]


def hashkey(analysis: Analysis, *args: Any) -> HashKey:
    """Compute a hash key for the analysis and its arguments.

    The key is used for caching the results.
    """
    def transform(arg: Any) -> Any:
        """Transform an argument into a hashable object."""
        if isinstance(arg, Sequence):
            return arg.name
        if isinstance(arg, Tracker):
            return arg.reference
        if isinstance(arg, Experiment):
            return arg.identifier
        if isinstance(arg, Mapping):
            return arg_hash(**{k: transform(v) for k, v in arg.items()})
        if isinstance(arg, Iterable):
            return arg_hash(*[transform(i) for i in arg])
        return arg

    return (analysis.identifier, *[transform(arg) for arg in args])


def unwrap(arg: Any) -> Any:
    """Unwrap a single element list."""

    if isinstance(arg, list) and len(arg) == 1:
        return arg[0]
    return arg


class _CachedFuture(Future):
    """``Future`` variant with a ``cached`` attribute consulted by the result
    storage hook on completion. Carrying this through a typed attribute avoids
    sprinkling ``setattr`` over plain ``Future`` instances."""

    def __init__(self, cached: bool = False) -> None:
        super().__init__()
        self.cached: bool = cached

class AnalysisError(ToolkitException):
    """An exception that is raised when an analysis fails."""

    def __init__(self, cause: BaseException | None, task: HashKey | None = None) -> None:
        """Creates an analysis error.

        :param cause: The underlying exception that triggered the failure.
        :param task: The :func:`hashkey` of the task that produced the error.
        """
        self._tasks: list[HashKey | None] = []
        self._cause: BaseException | None = cause
        super().__init__(cause, task)
        self._tasks.append(task)

    @property
    def task(self) -> HashKey | None:
        """The task key that caused the error."""
        return self._tasks[-1]

    def __str__(self) -> str:
        """String representation of the error."""
        return "Error during analysis {}".format(self.task)

    def print(self, logoutput: logging.Logger) -> None:
        """Print the error to the log output."""
        logoutput.error(str(self))
        if len(self._tasks) > 1:
            for task in reversed(self._tasks[:-1]):
                logoutput.debug("Caused by an error in subtask: %s", str(task))
        if self.__cause__ is not None:
            logoutput.exception(self.__cause__)

    @property
    def root_cause(self) -> BaseException | None:
        """The root cause of the error."""
        cause = self._cause
        if cause is None:
            return None
        if isinstance(cause, AnalysisError):
            return cause.root_cause
        return cause

class DebugExecutor(Executor):
    """A synchronous executor used for debugging.

    Do not use it in practice.
    """

    Task = namedtuple("Task", ["fn", "args", "kwargs", "promise"])

    def __init__(self, strict: bool = True) -> None:
        """Creates a single-thread debug executor.

        :param strict: Strict mode means that the executor stops if any of the tasks fails. Defaults to True.
        """
        self._queue: Queue[DebugExecutor.Task] = Queue()
        self._lock = threading.RLock()
        self._semaphor = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._run)
        self._alive: bool = True
        self._strict: bool = strict
        self._thread.start()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submits a task to the executor."""

        promise: Future[Any] = Future()
        with self._lock:
            self._queue.put(DebugExecutor.Task(fn, args, kwargs, promise))
            self._semaphor.notify()
            logger.debug("Adding task %s to queue", fn)
            return promise

    def _run(self) -> None:
        """The main loop of the executor."""

        while True:

            with self._lock:
                if not self._alive:
                    break

                try:
                    task = self._queue.get(False)
                except Empty:
                    self._semaphor.wait()
                    continue

            if task.promise.cancelled():
                logger.debug("Task %s cancelled, skipping", task.fn)
                continue

            error: BaseException | None = None

            try:

                logger.debug("Running task %s", task.fn)
                result = task.fn(*task.args, **task.kwargs)
                task.promise.set_result(result)
                logger.debug("Task %s completed", task.fn)


            except Exception as e:

                error = e

                logger.info("Task %s resulted in exception: %s", task.fn, e)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(e)

            if error is not None:
                task.promise.set_exception(error)

                if self._strict:
                    self._alive = False
                    self._clear()
                    break

    def _clear(self) -> None:
        """Clears the queue."""
        with self._lock:

            while True:
                try:
                    task = self._queue.get(False)
                    if not task.promise.done():
                        task.promise.cancel()
                except Empty:
                    break

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shuts down the executor. If wait is True, the method blocks until all tasks
        are completed.

        The ``cancel_futures`` keyword argument exists for signature compatibility
        with :class:`concurrent.futures.Executor` — it is treated identically to
        the always-on ``_clear`` behaviour of this executor.

        :param wait: Wait for all tasks to complete. Defaults to True.
        :param cancel_futures: Ignored — the queue is always cleared on shutdown.
        """

        del cancel_futures  # always-on behaviour, see docstring.
        self._alive = False
        self._clear()
        if wait:
            with self._lock:
                self._semaphor.notify()

            self._thread.join()

_MappingFn = Callable[..., list[Any]]


class ExecutorWrapper(object):
    """A wrapper for an executor that allows to submit tasks with dependencies."""

    def __init__(self, executor: Executor) -> None:
        """Creates an executor wrapper.

        :param executor: The executor to wrap.
        """
        self._lock = RLock()
        self._executor: Executor = executor
        self._pending: "OrderedDict[Future[Any], FuturesAggregator]" = OrderedDict()
        self._total: int = 0

    @property
    def total(self) -> int:
        """The total number of tasks submitted to the executor."""
        return self._total

    def submit(
        self,
        fn: Callable[..., Any],
        *futures: Future[Any],
        mapping: _MappingFn | None = None,
    ) -> Future[Any]:
        """Submits a task to the executor. The task will be executed when all futures
        are completed.

        :param fn: The task to execute.
        :param futures: The futures that must be completed before the task is executed.
        :param mapping: Optional callable that maps the aggregated dependency results
            into the positional arguments for ``fn``.

        :returns: A future that will be completed when the task is completed."""

        with self._lock:

            self._total += 1

            if len(futures) == 0:
                return self._executor.submit(fn)

            depend = FuturesAggregator(*futures)

            proxy: Future[Any] = Future()
            self._pending[proxy] = depend

            proxy.add_done_callback(self._proxy_done)
            depend.add_done_callback(partial(self._ready_callback, fn, mapping, proxy))

            return proxy

    def _ready_callback(
        self,
        fn: Callable[..., Any],
        mapping: _MappingFn | None,
        proxy: Future[Any],
        future: Future[Any],
    ) -> None:
        """Internally handles completion of dependencies.

        Submits the task to the executor.
        """

        with self._lock:

            if proxy not in self._pending:
                return

            del self._pending[proxy]

            if future.cancelled():
                proxy.cancel()
            if not proxy.set_running_or_notify_cancel():
                return
            exception = future.exception()
            if exception is not None:
                proxy.set_exception(exception)
                return

            if mapping is None:
                dependencies = future.result()
            else:
                dependencies = mapping(*future.result())

            internal = self._executor.submit(fn, *dependencies)
            internal.add_done_callback(partial(self._done_callback, proxy))

    def _done_callback(self, proxy: Future[Any], future: Future[Any]) -> None:
        """Internally handles completion of executor future, copies result to proxy."""

        if future.cancelled():
            proxy.cancel()
            return
        exception = future.exception()
        if exception is not None:
            proxy.set_exception(exception)
        else:
            result = future.result()
            proxy.set_result(result)

    def _proxy_done(self, future: Future[Any]) -> None:
        """Internally handles events for proxy futures, this means handling
        cancellation."""

        with self._lock:

            if future not in self._pending:
                return

            dependency = self._pending[future]

            del self._pending[future]

            if future.cancelled():
                dependency.cancel()

class FuturesAggregator(Future):
    """A future that aggregates results from other futures."""

    def __init__(self, *futures: Future[Any]) -> None:
        """Initializes the aggregator.

        :param futures: The futures to aggregate.
        """

        super().__init__()
        self._lock = RLock()
        self._results: list[Any] = [None] * len(futures)
        self._tasks: list[Future[Any]] = list(futures)

        for i, future in enumerate(futures):
            future.add_done_callback(partial(self._on_result, i))

        if not self._results:
            self.set_result([])

    def _on_result(self, i: int, future: Future[Any]) -> None:
        """Handles completion of a dependency future."""

        with self._lock:
            if self.done():
                return
            try:
                self._results[i] = future.result()
            except Exception as e:
                self.set_exception(e)
                return

            if all([x is not None for x in self._results]):
                self.set_result(self._results)

    def _on_done(self, future: Future[Any]) -> None:
        """Handles completion of the future."""

        with self._lock:
            try:
                self.set_result(future.result())
            except AnalysisError as e:
                self.set_exception(e)

    def cancel(self) -> bool:
        """Cancels the future and all dependencies."""

        with self._lock:
            for promise in self._tasks:
                promise.cancel()
            return super().cancel()


def _as_list_trackers(trackers: TrackersArg) -> list[Tracker]:
    """Normalize the polymorphic ``trackers`` argument to a list."""
    return [trackers] if isinstance(trackers, Tracker) else trackers


def _as_list_sequences(sequences: SequencesArg) -> list[Sequence]:
    """Normalize the polymorphic ``sequences`` argument to a list."""
    return [sequences] if isinstance(sequences, Sequence) else sequences


class AnalysisTask(object):
    """A task that computes an analysis."""

    def __init__(
        self,
        analysis: Analysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> None:
        """Initializes a new instance of the AnalysisTask class.

        :param analysis: The analysis to compute.
        :param experiment: The experiment to compute the analysis for.
        :param trackers: A single tracker or a list of trackers.
        :param sequences: A single sequence or a list of sequences.
        """

        self._analysis = analysis
        self._trackers: list[Tracker] = _as_list_trackers(trackers)
        self._experiment = experiment
        self._sequences: list[Sequence] = _as_list_sequences(sequences)
        self._key: HashKey = hashkey(analysis, experiment, self._trackers, self._sequences)

    def __call__(self, dependencies: list[Grid] | None = None) -> Grid:
        """Computes the analysis.

        :param dependencies: The dependencies to use. Defaults to None.

        :returns: The computed analysis."""

        try:
            if dependencies is None:
                dependencies = []
            return self._analysis.compute(self._experiment, self._trackers, self._sequences, dependencies)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)


class AnalysisPartTask(object):
    """A task that computes a part of a separable analysis."""

    def __init__(
        self,
        analysis: SeparableAnalysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> None:
        """Initializes a new instance of the AnalysisPartTask class.

        :param analysis: The analysis to compute.
        :param experiment: The experiment to compute the analysis for.
        :param trackers: A single tracker or a list of trackers.
        :param sequences: A single sequence or a list of sequences.
        """
        self._analysis = analysis
        self._trackers: list[Tracker] = _as_list_trackers(trackers)
        self._experiment = experiment
        self._sequences: list[Sequence] = _as_list_sequences(sequences)
        self._key: HashKey = hashkey(analysis, experiment, unwrap(self._trackers), unwrap(self._sequences))

    def __call__(self, dependencies: list[Grid] | None = None) -> Grid:
        """Computes the analysis.

        :param dependencies: The dependencies to use. Defaults to None.

        :returns: The computed analysis."""
        try:
            if dependencies is None:
                dependencies = []
            return self._analysis.compute(self._experiment, self._trackers, self._sequences, dependencies)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)


class AnalysisJoinTask(object):
    """A task that joins the results of a separable analysis."""

    def __init__(
        self,
        analysis: SeparableAnalysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> None:
        """Initializes a new instance of the AnalysisJoinTask class.

        :param analysis: The analysis to join.
        :param experiment: The experiment to join the analysis for.
        :param trackers: A single tracker or a list of trackers.
        :param sequences: A single sequence or a list of sequences.
        """
        self._analysis = analysis
        self._trackers: list[Tracker] = _as_list_trackers(trackers)
        self._experiment = experiment
        self._sequences: list[Sequence] = _as_list_sequences(sequences)
        self._key: HashKey = hashkey(analysis, experiment, self._trackers, self._sequences)

    def __call__(self, results: list[Grid]) -> Grid:
        """Joins the results of the analysis.

        :param results: The results to join.

        :returns: The joined analysis."""

        try:
            return self._analysis.join(self._trackers, self._sequences, results)
        except BaseException as e:
            raise AnalysisError(cause=e, task=self._key)


class AnalysisFuture(Future):
    """A future that represents the result of an analysis."""

    def __init__(self, key: HashKey) -> None:
        """Initializes a new instance of the AnalysisFuture class.

        :param key: The :func:`hashkey` tuple of the analysis.
        """

        super().__init__()
        self.key: HashKey = key

    def __repr__(self) -> str:
        """Gets a string representation of the future."""
        return "<AnalysisFuture key={}>".format(self.key)

class AnalysisProcessor(object):
    """A processor that computes analyses."""

    _context = threading.local()

    def __init__(
        self,
        executor: Executor | None = None,
        cache: Cache | None = None,
    ) -> None:
        """Initializes a new instance of the AnalysisProcessor class.

        :param executor: The executor to use for computations. Defaults to a single-thread pool.
        :param cache: The cache to use for computations. Defaults to no caching.
        """
        if executor is None:
            executor = ThreadPoolExecutor(1)

        self._executor: ExecutorWrapper = ExecutorWrapper(executor)
        self._cache: Cache | None = cache
        self._pending: "bidict[HashKey, _CachedFuture]" = bidict()
        self._promises: dict[HashKey, list[AnalysisFuture]] = dict()
        self._lock = RLock()
        self._wait_condition = Condition()

    def commit(
        self,
        analysis: Analysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> AnalysisFuture:
        """Commits an analysis for computation. If the analysis is already being
        computed, the existing future is returned.

        :param analysis: The analysis to commit.
        :param experiment: The experiment to commit the analysis for.
        :param trackers: A single tracker or a list of trackers.
        :param sequences: A single sequence or a list of sequences.

        :returns: A future that represents the result of the analysis."""

        trackers_list = _as_list_trackers(trackers)
        sequences_list = _as_list_sequences(sequences)
        key: HashKey = hashkey(analysis, experiment, trackers_list, sequences_list)

        with self._lock:

            existing = self._exists(key)
            if existing is not None and analysis.cached:
                return existing

            promise = AnalysisFuture(key)
            promise.add_done_callback(self._promise_cancelled)

            dependencies: list[Future[Any]] = [
                self.commit(dependency, experiment, trackers_list, sequences_list)
                for dependency in analysis.dependencies()
            ]

            executorpromise: _CachedFuture

            if isinstance(analysis, SeparableAnalysis):

                def select_dependencies(
                    analysis: SeparableAnalysis,
                    tracker: int | None,
                    sequence: int | None,
                    *dependencies: Grid,
                ) -> list[Grid]:
                    """Selects the dependencies for a part of a separable analysis."""
                    return [
                        analysis.select(meta, data, tracker, sequence)
                        for meta, data in zip(analysis.dependencies(), dependencies)
                    ]

                promise = AnalysisFuture(key)
                promise.add_done_callback(self._promise_cancelled)

                parts = analysis.separate(trackers_list, sequences_list)
                partpromises: list[Future[Any]] = []

                for part in parts:
                    partkey: HashKey = hashkey(
                        analysis, experiment, unwrap(part.trackers), unwrap(part.sequences),
                    )

                    existing_part = self._exists(partkey)
                    if existing_part is not None and analysis.cached:
                        partpromises.append(existing_part)
                        continue

                    partpromise = AnalysisFuture(partkey)
                    partpromises.append(partpromise)

                    submitted = self._executor.submit(
                        AnalysisPartTask(analysis, experiment, part.trackers, part.sequences),
                        *dependencies,
                        mapping=partial(select_dependencies, analysis, part.tid, part.sid),
                    )
                    part_executor_promise = self._adopt_cached(submitted, analysis.cached)
                    self._promises[partkey] = [partpromise]
                    self._pending[partkey] = part_executor_promise
                    part_executor_promise.add_done_callback(self._future_done)

                submitted = self._executor.submit(
                    AnalysisJoinTask(analysis, experiment, trackers_list, sequences_list),
                    *partpromises,
                    mapping=lambda *x: [list(x)],
                )
                executorpromise = self._adopt_cached(submitted, analysis.cached)
                self._pending[key] = executorpromise
            else:
                task = AnalysisTask(analysis, experiment, trackers_list, sequences_list)
                submitted = self._executor.submit(task, *dependencies, mapping=lambda *x: [list(x)])
                executorpromise = self._adopt_cached(submitted, analysis.cached)
                self._pending[key] = executorpromise

            self._promises[key] = [promise]
            executorpromise.add_done_callback(self._future_done)
            logger.debug("Adding analysis task %s", key)

            return promise

    @staticmethod
    def _adopt_cached(future: Future[Any], cached: bool) -> _CachedFuture:
        """Attach the ``cached`` flag to ``future`` (which is the proxy returned by
        :class:`ExecutorWrapper.submit`). The proxy is a plain :class:`Future`; we
        treat it as a :class:`_CachedFuture` since that subclass differs only in
        carrying a typed attribute that consumers read via ``getattr``.
        """
        setattr(future, "cached", cached)
        return future  # type: ignore[return-value]

    def _exists(self, key: HashKey) -> AnalysisFuture | None:
        """Checks if an analysis is already being computed.

        :param key: The :func:`hashkey` tuple of the analysis to check.

        :returns: The future that represents the analysis if it is already being computed, None otherwise."""

        if self._cache is not None and key in self._cache:
            promise = AnalysisFuture(key)
            promise.set_result(self._cache[key])
            return promise

        if key in self._promises:
            promise = AnalysisFuture(key)
            promise.add_done_callback(self._promise_cancelled)
            self._promises[key].append(promise)
            return promise

        return None

    def _future_done(self, future: Future[Any]) -> None:
        """Handles the completion of a future.

        :param future: The future that completed.
        """

        # Every future we register against this callback is one we stored in
        # ``self._pending`` as a ``_CachedFuture``; the parameter type is widened
        # to ``Future`` because that's the contract of ``add_done_callback``.
        cached_future: _CachedFuture = future  # type: ignore[assignment]

        with self._lock:

            key = self._pending.inverse[cached_future]

            if future.cancelled():
                del self._pending[key]
                del self._promises[key]
                return

            result: Any = None
            error: BaseException | None = None
            try:
                result = future.result()
                if self._cache is not None and getattr(future, "cached", False):
                    self._cache[key] = result
            except (AnalysisError, RuntimeError) as e:
                error = e

            if key not in self._promises:
                return

            if error is None:
                for promise in self._promises[key]:
                    promise.set_result(result)
            else:
                for promise in self._promises[key]:
                    promise.set_exception(error)

            del self._promises[key]
            del self._pending[key]

            with self._wait_condition:
                self._wait_condition.notify()

    def _promise_cancelled(self, future: Future[Any]) -> bool:
        """Handles the cancellation of a promise. If it was the last promise for a
        computation, the computation is cancelled.

        :param future: The promise that was cancelled.

        :returns: True if the promise was tracked and removed, False otherwise."""

        if not future.cancelled():
            return False

        # ``future`` is always one of our ``AnalysisFuture`` proxies because this
        # callback is only registered against them. The cast keeps pyright happy.
        analysis_future: AnalysisFuture = future  # type: ignore[assignment]
        key = analysis_future.key

        with self._lock:

            if key not in self._promises:
                return False

            if analysis_future not in self._promises[key]:
                return False

            self._promises[key].remove(analysis_future)
            if len(self._promises[key]) == 0:
                self._pending[key].cancel()
            return True

    @property
    def pending(self) -> int:
        """The number of pending analyses."""

        with self._lock:
            return len(self._pending)

    @property
    def total(self) -> int:
        """The total number of analyses."""

        with self._lock:
            return self._executor.total

    def cancel(self) -> None:
        """Cancels all pending analyses."""

        with self._lock:
            for _, future in list(self._pending.items()):
                future.cancel()

    def wait(self) -> None:
        """Waits for all pending analyses to complete.

        If no analyses are pending, this method returns immediately.
        """

        if self.total == 0:
            return

        with Progress("Running analysis", self.total) as progress:
            try:

                while True:
                    progress.absolute(self.total - self.pending)
                    if self.pending == 0:
                        break

                    with self._wait_condition:
                        self._wait_condition.wait(1)

            except KeyboardInterrupt:
                self.cancel()
                progress.close()

    def __enter__(self) -> "AnalysisProcessor":
        """Sets this analysis processor as the default for the current thread.

        :returns: This analysis processor."""

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor == self:
            return self

        if processor is not None:
            logger.warning("Changing default processor for thread %s", threading.current_thread().name)

        AnalysisProcessor._context.analysis_processor = self
        logger.debug("Setting default analysis processor for thread %s", threading.current_thread().name)

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Clears the default analysis processor for the current thread."""

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor == self:
            AnalysisProcessor._context.analysis_processor = None
            self.cancel()

    @staticmethod
    def default() -> "AnalysisProcessor":
        """Returns the default analysis processor for the current thread.

        :returns: The default analysis processor for the current thread."""

        processor = getattr(AnalysisProcessor._context, 'analysis_processor', None)

        if processor is None:
            logger.debug(
                "Default analysis processor not set for thread %s, using a simple one.",
                threading.current_thread().name,
            )
            from vot.utilities import ThreadPoolExecutor as _ToolkitThreadPool
            from cachetools import LRUCache
            executor: Executor = _ToolkitThreadPool(1)
            cache: Cache = LRUCache(1000)
            processor = AnalysisProcessor(executor, cache)
            AnalysisProcessor._context.analysis_processor = processor

        return processor

    @staticmethod
    def commit_default(
        analysis: Analysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> AnalysisFuture:
        """Commits an analysis to the default analysis processor.

        This method is thread-safe. If the analysis is already being computed, this
        method returns immediately.
        """
        processor = AnalysisProcessor.default()
        return processor.commit(analysis, experiment, trackers, sequences)

    def run(
        self,
        analysis: Analysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> Grid:
        """Runs an analysis on a set of trackers and sequences. This method is thread-
        safe. If the analysis is already being computed, this method returns
        immediately.

        :param analysis: The analysis to run.
        :param experiment: The experiment to run the analysis on.
        :param trackers: A single tracker or a list of trackers.
        :param sequences: A single sequence or a list of sequences.

        :returns: The results of the analysis."""

        assert self.pending == 0

        future = self.commit(analysis, experiment, trackers, sequences)

        self.wait()

        return future.result()

    @staticmethod
    def run_default(
        analysis: Analysis,
        experiment: Experiment,
        trackers: TrackersArg,
        sequences: SequencesArg,
    ) -> Grid:
        """Runs an analysis on a set of trackers and sequences via the per-thread
        default processor."""
        processor = AnalysisProcessor.default()
        return processor.run(analysis, experiment, trackers, sequences)


def process_stack_analyses(
    workspace: "Workspace",
    trackers: TrackersArg,
    sequences: list[str] | None = None,
    experiments: list[str] | None = None,
) -> dict[Experiment, dict[Analysis, Grid | None]] | None:
    """Process all analyses in the workspace stack. This function is used by the command
    line interface to run all the analyses provided in a stack.

    :param workspace: The workspace to process.
    :param trackers: A single tracker or a list of trackers to run analyses on.
    :param sequences: Optional list of sequence names to filter to. ``None`` means all.
    :param experiments: Optional list of experiment identifiers to filter to. ``None`` means all.

    :returns: A nested dict mapping each :class:`Experiment` to a per-analysis result grid,
        or ``None`` if the run was interrupted or one or more analyses errored.
    """

    processor = AnalysisProcessor.default()

    results: dict[Experiment, dict[Analysis, Grid | None]] = dict()
    condition = Condition()
    errors: list[BaseException] = []

    def insert_result(
        container: dict[Analysis, Grid | None],
        key: Analysis,
    ) -> Callable[[Future[Any]], None]:
        """Creates a callback that inserts the result of a computation into a container.
        The container is a dictionary that maps analyses to their results.

        :param container: The container to insert the result into.
        :param key: The analysis whose result is being collected.
        """
        def insert(future: Future[Any]) -> None:
            """Inserts the result of a computation into a container."""
            try:
                container[key] = future.result()
            except Exception as e:
                errors.append(e)
            with condition:
                condition.notify()
        return insert

    trackers_list: list[Tracker] = _as_list_trackers(trackers)

    assert experiments is None or isinstance(experiments, list)
    assert sequences is None or isinstance(sequences, list)

    # Rebind to fresh, well-typed names — the previous code reused ``experiments`` /
    # ``sequences`` for both their string-filter input and their resolved list output,
    # which confused the type checker (and made the code harder to read).
    selected_experiments: list[Experiment] = list(workspace.stack) if experiments is None \
        else [e for e in workspace.stack if e.identifier in experiments]
    selected_sequences: list[Sequence] = list(workspace.dataset) if sequences is None \
        else [s for s in workspace.dataset if s.name in sequences]

    for experiment in selected_experiments:

        logger.debug("Traversing experiment %s", experiment.identifier)

        experiment_results: dict[Analysis, Grid | None] = dict()

        results[experiment] = experiment_results

        experiment_sequences = experiment.transform(experiment.select(selected_sequences))

        for analysis in experiment.analyses:

            if not analysis.compatible(experiment):
                continue

            logger.debug("Traversing analysis %s", class_fullname(analysis))

            with condition:
                experiment_results[analysis] = None
            promise = processor.commit(analysis, experiment, trackers_list, experiment_sequences)
            promise.add_done_callback(insert_result(experiment_results, analysis))

    if processor.total == 0:
        return results

    logger.debug("Waiting for %d analysis tasks to finish", processor.total)

    with Progress("Running analysis", processor.total) as progress:
        try:

            while True:

                progress.absolute(processor.total - processor.pending)
                if processor.pending == 0:
                    progress.absolute(processor.total)
                    break

                with condition:
                    condition.wait(1)

        except KeyboardInterrupt:
            processor.cancel()
            progress.close()
            logger.info("Analysis interrupted by user, aborting.")
            return None

    if len(errors) > 0:
        logger.info("Errors occured during analysis, incomplete.")
        for e in errors:
            task = getattr(e, "task", None)
            root_cause = getattr(e, "root_cause", e)
            logger.info("Failed task {}: {}".format(task, root_cause))
        return None

    return results