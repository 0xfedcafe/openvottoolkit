"""Multi-run experiments.

This module contains the implementation of multi-run experiments. Multi-run experiments
are used to run a tracker multiple times on the same sequence.
"""
import logging
from typing import Callable, Literal, overload

from attributee import Boolean, Integer, Float, List, String

from vot.experiment import Experiment
from vot.tracker import Tracker, Trajectory, ObjectQuery, ObjectStatus, OnlineTrackerRuntime, TrackerException
from vot.dataset import Sequence
from vot.region import Special, SpecialCode, calculate_overlap
from vot.dataset.proxy import FrameMapSequence

logger = logging.getLogger("vot")

class MultiRunExperiment(Experiment):
    """Base class for multi-run experiments.

    Multi-run experiments run a tracker ``repetitions`` times on the same sequence.

    .. important::
        **Why a tracker may end up with fewer stored runs than ``repetitions``.**
        When ``early_stop`` is enabled (the default), the experiment stops repeating a
        sequence as soon as the tracker proves *deterministic*: once
        :data:`_EARLY_STOP_RUNS` (3) repetitions are stored and they are all identical,
        the remaining repetitions are skipped because they would only reproduce the same
        trajectory. So a deterministic tracker is stored with **3** runs while a
        stochastic one (or any tracker whose runs differ) is stored with the full
        ``repetitions`` (e.g. 10). This is expected and not a missing-results bug — the
        scan/analysis code treats an early-stopped sequence as complete. Set
        ``early_stop=False`` to always store all ``repetitions`` runs.
    """

    repetitions = Integer(val_min=1, default=1)
    early_stop = Boolean(default=True)

    #: Number of stored identical repetitions that prove a tracker is deterministic; once
    #: this many runs agree there is no point running the remaining repetitions.
    _EARLY_STOP_RUNS = 3

    def _can_stop(self, tracker: Tracker, sequence: Sequence) -> bool:
        """Check whether the experiment can be stopped early.

        Early stopping kicks in once a tracker proves deterministic: at least
        :data:`_EARLY_STOP_RUNS` repetitions are already stored for every object and they
        all agree, so the remaining repetitions would only reproduce the same trajectory.

        :param tracker: The tracker to be checked.
        :param sequence: The sequence to be checked.

        :returns: True if the experiment can be stopped early, False otherwise."""
        if not self.early_stop:
            return False

        for o in sequence.objects():

            trajectories = self.gather(tracker, sequence, objects=[o])
            if len(trajectories) < self._EARLY_STOP_RUNS:
                return False

            for trajectory in trajectories[1:]:
                if not trajectory.equals(trajectories[0]):
                    return False

        return True

    def _check_multiobject(self, sequence: Sequence) -> bool:
        """Whether ``sequence`` has multiple objects, asserting the experiment allows it."""
        multiobject = len(sequence.objects()) > 1
        assert self._multiobject or not multiobject
        return multiobject

    @staticmethod
    def _result_name(sequence: Sequence, object_id: str, repetition: int, multiobject: bool) -> str:
        """Name of the stored result file for one object/repetition."""
        if multiobject:
            return f"{sequence.name}_{object_id}_{repetition:03d}"
        return f"{sequence.name}_{repetition:03d}"

    def scan(self, tracker: Tracker, sequence: Sequence) -> tuple:
        """Scan the results of the experiment for the given tracker and sequence.

        :param tracker: The tracker to be scanned.
        :type tracker: Tracker
        :param sequence: The sequence to be scanned.
        :type sequence: Sequence

        :returns: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object.
        :rtype: [tuple]"""
        
        results = self.results(tracker, sequence)

        files = []
        complete = True
        multiobject = self._check_multiobject(sequence)

        for o in sequence.objects():
            for i in range(1, self.repetitions+1):
                name = self._result_name(sequence, o, i, multiobject)
                if Trajectory.exists(results, name):
                    files.extend(Trajectory.gather(results, name))
                elif self._can_stop(tracker, sequence):
                    break
                else:
                    complete = False
                    break

        return complete, files, results

    @overload
    def gather(self, tracker: Tracker, sequence: Sequence,
               objects: list[str] | None = ..., pad: Literal[False] = ...) -> list[Trajectory]: ...
    @overload
    def gather(self, tracker: Tracker, sequence: Sequence,
               objects: list[str] | None, pad: Literal[True]) -> list[Trajectory | None]: ...
    def gather(self, tracker: Tracker, sequence: Sequence,
               objects: list[str] | None = None,
               pad: bool = False) -> list[Trajectory] | list[Trajectory | None]:
        """Gather trajectories for the given tracker and sequence.

        Without ``pad`` only the stored repetitions are returned, so the result holds no
        ``None`` entries (see the overloads). With ``pad`` each missing repetition keeps a
        ``None`` placeholder so the result aligns with ``range(1, repetitions + 1)``.

        :param tracker: The tracker to be used.
        :param sequence: The sequence to be used.
        :param objects: The list of objects to be gathered. Defaults to all objects.
        :param pad: Whether to pad missing trajectories with ``None``. Defaults to False.

        :returns: The list of trajectories (with ``None`` placeholders when ``pad`` is True)."""
        trajectories: list[Trajectory | None] = list()

        multiobject = self._check_multiobject(sequence)
        results = self.results(tracker, sequence)

        if objects is None:
            objects = list(sequence.objects())

        for o in objects:
            for i in range(1, self.repetitions+1):
                name = self._result_name(sequence, o, i, multiobject)
                if Trajectory.exists(results, name):
                    trajectories.append(Trajectory.read(results, name))
                elif pad:
                    trajectories.append(None)
        return trajectories
    
    def execute(
        self,
        tracker: Tracker,
        sequence: Sequence,
        force: bool = False,
        callback: Callable[[float], None] | None = None,
    ) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")

class UnsupervisedExperiment(MultiRunExperiment):
    """Unsupervised experiment.

    This experiment is used to run a tracker multiple times on the same sequence without
    any supervision.
    """

    multiobject = Boolean(default=False)

    @property
    def _multiobject(self) -> bool:
        """Whether the experiment is multi-object or not.

        :returns: True if the experiment is multi-object, False otherwise.
        :rtype: bool"""
        return self.multiobject

    def execute(
        self,
        tracker: Tracker,
        sequence: Sequence,
        force: bool = False,
        callback: Callable[[float], None] | None = None,
    ) -> None:
        """Execute the experiment for the given tracker and sequence.

        :param tracker: The tracker to be used.
        :param sequence: The sequence to be used.
        :param force: Whether to force the execution. Defaults to False.
        :param callback: Optional callback called with progress in [0, 1] per repetition.
        """

        from .helpers import MultiObjectHelper

        results = self.results(tracker, sequence)

        multiobject = self._check_multiobject(sequence)

        helper = MultiObjectHelper(sequence)

        # Generate object queries for all objects in the sequence
        queries = []
        queries_keys = []
        for i in range(len(sequence)):
            for o in helper.new(i):
                state = self._get_initialization(sequence, i, o)
                queries.append(ObjectQuery(state.region, state.properties, i))
                queries_keys.append(o)

        with self._get_runtime(tracker, sequence, self._multiobject) as runtime:

            for i in range(1, self.repetitions+1):

                trajectories = {}

                times = []

                for o in helper.all(): 
                    trajectories[o] = Trajectory(len(sequence))

                if all([Trajectory.exists(results, self._result_name(sequence, o, i, multiobject)) for o in trajectories.keys()]) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                if runtime.multiobject:
                    
                    status = runtime.run(sequence, queries)
                    
                    for o, key in enumerate(queries_keys):
                        trajectories[key].set(0, Special(SpecialCode.INITIALIZATION), status.objects[o][0].properties)
                    for frame in range(1, len(sequence)):
                        for o, key in enumerate(queries_keys):
                            trajectories[key].set(frame, status.objects[o][frame].region, status.objects[o][frame].properties)

                    times = status.times

                    if callback:
                        callback(float(i) / self.repetitions)
                        
                else:
                    
                    times = [0] * len(sequence)
                    
                    for q, query in enumerate(queries):
                        offset = query.offset
                        
                        proxy = FrameMapSequence(sequence, list(range(offset, len(sequence))))
                        status = runtime.run(proxy, [ObjectQuery(query.state, query.properties, 0)])
                        
                        trajectory = trajectories[queries_keys[q]]
                        trajectory.set(offset, Special(SpecialCode.INITIALIZATION), status.objects[0][0].properties)
                        for frame in range(1, len(proxy)):
                            trajectory.set(frame + offset, status.objects[0][frame].region, status.objects[0][frame].properties)
                            times[frame + offset] += status.times[frame]
                            
                        if callback:
                            callback((float(i-1) / self.repetitions) + \
                                    (float(q) / (self.repetitions * len(trajectories))))
                        
                for o, trajectory in trajectories.items():
                    # Update only the time property with the trajectory's accumulated
                    # total. ``Trajectory.set`` merges properties per key, so the
                    # tracker-reported per-frame properties (e.g. confidence) set during
                    # the run above are preserved.
                    for frame in range(len(sequence)):
                        trajectory.set(frame, trajectory.region(frame), {"time": times[frame]})

                    trajectory.write(results, self._result_name(sequence, o, i, multiobject))


class SupervisedExperiment(MultiRunExperiment):
    """Supervised experiment. This experiment is used to run a tracker multiple times on
    the same sequence with supervision (reinitialization in case of failure).

    Due to the nature of the experiment, it requires online tracker runtimes and only
    works on single-target sequences. In all other cases the execution will fail with an
    error.

    When ``recover`` is enabled a tracker error (a crash or a runtime timeout) is handled
    rather than aborting the whole evaluation: the runtime is restarted with a fresh
    process and the experiment reinitializes after the usual ``skip_initialize`` burn-in
    period. The crash frame is recorded with ``SpecialCode.CRASH`` so that downstream
    analysis can tell process failures apart from legitimate tracking failures
    (``SpecialCode.FAILURE``) — both terminate a tracking run, but only crashes feed the
    ``Crashes`` measure and process-reliability accounting.

    ``recover_limit`` bounds how many such errors may happen back-to-back (without a
    single frame handled in between) before the sequence is given up on, which keeps a
    fundamentally broken tracker from crawling a whole sequence one timeout at a time;
    ``0`` disables the bound.
    """

    skip_initialize = Integer(val_min=1, default=1)
    skip_tags = List(String(), default=[])
    failure_overlap = Float(val_min=0, val_max=1, default=0)
    recover = Boolean(default=True)
    recover_limit = Integer(val_min=0, default=0)

    def execute(
        self,
        tracker: Tracker,
        sequence: Sequence,
        force: bool = False,
        callback: Callable[[float], None] | None = None,
    ) -> None:
        """Execute the experiment for the given tracker and sequence.

        :param tracker: The tracker to be used.
        :param sequence: The sequence to be used.
        :param force: Whether to force the execution. Defaults to False.
        :param callback: Optional callback called with progress in [0, 1] per repetition.
        """

        if len(sequence.objects()) != 1:
            raise ValueError("SupervisedExperiment only works on single-target sequences.")

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            if not isinstance(runtime, OnlineTrackerRuntime):
                raise ValueError("SupervisedExperiment requires an online tracker runtime.")

            for i in range(1, self.repetitions + 1):
                name = self._result_name(sequence, "", i, multiobject=False)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(len(sequence))

                def advance_after_failure(frame: int) -> int:
                    """Frame index to reinitialize at after a failure: skip the
                    ``skip_initialize`` burn-in period and any skip-tagged frames."""
                    frame = frame + self.skip_initialize
                    if self.skip_tags:
                        while frame < len(sequence):
                            if not [t for t in sequence.tags(frame) if t in self.skip_tags]:
                                break
                            frame = frame + 1
                    return frame

                # Counts tracker errors (crashes/timeouts) that happened back-to-back
                # without a single frame handled in between; reset on any successful
                # runtime call. Used to bound recovery via ``recover_limit``.
                consecutive_errors = 0

                def recover_from_error(frame: int, te: TrackerException, phase: str) -> int:
                    """Handle a tracker error: restart the runtime, mark a failure and
                    return the frame index to continue from. Re-raises when recovery is
                    disabled or the consecutive-error bound is exceeded."""
                    nonlocal consecutive_errors
                    if not self.recover:
                        raise te
                    consecutive_errors += 1
                    logger.warning(
                        "Tracker %s error while %s sequence %s at frame %d: %s; recovering after %d frames",
                        tracker.identifier, phase, sequence.name, frame, te, self.skip_initialize
                    )
                    if self.recover_limit and consecutive_errors > self.recover_limit:
                        raise TrackerException(
                            f"Tracker exceeded the recovery limit of {self.recover_limit} consecutive errors",
                            te, tracker=tracker
                        )
                    runtime.restart()
                    trajectory.set(frame, Special(SpecialCode.CRASH), {"time": 0})
                    return advance_after_failure(frame)

                frame = 0
                while frame < len(sequence):

                    try:
                        _, elapsed = runtime.initialize(sequence.frame(frame), self._get_initialization(sequence, frame))
                    except TrackerException as te:
                        frame = recover_from_error(frame, te, "initializing")
                        continue
                    consecutive_errors = 0

                    trajectory.set(frame, Special(SpecialCode.INITIALIZATION), {"time": elapsed})

                    frame = frame + 1

                    while frame < len(sequence):

                        try:
                            # ``runtime.update`` returns the same shape it was called with:
                            # a single ``ObjectStatus`` here (no ``new`` argument supplied,
                            # so the wrapper takes the legacy single-object path).
                            update_result, elapsed = runtime.update(sequence.frame(frame))
                        except TrackerException as te:
                            frame = recover_from_error(frame, te, "tracking")
                            break
                        consecutive_errors = 0
                        target = _ensure_single_status(update_result)

                        target.properties["time"] = elapsed

                        gt = sequence.groundtruth(frame)
                        if gt is None or calculate_overlap(target.region, gt, sequence.size) <= self.failure_overlap:
                            trajectory.set(frame, Special(SpecialCode.FAILURE), target.properties)
                            frame = advance_after_failure(frame)
                            break
                        trajectory.set(frame, target.region, target.properties)
                        frame = frame + 1

                if callback:
                    callback(i / self.repetitions)

                trajectory.write(results, name)


def _ensure_single_status(status: object) -> ObjectStatus:
    """Narrows the polymorphic ``ObjectStatus | list[ObjectStatus]`` return shape
    of single-target runtime calls to a single :class:`ObjectStatus`. The
    underlying wrappers (trax/python) unwrap length-1 lists at runtime; this
    helper exists to make the narrowing visible to the type checker.
    """
    if isinstance(status, list):
        if len(status) != 1:
            raise RuntimeError(
                f"Single-target runtime call returned {len(status)} objects; expected 1"
            )
        return status[0]
    assert status is not None, "Single-target runtime call returned None"
    assert isinstance(status, ObjectStatus)
    return status
