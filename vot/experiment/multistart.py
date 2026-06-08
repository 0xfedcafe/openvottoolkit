"""This module implements the multistart experiment."""

from typing import Callable

from attributee import String

from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import Special, SpecialCode
from vot.experiment import Experiment
from vot.tracker import ObjectQuery, Tracker, Trajectory


def find_anchors(sequence: Sequence, anchor: str = "anchor") -> tuple[list[int], list[int]]:
    """Find anchor frames in the sequence. Anchor frames are frames where the given
    object is visible and can be used for initialization.

    :param sequence: The sequence to be scanned.
    :param anchor: The name of the object to be used as an anchor. Defaults to ``"anchor"``.

    :returns: ``(forward_anchors, backward_anchors)`` — lists of frame indices."""
    forward: list[int] = []
    backward: list[int] = []
    for frame in range(len(sequence)):
        values = sequence.values(frame)
        if values is not None and anchor in values:
            if values[anchor] > 0:
                forward.append(frame)
            elif values[anchor] < 0:
                backward.append(frame)
    return forward, backward

class MultiStartExperiment(Experiment):
    """The multistart experiment. The experiment works by utilizing anchor frames in the
    sequence. Anchor frames are frames where the given object is visible and can be used
    for initialization.

    The tracker is then initialized in each anchor frame and run until the end of the
    sequence either forward or backward.

    This experiment assumes that anchor frames are labeled in the sequence with a
    specific value (default is "anchor") and that the value of the object is positive
    for forward anchors and negative for backward anchors. If no anchor information is
    present in the sequence, the experiment will fail with an error. The experiment can
    be run with or without supervision.
    """

    anchor = String(default="anchor")

    def _find_validated_anchors(self, sequence: Sequence) -> tuple[list[int], list[int]]:
        """Forward and backward anchor frames, raising if the sequence has none."""
        forward, backward = find_anchors(sequence, self.anchor)
        if len(forward) == 0 and len(backward) == 0:
            raise RuntimeError("Sequence does not contain any anchors")
        return forward, backward

    def scan(self, tracker: Tracker, sequence: Sequence) -> tuple:
        """Scan the results of the experiment for the given tracker and sequence.

        :param tracker: The tracker to be scanned.
        :type tracker: Tracker
        :param sequence: The sequence to be scanned.
        :type sequence: Sequence

        :returns: A tuple containing three elements. The first element is a boolean indicating whether the experiment is complete. The second element is a list of files that are present. The third element is the results object.
        :rtype: [tuple]"""
    
        files = []
        complete = True

        results = self.results(tracker, sequence)

        forward, backward = self._find_validated_anchors(sequence)

        for i in forward + backward:
            name = f"{sequence.name}_{i:08d}"
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False

        return complete, files, results

    def gather(self, tracker: Tracker, sequence: Sequence, objects=None, pad: bool = False) -> list[Trajectory | None]:
        """Gather anchor trajectories, remapped to absolute sequence coordinates.

        Each anchor run is stored over an anchor-relative proxy sequence (see :meth:`execute`),
        so its frame 0 is the anchor frame and it spans only part of the sequence. This method
        reads those per-anchor trajectories and maps every frame back to its position in the
        original ``sequence``, so consumers (speed analysis, preview videos) can treat the
        result like a regular full-length :class:`Trajectory`, the same way they treat the
        output of :meth:`MultiRunExperiment.gather`.

        :param tracker: The tracker whose results are gathered.
        :param sequence: The sequence to gather for.
        :param objects: Ignored; the multistart experiment is single-target. Accepted for
            signature compatibility with :meth:`MultiRunExperiment.gather`.
        :param pad: When set, missing anchor results are kept as ``None`` placeholders.

        :returns: One full-length trajectory per anchor, in original sequence coordinates."""
        del objects

        results = self.results(tracker, sequence)
        forward, backward = find_anchors(sequence, self.anchor)

        trajectories: list[Trajectory | None] = []

        for anchor, reverse in [(f, False) for f in forward] + [(b, True) for b in backward]:
            name = f"{sequence.name}_{anchor:08d}"
            if not Trajectory.exists(results, name):
                if pad:
                    trajectories.append(None)
                continue

            source = Trajectory.read(results, name)
            mapped = Trajectory(len(sequence))
            for frame in range(len(source)):
                # Proxy frame 0 is the anchor; forward runs advance, backward runs rewind.
                target = anchor - frame if reverse else anchor + frame
                if target < 0 or target >= len(sequence):
                    continue
                mapped.set(target, source.region(frame), source.properties(frame))
            trajectories.append(mapped)

        return trajectories

    def execute(
        self,
        tracker: Tracker,
        sequence: Sequence,
        force: bool = False,
        callback: Callable[[float], None] | None = None,
    ) -> None:
        """Execute the experiment for the given tracker and sequence.

        :param tracker: The tracker to be executed.
        :param sequence: The sequence to be executed.
        :param force: Force re-execution of the experiment. Defaults to False.
        :type force: bool, optional
        :param callback: A callback function that is called after each frame. Defaults to None.
        :type callback: Callable, optional

        :raises RuntimeError: If the sequence does not contain any anchors."""

        results = self.results(tracker, sequence)

        forward, backward = self._find_validated_anchors(sequence)

        total = len(forward) + len(backward)
        current = 0

        with self._get_runtime(tracker, sequence) as runtime:

            for i, reverse in [(f, False) for f in forward] + [(b, True) for b in backward]:
                name = f"{sequence.name}_{i:08d}"

                if Trajectory.exists(results, name) and not force:
                    continue

                if reverse:
                    proxy = FrameMapSequence(sequence, list(reversed(range(0, i + 1))))
                else:
                    proxy = FrameMapSequence(sequence, list(range(i, len(sequence))))

                # The proxy is already aligned so that its frame 0 is the anchor frame;
                # the query offset must be expressed in proxy coordinates (0), not in
                # the original sequence coordinates (``i``). Otherwise
                # ``OnlineTrackerRuntime.run`` filters the queries with ``offset == 0``
                # and passes an empty list to ``initialize``, which raises the
                # "multiple objects" exception for single-target trackers.
                #
                # ``ObjectQuery.state`` must be the Region itself, not the whole
                # ``ObjectStatus`` (see ``multirun.MultiRunExperiment`` for the same
                # convention) — otherwise the trax runtime gets a nested ObjectStatus
                # inside ``new`` and the trax protocol fails on the wrong type.
                init = self._get_initialization(proxy, 0)
                queries = [ObjectQuery(init.region, init.properties, 0)]
                status = runtime.run(proxy, queries)

                trajectory = Trajectory(len(proxy))

                # ``status.objects`` is indexed [query][frame], but ``status.times``
                # is a flat list with one entry per frame (see RunResult / multirun).
                init_properties = dict(status.objects[0][0].properties)
                init_properties["time"] = status.times[0]
                trajectory.set(0, Special(SpecialCode.INITIALIZATION), init_properties)
                for frame in range(1, len(proxy)):
                    frame_status = status.objects[0][frame]
                    frame_properties = dict(frame_status.properties)
                    frame_properties["time"] = status.times[frame]
                    trajectory.set(frame, frame_status.region, frame_properties)

                trajectory.write(results, name)

                current = current + 1
                if  callback:
                    callback(current / total)
