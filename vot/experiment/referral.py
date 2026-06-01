"""Referral experiment — text-prompt based tracking initialization."""

from hashlib import md5
from typing import Any, Callable

from attributee import String

from vot.experiment import Experiment
from vot.tracker import OnlineTrackerRuntime, Tracker, Trajectory, ObjectQuery
from vot.dataset import Sequence
from vot.region import Special, SpecialCode


class ReferralExperiment(Experiment):
    """Experiment with text referral initialization.

    The experiment does not provide an initialization region, but a text prompt that
    describes the object to be tracked. The tracker is then expected to use this prompt
    to find the object in the sequence.
    """

    prompt_name = String(default="prompts")

    @property
    def _multiobject(self) -> bool:
        """Prevent SingleObject transformer from splitting sequences with ignore objects."""
        return True

    def _extract_prompt(self, sequence: Sequence) -> list[str]:
        # ``sequence.metadata`` returns ``object`` per the abstract contract;
        # narrow to ``str`` here so ``.split`` is type-safe.
        prompts = sequence.metadata(self.prompt_name, "")
        if not isinstance(prompts, str) or not prompts:
            raise ValueError(f"Sequence {sequence.name} does not contain any prompts.")
        return prompts.split(";")

    def scan(self, tracker: Tracker, sequence: Sequence) -> tuple[bool, list[Any], Any]:
        """Scan the results of the experiment for the given tracker and sequence.

        :param tracker: The tracker to be scanned.
        :param sequence: The sequence to be scanned.

        :returns: ``(complete, files, results)`` — completeness flag, list of present
            files, and the results object."""

        results = self.results(tracker, sequence)

        files: list[Any] = []
        complete = True
        assert len(sequence.objects()) == 1, "Referral experiment only supports single object sequences."

        prompts = self._extract_prompt(sequence)

        for prompt in prompts:
            prompt_hash = md5(prompt.encode()).hexdigest()[:8]
            name = f"{sequence.name}_{prompt_hash}"
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            else:
                complete = False
                break

        return complete, files, results

    def gather(
        self,
        tracker: Tracker,
        sequence: Sequence,
        objects: list[str] | None = None,
        pad: bool = False,
    ) -> list[Trajectory | None]:
        """Gather trajectories for the given tracker and sequence.

        :param tracker: The tracker to be used.
        :param sequence: The sequence to be used.
        :param objects: Ignored — referral is single-object — kept for signature compatibility.
        :param pad: When ``True`` insert ``None`` placeholders for missing trajectories.

        :returns: List of trajectories (with ``None`` placeholders when ``pad`` is True)."""
        del objects  # referral is single-object; the param exists for API parity only.
        trajectories: list[Trajectory | None] = []

        assert len(sequence.objects()) == 1, "Referral experiment only supports single object sequences."

        prompts = self._extract_prompt(sequence)

        results = self.results(tracker, sequence)

        for prompt in prompts:
            prompt_hash = md5(prompt.encode()).hexdigest()[:8]
            name = f"{sequence.name}_{prompt_hash}"
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
        prompts = self._extract_prompt(sequence)

        assert len(sequence.objects()) == 1, "Referral experiment only supports single object sequences."

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            if not isinstance(runtime, OnlineTrackerRuntime):
                raise ValueError("ReferralExperiment requires an online tracker runtime.")

            for prompt_index, prompt in enumerate(prompts):

                # Check if trajectory already exists
                prompt_hash = md5(prompt.encode()).hexdigest()[:8]
                name = f"{sequence.name}_{prompt_hash}"
                if Trajectory.exists(results, name) and not force:
                    continue

                queries = [ObjectQuery(Special(SpecialCode.UNKNOWN), {"prompt": prompt}, 0)]
                status = runtime.run(sequence, queries)

                trajectory = Trajectory(len(sequence))
                for frame_index in range(len(sequence)):
                    region = status.objects[0][frame_index].region
                    trajectory.set(frame_index, region)

                trajectory.write(results, name)

                if callback is not None:
                    callback(prompt_index / len(prompts))