"""Unit tests for the experiment module.

Covers the multi-object helper, the sequence transformers (including the per-channel
image bug in :class:`Redetection`) and the :class:`UnsupervisedExperiment` run loop:
per-frame property preservation and the deterministic early-stop behaviour (the reason a
deterministic tracker is stored with three runs while a stochastic one keeps all
``repetitions``).
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np


class TestMultiObjectHelper(unittest.TestCase):
    """Tests for :class:`vot.experiment.helpers.MultiObjectHelper`."""

    class _FakeSequence:
        """Minimal sequence exposing only what ``MultiObjectHelper`` reads."""

        name = "fake"

        def __init__(self, trajectories: dict) -> None:
            self._trajectories = trajectories

        def objects(self, index=None):
            return list(self._trajectories.keys())

        def object(self, oid, index=None):
            return self._trajectories.get(oid)

    def _make_helper(self):
        from vot.dataset import Sequence
        from vot.experiment.helpers import MultiObjectHelper
        from vot.region import Rectangle, Special, SpecialCode

        hidden = Special(SpecialCode.UNKNOWN)
        box = Rectangle(0, 0, 4, 4)
        # "a" visible from frame 0, "b" appears at frame 2.
        sequence = self._FakeSequence({
            "a": [box, box, box, box],
            "b": [hidden, hidden, box, box],
        })
        return MultiObjectHelper(cast(Sequence, sequence))

    def test_new_reports_first_visible_frame(self):
        helper = self._make_helper()
        self.assertEqual(helper.new(0), ["a"])
        self.assertEqual(helper.new(1), [])
        self.assertEqual(helper.new(2), ["b"])

    def test_objects_active_at_frame(self):
        helper = self._make_helper()
        self.assertEqual(helper.objects(0), ["a"])
        self.assertEqual(helper.objects(2), ["a", "b"])

    def test_all_objects(self):
        self.assertEqual(self._make_helper().all(), ["a", "b"])

    def test_never_visible_object_raises(self):
        from vot.dataset import Sequence
        from vot.experiment.helpers import MultiObjectHelper
        from vot.region import RegionException, Special, SpecialCode

        sequence = self._FakeSequence({"a": [Special(SpecialCode.UNKNOWN)] * 3})
        with self.assertRaises(RegionException):
            MultiObjectHelper(cast(Sequence, sequence))


class _ExperimentTempCase(unittest.TestCase):
    """Base case providing a fresh temporary storage directory per test."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vot_experiment_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestTransformers(_ExperimentTempCase):
    """Tests for the experiment transformers."""

    def _make_rgb_multichannel(self, length: int = 4, size=(64, 48)):
        """Builds an on-disk two-channel (color, ir) sequence with distinct RGB content
        per channel, so a transformer that confuses channels is detectable."""
        import cv2

        from vot.dataset import load_sequence
        from vot.utilities import write_properties

        width, height = size
        out = self.tmp / "multi"
        # OpenCV writes BGR; red is (0, 0, 200) and blue is (200, 0, 0) in BGR.
        for channel, bgr in (("color", (0, 0, 200)), ("ir", (200, 0, 0))):
            cdir = out / channel
            cdir.mkdir(parents=True, exist_ok=True)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = bgr
            for i in range(1, length + 1):
                cv2.imwrite(str(cdir / "{:08d}.jpg".format(i)), frame)

        (out / "groundtruth.txt").write_text("\n".join("10,10,8,8" for _ in range(length)) + "\n")
        write_properties(str(out / "sequence"), {
            "channels.color": "color/%08d.jpg", "channels.ir": "ir/%08d.jpg",
            "width": width, "height": height, "length": length,
            "fps": 30, "format": "default", "channel.default": "color",
        })
        return load_sequence(str(out))

    def test_redetection_uses_per_channel_images(self):
        """Each generated channel must carry its own image, not the default channel's.

        Regression test: ``Redetection`` previously called ``frame.image()`` without a
        channel argument, so every channel got the default (color) pixels.
        """
        from vot.experiment.transformer import Redetection
        from vot.workspace.storage import LocalStorage

        source = self._make_rgb_multichannel()
        cache = LocalStorage(str(self.tmp / "cache"))

        result = Redetection(cache=cache)(source)[0]
        frame = result.frame(0)
        color = frame.image("color")
        ir = frame.image("ir")

        assert color is not None and ir is not None
        # Distinct channels => distinct pixels. The bug made these identical.
        self.assertFalse(np.array_equal(color, ir))
        self.assertGreater(int(color[..., 0].mean()), int(ir[..., 0].mean()))  # red vs blue

    def test_single_object_splits_multi_object_sequence(self):
        from vot.experiment.transformer import SingleObject
        from vot.dataset.dummy import generate_dummy

        sequence = generate_dummy(length=4, objects=3)
        result = SingleObject(cache=None)(sequence)
        self.assertEqual(len(result), 3)

    def test_single_object_passthrough_for_single_object(self):
        from vot.experiment.transformer import SingleObject
        from vot.dataset.dummy import generate_dummy

        sequence = generate_dummy(length=4, objects=1)
        result = SingleObject(cache=None)(sequence)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], sequence)

    def test_downsample_keeps_every_factor_th_frame(self):
        from vot.experiment.transformer import Downsample
        from vot.dataset.dummy import generate_dummy

        sequence = generate_dummy(length=6, objects=1)
        result = Downsample(cache=None, factor=2)(sequence)[0]
        self.assertEqual(len(result), 3)  # frames 0, 2, 4


class _StubRuntime:
    """In-process tracker runtime that echoes deterministic per-frame results.

    Each frame gets a distinct region and a ``confidence`` property so tests can check
    that per-frame properties survive the experiment's run loop.
    """

    multiobject = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def restart(self) -> None:
        pass

    def run(self, frames, queries):
        from vot.tracker import RunResult, ObjectStatus
        from vot.region import Rectangle

        count = len(frames)
        statuses = [
            ObjectStatus(Rectangle(f, f, 5, 5), {"confidence": 0.5 + 0.1 * f})
            for f in range(count)
        ]
        times = [float(f) + 1.0 for f in range(count)]
        return RunResult([statuses], times)


class _StubTracker:
    """Minimal tracker exposing what the experiment and storage layers touch."""

    identifier = "stub"
    reference = "stub"
    storage = None

    def runtime(self, log: bool = False):
        return _StubRuntime()


class TestUnsupervisedExperiment(_ExperimentTempCase):
    """Tests for :class:`vot.experiment.multirun.UnsupervisedExperiment`."""

    def _make_experiment(self, **kwargs):
        from vot.experiment.multirun import UnsupervisedExperiment
        from vot.workspace.storage import LocalStorage

        storage = LocalStorage(str(self.tmp / "workspace"))
        return UnsupervisedExperiment("unsupervised", storage, **kwargs)

    def _stored_runs(self, experiment, tracker, sequence) -> int:
        from vot.tracker import Trajectory

        results = experiment.results(tracker, sequence)
        return sum(1 for i in range(1, 50)
                   if Trajectory.exists(results, "{}_{:03d}".format(sequence.name, i)))

    def test_preserves_per_frame_properties(self):
        """The run loop must keep tracker-reported properties (confidence) and only
        overwrite ``time`` with the accumulated total."""
        from vot.dataset.dummy import generate_dummy
        from vot.tracker import Tracker, Trajectory

        sequence = generate_dummy(length=4, objects=1)
        tracker = cast(Tracker, _StubTracker())
        experiment = self._make_experiment(repetitions=1)

        experiment.execute(tracker, sequence)

        results = experiment.results(tracker, sequence)
        trajectory = Trajectory.read(results, "{}_001".format(sequence.name))

        properties = trajectory.properties(1)
        self.assertIn("confidence", properties)
        self.assertIn("time", properties)
        self.assertAlmostEqual(properties["confidence"], 0.6, places=6)
        self.assertAlmostEqual(properties["time"], 2.0, places=6)  # accumulated times[1]

    def test_early_stop_stops_after_deterministic_runs(self):
        """A deterministic tracker is stored with three identical runs, then stopped."""
        from vot.dataset.dummy import generate_dummy

        from vot.tracker import Tracker

        sequence = generate_dummy(length=4, objects=1)
        tracker = cast(Tracker, _StubTracker())
        experiment = self._make_experiment(repetitions=5, early_stop=True)

        experiment.execute(tracker, sequence)

        self.assertEqual(self._stored_runs(experiment, tracker, sequence), 3)

    def test_early_stop_disabled_stores_all_repetitions(self):
        from vot.dataset.dummy import generate_dummy
        from vot.tracker import Tracker

        sequence = generate_dummy(length=4, objects=1)
        tracker = cast(Tracker, _StubTracker())
        experiment = self._make_experiment(repetitions=5, early_stop=False)

        experiment.execute(tracker, sequence)

        self.assertEqual(self._stored_runs(experiment, tracker, sequence), 5)


if __name__ == "__main__":
    unittest.main()
