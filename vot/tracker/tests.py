"""Unit tests for the tracker module."""

import unittest
from typing import Any

import numpy as np
import matplotlib.pylab as plt

from vot.utilities.draw import MatplotlibDrawHandle
from vot.region import Region
from vot.dataset import Sequence, load_sequence, Frame
from vot.dataset.dummy import generate_dummy
from vot.tracker import ObjectStatus, Tracker, TrackerRuntime, \
    TrackerException, ObjectQuery, OnlineTrackerRuntime, FrameObjects, Registry

from vot.experiment.helpers import MultiObjectHelper
from vot.dataset.proxy import ObjectsHideFilterSequence
from vot import get_logger
from vot.utilities import normalize_path

class TestStacks(unittest.TestCase):
    """Tests for the stacks module."""

    def test_trax_tracker_test(self) -> None:
        """Test tracker runtime with dummy sequence and dummy tracker."""
        from vot.tracker.dummy import DummyTraxTracker

        tracker = DummyTraxTracker
        sequence = generate_dummy(10)

        with tracker.runtime(log=False) as runtime:
            # The dummy trax tracker uses the online protocol — narrow so the
            # type checker accepts ``initialize`` / ``update`` access.
            assert isinstance(runtime, OnlineTrackerRuntime)
            gt = sequence.groundtruth(0)
            assert gt is not None
            runtime.initialize(sequence.frame(0), [ObjectStatus(gt, {})])
            for i in range(1, len(sequence)):
                runtime.update(sequence.frame(i))

    def test_folder_tracker_test(self) -> None:
        """Test folder tracker with dummy sequence and dummy tracker."""
        from vot.tracker.dummy import DummyFolderTracker

        tracker = DummyFolderTracker
        sequence = generate_dummy(10)

        with tracker.runtime(log=False) as runtime:
            gt = sequence.groundtruth(0)
            assert gt is not None
            queries = [ObjectQuery(gt, {}, 0)]
            runtime.run(sequence, queries=queries)

def run_tracker_test(runtime: TrackerRuntime, visualize: bool = False, sequence: str | None = None, ignore: list[str] | None = None) -> None:
    """Run a test for a tracker."""

    logger = get_logger()

    handle: MatplotlibDrawHandle | None = None
    figure = None
    axes = None

    def visualize_state(axes, frame: Frame, reference: list[Region | None], state: FrameObjects) -> None:
        """Visualize the frame and the state of the tracker."""
        assert handle is not None, "visualize_state() called without an initialised draw handle"
        axes.clear()
        image = frame.image()
        assert image is not None, "frame.image() returned None"
        handle.image(image)
        if not isinstance(state, list):
            state = [state]
        for gt, st in zip(reference, state):
            if gt is not None:
                handle.style(color="green").region(gt)
            handle.style(color="red").region(st.region)

    try:

        # Coerce the optional ``sequence`` parameter (a path or None) into a
        # concrete ``Sequence`` and use a separate variable so the type checker
        # can keep the parameter type clean.
        seq: Sequence
        if sequence is None:
            logger.info("Generating dummy sequence")
            seq = generate_dummy(50, objects=3 if runtime.multiobject else 1)
        else:
            logger.info(f"Loading sequence from {sequence}")
            seq = load_sequence(normalize_path(sequence))

        if ignore:
            seq = ObjectsHideFilterSequence(seq, set(ignore))

        context: dict[str, Any] = {"continue": True}

        def on_press(event) -> None:
            """Callback for key press event."""
            if event.key == 'q':
                context["continue"] = False

        if visualize:

            figure = plt.figure()
            canvas = figure.canvas
            # ``set_window_title`` only exists on some matplotlib backends; the
            # ``getattr`` guard keeps the call duck-typed and silences the
            # attribute-access diagnostic.
            set_title = getattr(canvas, "set_window_title", None)
            if callable(set_title):
                set_title('VOT Test')
            axes = figure.add_subplot(1, 1, 1)
            axes.set_aspect("equal")
            handle = MatplotlibDrawHandle(axes, size=seq.size)
            context["click"] = canvas.mpl_connect('key_press_event', on_press)
            handle.style(fill=False)
            figure.show()

        helper = MultiObjectHelper(seq)

        if isinstance(runtime, OnlineTrackerRuntime):

            logger.info("Initializing tracker")

            frame = seq.frame(0)
            state, _ = runtime.initialize(frame, [ObjectStatus(r, {}) for x in helper.new(0) if (r := frame.object(x)) is not None])

            if visualize:
                assert axes is not None and figure is not None
                visualize_state(axes, frame, [frame.object(x) for x in helper.objects(0)], state)
                figure.canvas.draw()

            for i in range(1, len(seq)):

                logger.info(f"Processing frame {i}/{len(seq)-1}")
                frame = seq.frame(i)
                state, _ = runtime.update(frame, [ObjectStatus(r, {}) for x in helper.new(i) if (r := frame.object(x)) is not None])

                if visualize:
                    assert axes is not None and figure is not None
                    visualize_state(axes, frame, [frame.object(x) for x in helper.objects(i)], state)
                    figure.canvas.draw()
                    figure.canvas.flush_events()

                if not context["continue"]:
                    break

            logger.info("Stopping tracker")

            runtime.stop()

            logger.info("Test concluded successfully")


        else:

            # Run tracker in batch mode

            queries = []
            for i in range(len(seq)):
                frame = seq.frame(i)
                queries.extend([ObjectQuery(r, {}, i) for x in helper.new(i) if (r := frame.object(x)) is not None])

            status = runtime.run(seq, queries=queries)

            # Visualize results offline if requested
            if visualize:
                logger.info("Visualizing results")
                for i in range(len(seq)):
                    assert axes is not None and figure is not None
                    frame = seq.frame(i)
                    state = [obj[i] for obj in status.objects]
                    visualize_state(axes, frame, [frame.object(x) for x in helper.objects(i)], state)
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    if not context["continue"]:
                        break

    except TrackerException as te:
        logger.error(f"Error during tracker execution: {te}")
        if runtime:
            runtime.stop()
    except KeyboardInterrupt:
        if runtime:
            runtime.stop()


class TestTrackerAdapters(unittest.TestCase):
    """Tests for the Matlab and Octave tracker adapters.

    These exercise the command and script generation only — they require no
    Matlab or Octave installation, since the runtime constructor is replaced
    by :class:`_RecordingRuntime`, which just captures its arguments.
    """

    class _RecordingRuntime(TrackerRuntime):
        """A no-op tracker runtime that records the arguments used to build it.

        It satisfies the ``Callable[..., TrackerRuntime]`` contract the adapters
        expect of their ``constructor`` while never launching a process."""

        def __init__(self, tracker: Tracker, command: str, **kwargs: Any) -> None:
            super().__init__(tracker)
            self.command: str = command
            self.kwargs: dict[str, Any] = kwargs

        def stop(self) -> None:
            """No process to stop."""

    def _tracker(self) -> Tracker:
        """Builds a minimal real tracker rooted at ``/trackers``.

        The adapters only read ``tracker.source`` (to resolve relative paths),
        so a bare manifest-less tracker is enough."""
        return Tracker(_identifier="dummy", _source="/trackers/trackers.ini",
                       command="run_tracker", protocol="trax")

    def test_escape_matlab_path(self) -> None:
        """A single quote in a path is doubled, Matlab/Octave string style."""
        from vot.tracker.adapters import escape_matlab_path

        self.assertEqual(escape_matlab_path("/plain/path"), "/plain/path")
        self.assertEqual(escape_matlab_path("/o'brien/tracker"), "/o''brien/tracker")

    def test_matlab_adapter_builds_command(self) -> None:
        """The Matlab adapter wraps the tracker command in a guarded ``-r`` script."""
        from vot.tracker.adapters import MatlabAdapter

        adapter = MatlabAdapter(self._RecordingRuntime)
        runtime = adapter(self._tracker(), "run_tracker", {}, paths=["/abs/tracker/path"],
                          matlab="/opt/matlab/bin/matlab")

        assert isinstance(runtime, self._RecordingRuntime)
        command = runtime.command
        self.assertTrue(command.startswith("/opt/matlab/bin/matlab "))
        self.assertIn('-r "', command)
        self.assertIn("diary('runtime.log')", command)
        self.assertIn("addpath('/abs/tracker/path')", command)
        self.assertIn("run_tracker", command)
        self.assertIn("getReport(ex)", command)

    def test_matlab_adapter_escapes_quotes_in_paths(self) -> None:
        """A single quote in an added path is escaped so the script stays valid."""
        from vot.tracker.adapters import MatlabAdapter

        adapter = MatlabAdapter(self._RecordingRuntime)
        runtime = adapter(self._tracker(), "run_tracker", {}, paths=["/o'brien/code"],
                          matlab="/opt/matlab/bin/matlab")

        assert isinstance(runtime, self._RecordingRuntime)
        self.assertIn("addpath('/o''brien/code')", runtime.command)

    def test_octave_adapter_builds_command(self) -> None:
        """The Octave adapter wraps the tracker command in a guarded ``--eval`` script."""
        from vot.tracker.adapters import OctaveAdapter

        adapter = OctaveAdapter(self._RecordingRuntime)
        runtime = adapter(self._tracker(), "run_tracker", {}, paths=["/abs/tracker/path"],
                          octave="/usr/bin/octave")

        assert isinstance(runtime, self._RecordingRuntime)
        command = runtime.command
        self.assertTrue(command.startswith("/usr/bin/octave "))
        self.assertIn('--eval "', command)
        self.assertIn("diary('runtime.log')", command)
        self.assertIn("addpath('/abs/tracker/path')", command)
        self.assertIn("run_tracker", command)

    def test_octave_adapter_error_reporter_uses_string_labels(self) -> None:
        """The Octave error reporter prints the literal ``filename``/``line``
        labels — regression test for the single-quoted-literal concatenation
        that turned ``disp('filename')`` into ``disp(filename)``."""
        from vot.tracker.adapters import OctaveAdapter

        adapter = OctaveAdapter(self._RecordingRuntime)
        runtime = adapter(self._tracker(), "run_tracker", {}, octave="/usr/bin/octave")

        assert isinstance(runtime, self._RecordingRuntime)
        command = runtime.command
        self.assertIn("disp('filename')", command)
        self.assertIn("disp('line')", command)
        self.assertNotIn("disp(filename)", command)
        self.assertNotIn("disp(line)", command)

    def test_octave_adapter_escapes_quotes_in_paths(self) -> None:
        """A single quote in an added path is escaped for Octave too."""
        from vot.tracker.adapters import OctaveAdapter

        adapter = OctaveAdapter(self._RecordingRuntime)
        runtime = adapter(self._tracker(), "run_tracker", {}, paths=["/o'brien/code"],
                          octave="/usr/bin/octave")

        assert isinstance(runtime, self._RecordingRuntime)
        self.assertIn("addpath('/o''brien/code')", runtime.command)

    def test_trax_matlab_adapter_socket_bypass(self) -> None:
        """The Matlab TraX adapter forces socket communication on Windows only."""
        from unittest.mock import patch
        from vot.tracker.trax import TraxMatlabAdapter

        # Non-Windows: stdio communication, no socket bypass.
        adapter = TraxMatlabAdapter()
        adapter.constructor = self._RecordingRuntime
        with patch("vot.tracker.trax.sys.platform", "linux"), \
             patch("vot.tracker.adapters.sys.platform", "linux"):
            runtime = adapter(self._tracker(), "run_tracker", {}, matlab="/opt/matlab/bin/matlab")
        assert isinstance(runtime, self._RecordingRuntime)
        self.assertNotIn("socket", runtime.kwargs)

        # Windows: socket bypass injected for the runtime constructor.
        adapter = TraxMatlabAdapter()
        adapter.constructor = self._RecordingRuntime
        with patch("vot.tracker.trax.sys.platform", "win32"), \
             patch("vot.tracker.adapters.sys.platform", "win32"):
            runtime = adapter(self._tracker(), "run_tracker", {}, matlab="C:/matlab/bin/matlab.exe")
        assert isinstance(runtime, self._RecordingRuntime)
        self.assertIs(runtime.kwargs.get("socket"), True)


class TestRegionCodec(unittest.TestCase):
    """Round-trip tests for the region (de)serialization used by ``PythonRuntime``.

    The worker process exchanges regions with the parent over a multiprocessing
    queue, so ``encode_region`` and ``decode_region`` must be exact inverses.
    """

    def test_rectangle_roundtrip(self) -> None:
        """A rectangle survives encode/decode unchanged."""
        from vot.region import Rectangle
        from vot.tracker.helpers import encode_region, decode_region

        decoded = decode_region(encode_region(Rectangle(10, 20, 30, 40)))
        self.assertIsInstance(decoded, Rectangle)
        assert isinstance(decoded, Rectangle)
        self.assertEqual((decoded.x, decoded.y, decoded.width, decoded.height), (10, 20, 30, 40))

    def test_point_roundtrip(self) -> None:
        """A point survives encode/decode unchanged."""
        from vot.region import Point
        from vot.tracker.helpers import encode_region, decode_region

        decoded = decode_region(encode_region(Point(5, 7)))
        self.assertIsInstance(decoded, Point)
        assert isinstance(decoded, Point)
        self.assertEqual((decoded.x, decoded.y), (5, 7))

    def test_polygon_roundtrip(self) -> None:
        """A polygon survives encode/decode unchanged."""
        from vot.region import Polygon
        from vot.tracker.helpers import encode_region, decode_region

        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        decoded = decode_region(encode_region(Polygon(points)))
        self.assertIsInstance(decoded, Polygon)
        assert isinstance(decoded, Polygon)
        self.assertEqual(decoded.points(), points)

    def test_mask_roundtrip_preserves_bitmap(self) -> None:
        """The mask bitmap survives encode/decode unchanged."""
        from vot.region import Mask
        from vot.tracker.helpers import encode_region, decode_region

        bitmap = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
        decoded = decode_region(encode_region(Mask(bitmap, offset=(0, 0))))
        self.assertIsInstance(decoded, Mask)
        assert isinstance(decoded, Mask)
        np.testing.assert_array_equal(decoded.mask, bitmap)

    def test_mask_roundtrip_preserves_offset(self) -> None:
        """The mask offset survives encode/decode — regression test for the
        offset being silently dropped on decode."""
        from vot.region import Mask
        from vot.tracker.helpers import encode_region, decode_region

        bitmap = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        decoded = decode_region(encode_region(Mask(bitmap, offset=(37, 51))))
        self.assertIsInstance(decoded, Mask)
        assert isinstance(decoded, Mask)
        self.assertEqual(decoded.offset, (37, 51))
        np.testing.assert_array_equal(decoded.mask, bitmap)

    def test_decode_rejects_unknown_data(self) -> None:
        """Decoding an unrecognized payload raises ``ValueError``."""
        from vot.tracker.helpers import decode_region

        with self.assertRaises(ValueError):
            decode_region("not a region")


class TestRegistry(unittest.TestCase):
    """Tests for tracker reference resolution in :class:`vot.tracker.Registry`."""

    def _registry(self) -> "Registry":
        """Builds a registry from a temporary ``trackers.ini`` with two tagged trackers."""
        import tempfile
        from vot.tracker import Registry

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        with open(self._tmp.name + "/trackers.ini", "w") as fp:
            fp.write("[fast_a]\nlabel = A\nprotocol = python\ncommand = a\ntags = fast\n\n")
            fp.write("[fast_b]\nlabel = B\nprotocol = python\ncommand = b\ntags = fast\n\n")
            fp.write("[slow_c]\nlabel = C\nprotocol = python\ncommand = c\ntags = slow\n")
        return Registry([self._tmp.name])

    def test_resolve_tag_without_results(self) -> None:
        # A '#tag' reference resolves every tagged tracker even when none have been evaluated.
        registry = self._registry()
        resolved = registry.resolve("#fast", storage=None, skip_unknown=False)
        self.assertEqual(sorted(t.identifier for t in resolved), ["fast_a", "fast_b"])

    def test_resolve_tag_unknown(self) -> None:
        # An unmatched tag resolves to nothing rather than raising.
        registry = self._registry()
        self.assertEqual(registry.resolve("#missing", storage=None), [])