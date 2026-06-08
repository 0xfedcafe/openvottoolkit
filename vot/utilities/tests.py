"""Unit tests for vot.utilities."""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


class TestParentWatchdog(unittest.TestCase):
    """:func:`vot.utilities.arm_parent_watchdog` must terminate the worker when
    its parent dies, so a SIGKILL'd ``vot`` CLI doesn't leave orphan workers
    burning CPU on a now-broken pipe."""

    def test_child_exits_when_parent_dies(self) -> None:
        """A grandchild that armed the watchdog exits shortly after its
        immediate parent is killed — the OS reparents it to init/launchd,
        ``os.getppid()`` changes, and the watchdog calls ``os._exit``."""
        import vot
        toolkit_root = os.path.dirname(os.path.dirname(vot.__file__))

        pid_fd, pid_path = tempfile.mkstemp(suffix=".pid")
        os.close(pid_fd)
        os.unlink(pid_path)

        # The grandchild arms the watchdog with a tight poll interval, records
        # its PID, and busy-waits forever. Without the watchdog it would never
        # exit when its parent is killed.
        grandchild_code = textwrap.dedent(f"""
            import os, sys, time
            sys.path.insert(0, {toolkit_root!r})
            from vot.utilities import arm_parent_watchdog
            arm_parent_watchdog(poll_seconds=0.2)
            with open({pid_path!r}, "w") as f:
                f.write(str(os.getpid()))
            while True:
                time.sleep(0.1)
        """)
        # The immediate parent just spawns the grandchild and then sleeps long
        # enough for the test to SIGKILL it.
        parent_code = textwrap.dedent(f"""
            import subprocess, sys, time
            subprocess.Popen([sys.executable, "-c", {grandchild_code!r}])
            time.sleep(60)
        """)

        parent_proc = subprocess.Popen([sys.executable, "-c", parent_code])
        try:
            child_pid = self._wait_for_pid_file(pid_path, timeout_s=10.0)
            assert child_pid is not None, "grandchild never wrote its PID"

            parent_proc.kill()
            parent_proc.wait()

            self.assertTrue(
                self._wait_for_process_exit(child_pid, timeout_s=3.0),
                f"grandchild PID {child_pid} still alive 3s after parent SIGKILL",
            )
        finally:
            if parent_proc.poll() is None:
                parent_proc.kill()
                parent_proc.wait()
            try:
                os.unlink(pid_path)
            except OSError:
                pass

    @staticmethod
    def _wait_for_pid_file(path: str, timeout_s: float) -> int | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with open(path) as f:
                    content = f.read().strip()
                if content:
                    return int(content)
            except OSError:
                pass
            time.sleep(0.05)
        return None

    @staticmethod
    def _wait_for_process_exit(pid: int, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        # Clean up the survivor so it doesn't linger past the test.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return False


class TestToLogical(unittest.TestCase):
    """:func:`vot.utilities.to_logical` interprets strings and falls back to ``bool``."""

    def test_truthy_strings(self) -> None:
        from vot.utilities import to_logical
        for value in ("true", "TRUE", "1", "t", "y", "yes", "Yes"):
            self.assertTrue(to_logical(value), value)

    def test_falsey_strings(self) -> None:
        from vot.utilities import to_logical
        for value in ("false", "0", "n", "no", "", "maybe"):
            self.assertFalse(to_logical(value), value)

    def test_non_string_uses_bool(self) -> None:
        from vot.utilities import to_logical
        self.assertTrue(to_logical(1))
        self.assertFalse(to_logical(0))
        self.assertFalse(to_logical(None))
        self.assertTrue(to_logical([0]))


class TestColor(unittest.TestCase):
    """:class:`vot.utilities.draw.Color` resolution and conversions."""

    def test_resolve_palette_name(self) -> None:
        from vot.utilities.draw import Color
        self.assertEqual(Color.resolve("red"), Color(1.0, 0.0, 0.0))

    def test_resolve_unknown_name_is_black(self) -> None:
        from vot.utilities.draw import Color
        # Previously the fallback was a 4-tuple (0, 0, 0, 1); now it is a proper Color.
        self.assertEqual(Color.resolve("nosuchcolor"), Color(0.0, 0.0, 0.0))

    def test_resolve_rgb_and_rgba_tuples(self) -> None:
        from vot.utilities.draw import Color
        self.assertEqual(Color.resolve((0.1, 0.2, 0.3)), Color(0.1, 0.2, 0.3, 1.0))
        self.assertEqual(Color.resolve((0.1, 0.2, 0.3, 0.4)), Color(0.1, 0.2, 0.3, 0.4))

    def test_resolve_clamps_components(self) -> None:
        from vot.utilities.draw import Color
        self.assertEqual(Color.resolve((2.0, -1.0, 0.5)), Color(1.0, 0.0, 0.5))

    def test_resolve_passthrough(self) -> None:
        from vot.utilities.draw import Color
        color = Color(0.2, 0.4, 0.6)
        self.assertIs(Color.resolve(color), color)

    def test_conversions(self) -> None:
        from vot.utilities.draw import Color
        color = Color(1.0, 0.0, 0.5)
        self.assertEqual(color.rgb(), (1.0, 0.0, 0.5))
        self.assertEqual(color.rgba(), (1.0, 0.0, 0.5, 1.0))
        self.assertEqual(color.to_int(), (255, 0, 127, 255))
        self.assertEqual(color.to_int(128), (255, 0, 127, 128))
        self.assertEqual(color.with_alpha(0.25), Color(1.0, 0.0, 0.5, 0.25))

    def test_resolve_color_compat_returns_rgb(self) -> None:
        from vot.utilities.draw import resolve_color
        self.assertEqual(resolve_color("red"), (1.0, 0.0, 0.0))


class TestImportClass(unittest.TestCase):
    """:func:`vot.utilities.import_class` imports by dotted path and validates input."""

    def test_imports_class_by_path(self) -> None:
        from vot.utilities import import_class
        from collections import OrderedDict
        self.assertIs(import_class("collections.OrderedDict"), OrderedDict)

    def test_bare_name_raises_import_error(self) -> None:
        from vot.utilities import import_class
        with self.assertRaises(ImportError):
            import_class("OrderedDict")
