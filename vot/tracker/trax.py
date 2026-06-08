"""TraX protocol implementation for the toolkit.

TraX is a communication protocol for visual object tracking. It enables communication
between a tracker and a client. The protocol was originally developed for the VOT
challenge to address the need for a unified communication interface between trackers and
benchmarking tools.
"""
import sys
import os
import time
import re
import subprocess
import shutil
import socket as socketio
import tempfile
import logging
import unittest
from typing import Any, NoReturn, Callable
from threading import Thread, Lock

import numpy as np

import colorama

from trax import TraxException
from trax.client import Client
from trax.image import FileImage
from trax.region import Region as TraxRegion
from trax.region import Polygon as TraxPolygon
from trax.region import Mask as TraxMask
from trax.region import Rectangle as TraxRectangle

from vot.dataset import Frame, DatasetException
from vot.region import Region, Polygon, Rectangle, Mask
from vot.tracker import Tracker, TrackerException, TrackerRuntime, FrameObjects, ObjectStatus, OnlineTrackerRuntime
from vot.utilities import to_logical, to_number
from vot.tracker.helpers import normalize_paths, spawn_process

PORT_POOL_MIN = 9090
PORT_POOL_MAX = 65535

logger = logging.getLogger("vot")

class LogAggregator(object):
    """Aggregates log messages from the tracker."""

    def __init__(self) -> None:
        """Initializes the aggregator."""
        self._fragments: list[str] = []

    def __call__(self, fragment: str) -> None:
        """Appends a new fragment to the log."""
        self._fragments.append(fragment)

    def __str__(self) -> str:
        """Returns the aggregated log."""
        return "".join(self._fragments)

class ColorizedOutput(object):
    """Colorized output for the tracker."""

    def __init__(self) -> None:
        """Initializes the colorized output."""
        colorama.init()

    def __call__(self, fragment: str) -> None:
        """Prints a new fragment to the output.

        :param fragment: The fragment to be printed.
        """
        print(colorama.Fore.CYAN + fragment + colorama.Fore.RESET, end="")

class PythonCrashHelper(object):
    """Helper class for detecting Python crashes in the tracker."""

    def __init__(self) -> None:
        """Initializes the crash helper."""
        self._matcher = re.compile(r'''
            ^Traceback
            [\s\S]+?
            (?=^\[|\Z)
            ''', re.M | re.X)

    def __call__(self, log: str, directory: str | None) -> str | None:
        """Detects Python crashes in the log.

        :param log: The log to be checked.
        :param directory: The directory where the log is stored.
        """
        matches = self._matcher.findall(log)
        if len(matches) > 0:
            return matches[-1]
        return None

def convert_frame(frame: Frame, channels: list) -> dict:
    """Converts a frame to a dictionary of Trax images.

    :param frame: The frame to be converted.
    :param channels: The list of channels to be converted.

    :returns: A dictionary of Trax images."""
    tlist = dict()

    for channel in channels:
        image = frame.filename(channel)
        if image is None:
            raise DatasetException("Frame does not have information for channel: {}".format(channel))

        tlist[channel] = FileImage.create(image)

    return tlist

def convert_region(region: Region) -> TraxRegion:
    """Converts a toolkit region to a Trax region.

    :param region: The region to be converted.

    :returns: A Trax region."""
    if isinstance(region, Rectangle):
        # ``TraxRectangle.create`` declares ``int`` coordinates in its stubs while
        # the toolkit stores floats; round at the boundary.
        return TraxRectangle.create(int(region.x), int(region.y), int(region.width), int(region.height))
    if isinstance(region, Polygon):
        return TraxPolygon.create([region[i] for i in range(region.size)])
    if isinstance(region, Mask):
        return TraxMask.create(region.mask, x=region.offset[0], y=region.offset[1])
    raise TraxException("Unknown region type {}".format(type(region)))


def convert_traxregion(region: TraxRegion) -> Region:
    """Converts a Trax region to a toolkit region.

    Uses ``isinstance`` (rather than ``region.type ==``) so pyright can narrow to
    the concrete subclass (``TraxRectangle`` / ``TraxPolygon`` / ``TraxMask``) and
    type-check the attribute access that follows.

    :param region: The Trax region to be converted.

    :returns: A toolkit region."""
    if isinstance(region, TraxRectangle):
        x, y, width, height = region.bounds()
        return Rectangle(x, y, width, height)
    if isinstance(region, TraxPolygon):
        # ``TraxPolygon``'s iterator type is not formally an ``Iterator`` in its
        # stubs, but at runtime it yields ``(x, y)`` tuples; iterate manually.
        points = [tuple(region[i]) for i in range(region.size())]
        return Polygon(points)
    if isinstance(region, TraxMask):
        return Mask(region.array(), region.offset(), optimize=True)
    raise TraxException("Unknown region type {}".format(getattr(region, "type", type(region))))


def convert_objects(objects: FrameObjects | None) -> list:
    """Converts the polymorphic ``FrameObjects`` input shape to the
    ``[(TraxRegion, properties_dict), ...]`` form expected by the Trax client.

    :param objects: List of :class:`ObjectStatus`, a single one, or ``None``.

    :returns: List of ``(TraxRegion, properties_dict)`` tuples.
    """
    if objects is None:
        return []
    if isinstance(objects, list):
        return [(convert_region(o.region), dict(o.properties)) for o in objects]
    if isinstance(objects, ObjectStatus):
        return [(convert_region(objects.region), dict(objects.properties))]
    # ``Region`` fallback for callers that pass a bare region (legacy code paths).
    return [(convert_region(objects), dict())]


def convert_traxstatus(status: list) -> list[ObjectStatus]:
    """Converts a Trax client ``[(TraxRegion, properties), ...]`` status into a list of
    :class:`ObjectStatus` (inverse of :func:`convert_objects`)."""
    return [ObjectStatus(convert_traxregion(region), properties) for region, properties in status]


def convert_traxobjects(region: TraxRegion) -> Region:
    """Same as :func:`convert_traxregion` — kept for historical naming."""
    return convert_traxregion(region)

class TestRasterMethods(unittest.TestCase):
    """Tests for the raster methods."""

    def test_convert_traxregion(self) -> None:
        """Tests the conversion of Trax regions."""
        convert_traxregion(TraxRectangle.create(0, 0, 10, 10))
        convert_traxregion(TraxPolygon.create([(0, 0), (10, 0), (10, 10), (0, 10)]))
        convert_traxregion(TraxMask.create(np.ones((100, 100), dtype=np.uint8)))

    def test_convert_region(self) -> None:
        """Tests the conversion of regions."""
        convert_region(Rectangle(0, 0, 10, 10))
        convert_region(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
        convert_region(Mask(np.ones((100, 100), dtype=np.uint8)))

def open_local_port(port: int) -> socketio.socket | None:
    """Opens a local port for listening."""
    socket = socketio.socket(socketio.AF_INET, socketio.SOCK_STREAM)
    try:
        socket.setsockopt(socketio.SOL_SOCKET, socketio.SO_REUSEADDR, 1)
        socket.bind(('127.0.0.1', port))
        socket.listen(1)
        return socket
    except OSError:
        try:
            socket.close()
        except OSError:
            pass
        return None

class TrackerProcess(object):
    """A tracker process.

    This class is used to run trackers in a separate process and handles starting,
    stopping and communication with the process.
    """
    
    def __init__(self, command: str, envvars: dict | None = None, timeout: int = 30, log: bool = False, socket: bool = False) -> None:
        """Initializes a new tracker process.

        :param command: The command to run the tracker.
        :param envvars: A dictionary of environment variables to be set for the tracker process.
        :param timeout: The timeout for the tracker process.
        :param log: Whether to log the tracker output.
        :param socket: Whether to use a socket for communication.
        """
        
        environment = dict(os.environ)
        if envvars is not None:
            environment.update(envvars)

        self._workdir = tempfile.mkdtemp()

        self._returncode: int | None = None
        self._socket: Any = None

        if socket:
            port: int | None = None
            for candidate_port in range(PORT_POOL_MIN, PORT_POOL_MAX):
                opened = open_local_port(candidate_port)
                if opened is not None:
                    self._socket = opened
                    port = candidate_port
                    break
            if port is None:
                raise TraxException("Unable to open any TRAX socket in the configured port range")
            environment["TRAX_SOCKET"] = "{}".format(port)

        logger.debug("Running process: %s", command)

        self._process = spawn_process(command, self._workdir, environment)

        self._timeout = timeout
        self._client = None

        self._watchdog_lock = Lock()
        self._watchdog_counter = 0
        self._watchdog = Thread(target=self._watchdog_loop)
        self._watchdog.start()

        self._watchdog_reset(True)

        try:
            if socket:
                self._client = Client(stream=self._socket.fileno(), timeout=30, log=log)
            else:
                # ``spawn_process`` sets ``stdin``/``stdout`` to ``subprocess.PIPE``, so
                # they are guaranteed non-None at runtime; narrow for the type checker,
                # which sees them as ``IO[bytes] | None`` in the stubs.
                assert self._process.stdin is not None and self._process.stdout is not None
                self._client = Client(
                    stream=(self._process.stdin.fileno(), self._process.stdout.fileno()), log=log
                )

        except TraxException as e:
            self.terminate()
            self._watchdog_reset(False)
            raise e
        self._watchdog_reset(False)

        self._has_vot_wrapper = self._client.get("vot") is not None
        self._multiobject = self._client.get("multiobject")

    def _watchdog_reset(self, enable: bool = True) -> None:
        """Resets the watchdog.

        :param enable: Whether to enable the watchdog.
        """
        if self._watchdog_counter == 0:
            return

        if enable:
            self._watchdog_counter = self._timeout * 10
        else:
            self._watchdog_counter = -1

    def _watchdog_loop(self) -> None:
        """The watchdog loop.

        This loop is used to monitor the tracker process and terminate it if it does not
        respond anymore.
        """

        while self.alive:
            time.sleep(0.1)
            if self._watchdog_counter < 0:
                continue
            self._watchdog_counter = self._watchdog_counter - 1
            if not self._watchdog_counter:
                logger.warning("Timeout reached, terminating tracker")
                self.terminate()
                break

    @property
    def has_vot_wrapper(self) -> bool:
        """Whether the tracker has a VOT wrapper.

        VOT wrapper limits TraX functionality and injects a property at handshake to let
        the client know this.
        """
        return self._has_vot_wrapper

    @property
    def returncode(self) -> int | None:
        """The return code of the tracker process."""
        return self._returncode

    @property
    def workdir(self) -> str:
        """The working directory of the tracker process."""
        return self._workdir

    @property
    def interrupted(self) -> bool:
        """Whether the tracker process was interrupted."""
        return self._watchdog_counter == 0

    @property
    def alive(self) -> bool:
        """Whether the tracker process is alive."""
        if self._process is None:
            return False
        self._returncode = self._process.returncode
        return self._returncode is None

    def initialize(self, frame: Frame, new: FrameObjects | None = None) -> tuple[FrameObjects, float]:
        """ Initializes the tracker. This method is used to initialize the tracker with the first frame. It returns the initial state of the tracker.

        :param frame: The first frame.
        :param new: The initial state of the tracker.

        :returns: The initial state of the tracker.
        :raises TraxException: If the tracker is not alive."""

        if not self.alive:
            raise TraxException("Tracker not alive")

        # ``alive`` returning ``True`` guarantees both ``_process`` and ``_client``
        # are set (the constructor either assigns both or raises). Narrow here so
        # the type checker doesn't flag every attribute access below.
        assert self._client is not None

        tlist = convert_frame(frame, self._client.channels)
        tobjects = convert_objects(new)

        self._watchdog_reset(True)

        status, elapsed = self._client.initialize(tlist, tobjects, dict())

        self._watchdog_reset(False)

        status = convert_traxstatus(status)

        return status, elapsed


    def update(self, frame: Frame, new: FrameObjects | None = None) -> tuple[FrameObjects, float]:
        """ Updates the tracker with a new frame. This method is used to update the tracker with a new frame. It returns the new state of the tracker.

        :param frame: The new frame.
        :param new: The new state of the tracker.

        :returns: The new state of the tracker.
        :raises TraxException: If the tracker is not alive."""

        if not self.alive:
            raise TraxException("Tracker not alive")

        # ``alive`` ⇒ ``_client`` is set.
        assert self._client is not None

        tlist = convert_frame(frame, self._client.channels)

        tobjects = convert_objects(new)

        self._watchdog_reset(True)

        status, elapsed = self._client.frame(tlist, dict(), tobjects)

        self._watchdog_reset(False)

        status = convert_traxstatus(status)

        return status, elapsed

    def terminate(self) -> None:
        """Terminates the tracker.

        This method is used to terminate the tracker. It closes the connection to the
        tracker and terminates the tracker process.
        """
        with self._watchdog_lock:

            if not self.alive:
                return

            # ``alive`` ⇒ ``_process`` is set; capture into a local for narrowing.
            process = self._process
            assert process is not None

            if self._client is not None:
                self._client.quit()

            try:
                process.wait(3)
            except subprocess.TimeoutExpired:
                pass

            if process.returncode is None:
                process.terminate()
                try:
                    process.wait(3)
                except subprocess.TimeoutExpired:
                    pass

                if process.returncode is None:
                    process.kill()

            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()

            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()

            if self._socket is not None:
                self._socket.close()

            self._returncode = process.returncode

            self._client = None
            self._process = None

    def __del__(self) -> None:
        """Destructor.

        This method is used to terminate the tracker process if it is still alive.
        """
        if hasattr(self, "_workdir"):
            shutil.rmtree(self._workdir, ignore_errors=True)

    def wait(self) -> None:
        """Waits for the tracker to terminate.

        This method is used to wait for the tracker to terminate. It waits until the
        tracker process terminates.
        """

        self._watchdog_reset(True)

        if self._process is None or self._client is None:
            self._watchdog_reset(False)
            return

        stdout = self._process.stdout
        if stdout is None:
            self._watchdog_reset(False)
            return

        client = self._client

        # Flush remaining output
        while True:
            line = stdout.readline()
            if not line:
                break
            client_logger = getattr(client, "_logger", None)
            if client_logger is not None:
                client_logger.handle(line.decode("utf-8"))

        self._watchdog_reset(False)

    @property
    def multiobject(self) -> str:
        """Whether the tracker supports multiple objects."""
        return self._multiobject

class TraxTrackerRuntime(OnlineTrackerRuntime):
    """The TraX tracker runtime.

    This class is used to run a tracker using the TraX protocol.
    """

    def __init__(
        self,
        tracker: Tracker,
        command: str,
        log: bool = False,
        timeout: int = 30,
        linkpaths: Any | None = None,
        paths: Any | None = None,
        envvars: dict | None = None,
        arguments: Any | None = None,
        socket: bool = False,
        restart: bool = False,
        onerror: Callable[..., Any] | None = None,
    ) -> None:
        """Initializes the TraX tracker runtime.

        :param tracker: The tracker to be run.
        :param command: The command to run the tracker.
        :param log: Whether to log the output of the tracker.
        :param timeout: The timeout in seconds for the tracker to respond.
        :param linkpaths: The paths to be added to the PATH environment variable.
        :param paths: Registry-key alias of ``linkpaths``; merged into it.
        :param envvars: The environment variables to be set for the tracker.
        :param arguments: The arguments to be passed to the tracker.
        :param socket: Whether to use a socket to communicate with the tracker.
        :param restart: Whether to restart the tracker if it crashes.
        :param onerror: The error handler to be called if the tracker crashes.
        """
        super().__init__(tracker)
        self._command = command
        self._process: TrackerProcess | None = None
        self._tracker = tracker
        if linkpaths is None:
            linkpaths = []
        if isinstance(linkpaths, str):
            linkpaths = linkpaths.split(os.pathsep)
        # ``paths`` is the registry key for the same notion as ``linkpaths``. The
        # trax* adapter protocols consume it themselves, but the plain ``trax``
        # protocol constructs this runtime directly, so accept and merge it here.
        if paths:
            if isinstance(paths, str):
                paths = paths.split(os.pathsep)
            linkpaths = list(linkpaths) + list(paths)
        linkpaths = normalize_paths(linkpaths, tracker)
        self._socket = to_logical(socket)
        self._restart = to_logical(restart)
        self._output: Any = LogAggregator() if not log else None
        self._timeout = to_number(timeout, min_n=1)
        self._arguments = arguments
        self._onerror = onerror
        self._workdir: str | None = None

        pathvar = "PATH" if sys.platform.startswith("win") else "LD_LIBRARY_PATH"

        envvars = dict(envvars) if envvars is not None else {}
        if pathvar in envvars:
            envvars[pathvar] = envvars[pathvar] + os.pathsep + os.pathsep.join(linkpaths)
        else:
            envvars[pathvar] = os.pathsep.join(linkpaths)
        envvars["TRAX"] = "1"

        self._envvars: dict = envvars

    @property
    def tracker(self) -> Tracker:
        """The associated tracker object."""
        return self._tracker

    @property
    def multiobject(self) -> bool:
        """Whether the tracker supports multiple objects."""
        self._connect()
        assert self._process is not None, "Tracker process should be available after _connect()"
        return bool(self._process.multiobject)

    def _connect(self) -> None:
        """Connects to the tracker.

        This method is used to connect to the tracker. It starts the tracker process if
        it is not running yet.
        """
        if not self._process:
            log: Any = self._output if self._output is not None else ColorizedOutput()
            self._process = TrackerProcess(self._command, self._envvars, log=log, socket=self._socket, timeout=self._timeout)
            if self._process.has_vot_wrapper:
                self._restart = True

    def _error(self, exception: Exception) -> NoReturn:
        """Handles an error and re-raises it as a ``TrackerException``.

        This method always raises — annotated ``NoReturn`` so callers don't need
        to handle a fall-through ``None`` return.
        """
        timeout = False
        if not self._output is None:
            if not self._process is None:
                if self._process.alive:
                    self._process.terminate()
                
                self._output(f"Process exited with code ({self._process.returncode})\n")
                timeout = self._process.interrupted
                self._workdir = self._process.workdir
            else:
                self._output("Process not alive anymore, unable to retrieve return code\n")

        log = str(self._output)

        try:

            if self._onerror is not None:
                self._onerror(log, self._workdir)

        except Exception as e:
            logger.exception("Error during error handler for runtime of tracker %s", self._tracker.identifier, exc_info=e)

        if timeout:
            raise TrackerException(f"Tracker interrupted, it did not reply in {self._timeout} seconds", tracker=self._tracker, \
                tracker_log=log if not self._output is None else None)

        raise TrackerException(exception, tracker=self._tracker, \
            tracker_log=log if not self._output is None else None)

    def restart(self) -> None:
        """Restarts the tracker.

        This method is used to restart the tracker. It stops the tracker process and
        starts it again.
        """
        try:
            self.stop()
            self._connect()
        except TraxException as e:
            self._error(e)

    def initialize(
        self,
        frame: Frame,
        new: FrameObjects | None = None,
        properties: dict | None = None,
    ) -> tuple[FrameObjects, float]:
        """Initializes the tracker. This method is used to initialize the tracker. It
        starts the tracker process if it is not running yet.

        ``new`` accepts either a single ``ObjectStatus`` (legacy per-frame caller, e.g.
        ``SupervisedExperiment``) or a ``list[ObjectStatus]`` (query-based caller via
        ``OnlineTrackerRuntime.run``). The return shape mirrors the input shape so
        legacy callers get back a single ``ObjectStatus`` and query-based callers get
        the list — restoring the asymmetric behaviour previously provided by the
        ``SingleObjectTrackerRuntime`` wrapper.

        :param frame: The initial frame.
        :param new: The initial objects — either a single ``ObjectStatus`` or a list thereof.
        :param properties: The initial properties (unused — kept for signature compatibility).

        :returns: A tuple of (objects, elapsed). ``objects`` mirrors the input shape."""
        del properties  # reserved
        try:
            input_is_list = isinstance(new, list)
            if not self.multiobject and input_is_list:
                assert isinstance(new, list)
                if len(new) != 1:
                    raise TrackerException(
                        "Tracker does not support multiple objects, but multiple objects were provided for initialization",
                        tracker=self._tracker,
                    )
                new = new[0]

            if self._restart:
                self.stop()
            self._connect()
            assert self._process is not None, "Tracker process should be available after _connect()"

            status, elapsed = self._process.initialize(frame, new)
            if not self.multiobject and not input_is_list and isinstance(status, list):
                status = status[0]
            return status, elapsed
        except TraxException as e:
            self._error(e)

    def update(
        self,
        frame: Frame,
        new: FrameObjects | None = None,
        properties: dict | None = None,
    ) -> tuple[FrameObjects, float]:
        """Updates the tracker. This method is used to update the tracker state with a
        new frame.

        ``new`` accepts either a single ``ObjectStatus`` (legacy callers) or a
        ``list[ObjectStatus]`` (query-based callers). The return shape mirrors the
        input shape: legacy callers get a single ``ObjectStatus``, query-based
        callers get the list.

        :param frame: The current frame.
        :param new: The current objects — either a single ``ObjectStatus``, a list, or ``None``.
        :param properties: The current properties (unused — kept for signature compatibility).

        :returns: A tuple of (objects, elapsed). ``objects`` mirrors the input shape."""
        del properties  # reserved
        try:
            input_is_list = isinstance(new, list)
            if not self.multiobject and input_is_list:
                assert isinstance(new, list)
                if len(new) > 1:
                    raise TrackerException(
                        "Tracker does not support multiple objects, but multiple objects were provided for update",
                        tracker=self._tracker,
                    )
                new = new[0] if len(new) == 1 else None

            assert self._process is not None, "Tracker process should be available for update()"
            status, elapsed = self._process.update(frame, new)
            if not self.multiobject and not input_is_list and isinstance(status, list):
                status = status[0]
            return status, elapsed
        except TraxException as e:
            self._error(e)

    def stop(self) -> None:
        """Stops the tracker.

        This method is used to stop the tracker. It stops the tracker process.
        """
        # ``__del__`` may run on a half-constructed object when ``__init__`` raised
        # before ``self._process`` was assigned, so guard with ``getattr``.
        process = getattr(self, "_process", None)
        if process is not None:
            process.terminate()
            self._process = None

    def __del__(self) -> None:
        """Destructor.

        This method is used to stop the tracker process when the object is deleted.
        """
        self.stop()

from vot.tracker import register_runtime_protocol
from vot.tracker.adapters import PythonAdapter, MatlabAdapter, OctaveAdapter

class TraxMatlabAdapter(MatlabAdapter):
    """Adapter for running a tracker using the TraX protocol in Matlab.

    It only adds the bypass to use socket communication on Windows, which is required
    for Matlab to work properly.
    """

    def __init__(self) -> None:
        """Binds the adapter to the TraX tracker runtime constructor."""
        super().__init__(TraxTrackerRuntime)

    def __call__(
        self,
        tracker: Tracker,
        command: str,
        envvars: dict[str, str],
        paths: str | list[str] = "",
        log: bool = False,
        timeout: int = 30,
        linkpaths: str | list[str] | None = None,
        arguments: dict[str, Any] | None = None,
        matlab: str | None = None,
        **kwargs: Any,
    ) -> TrackerRuntime:
        """Builds the Matlab command line, forcing socket communication on Windows.

        :param tracker: The tracker to create the adapter for.
        :param command: The command to run the tracker.
        :param envvars: The environment variables to set.
        :param paths: The paths to add to the Matlab path.
        :param log: Whether to log the tracker output.
        :param timeout: The timeout in seconds.
        :param linkpaths: The paths to link.
        :param arguments: The arguments to pass to the tracker.
        :param matlab: The Matlab executable to use.
        :param kwargs: Additional keyword arguments for the runtime constructor.

        :returns: The tracker runtime object."""
        if sys.platform.startswith("win"):
            kwargs["socket"] = True # We have to use socket connection in this case

        return super().__call__(tracker, command, envvars, paths, log, timeout, linkpaths, arguments, matlab, **kwargs)

register_runtime_protocol("trax", TraxTrackerRuntime)
register_runtime_protocol("traxpython", PythonAdapter(TraxTrackerRuntime))
register_runtime_protocol("traxmatlab", TraxMatlabAdapter())
register_runtime_protocol("traxoctave", OctaveAdapter(TraxTrackerRuntime))
