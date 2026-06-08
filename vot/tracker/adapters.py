"""Runtime adapters that translate a tracker manifest entry into the concrete
command line used to launch the tracker process (Python, Matlab or Octave)."""

from __future__ import annotations

import os
import sys
import re
from typing import TYPE_CHECKING, Any, Callable

from vot.tracker.helpers import normalize_paths

if TYPE_CHECKING:
    from vot.tracker import Tracker, TrackerRuntime

def escape_path(path: str) -> str:
    """Escapes a path for safe embedding: on Windows, doubles backslashes for a Python
    string literal; on POSIX, escapes single quotes for the shell.

    :param path: The path to escape.

    :returns: The escaped path."""
    if sys.platform.startswith("win"):
        return path.replace("\\\\", "\\").replace("\\", "\\\\")
    else:
        return path.replace("'", "'\\''")

def escape_matlab_path(path: str) -> str:
    """Escapes a path for use inside a single-quoted Matlab/Octave string.

    Matlab and Octave escape an embedded single quote by doubling it; backslashes
    (Windows separators) need no escaping inside a single-quoted string.

    :param path: The path to escape.

    :returns: The escaped path."""
    return path.replace("'", "''")

def split_paths(paths: str | list[str]) -> list[str]:
    """Normalizes ``paths`` to a list, splitting a string on the path separator."""
    if isinstance(paths, list):
        return paths
    return paths.split(os.pathsep)

def find_executable_root(root_env_var: str, executable_name: str, error_msg: str) -> str:
    """Locate an installation root for ``executable_name``.

    Uses ``$root_env_var`` if set, otherwise scans ``$PATH`` for the executable and
    takes its parent directory.

    :raises RuntimeError: with ``error_msg`` if the root cannot be determined."""
    root: str | None = os.getenv(root_env_var, None)
    if root is None:
        for testdir in os.getenv("PATH", "").split(os.pathsep):
            if os.path.isfile(os.path.join(testdir, executable_name)):
                root = os.path.dirname(testdir)
                break
        if root is None:
            raise RuntimeError(error_msg)
    return root

class PythonAdapter():
    """Builds the command line for a tracker integrated through the Python TraX wrapper."""

    def __init__(self, constructor: Callable[..., "TrackerRuntime"]) -> None:
        """Stores the runtime constructor invoked once the command line is built.

        :param constructor: The tracker runtime class to instantiate."""
        self.constructor = constructor

    def __call__(
        self,
        tracker: "Tracker",
        command: str,
        envvars: dict[str, str],
        paths: str | list[str] = "",
        log: bool = False,
        timeout: int = 30,
        linkpaths: str | list[str] | None = None,
        arguments: dict[str, Any] | None = None,
        python: str | None = None,
        **kwargs: Any,
    ) -> "TrackerRuntime":
        """Builds the Python interpreter command line for the tracker and returns the runtime.

        :param tracker: The tracker to create the adapter for.
        :param command: The command to run the tracker.
        :param envvars: The environment variables to set.
        :param paths: The paths to add to the Python path.
        :param log: Whether to log the tracker output.
        :param timeout: The timeout in seconds.
        :param linkpaths: The paths to link.
        :param arguments: The arguments to pass to the tracker.
        :param python: The Python interpreter to use.
        :param kwargs: Additional keyword arguments for constructor.

        :returns: The tracker runtime object."""
        paths = split_paths(paths)

        pathimport = " ".join(["sys.path.insert(0, '{}');".format(escape_path(x)) for x in normalize_paths(paths[::-1], tracker)])
        interpreter = sys.executable if python is None else python

        # simple check if the command is only a package name to be imported or a script
        if re.match("^[a-zA-Z_][a-zA-Z0-9_\\.]*$", command) is None:
            # We have to escape all double quotes
            command = command.replace("\"", "\\\"")
            command = '{} -c "import sys;{} {}"'.format(interpreter, pathimport, command)
        else:
            command = '{} -m {}'.format(interpreter, command)

        envvars["PYTHONPATH"] = os.pathsep.join(normalize_paths(paths[::-1], tracker))
        envvars["PYTHONUNBUFFERED"] = "1"

        return self.constructor(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, **kwargs)

class MatlabAdapter():
    """Builds the command line for a tracker integrated through the Matlab TraX wrapper."""

    def __init__(self, constructor: Callable[..., "TrackerRuntime"]) -> None:
        """Stores the runtime constructor invoked once the command line is built.

        :param constructor: The tracker runtime class to instantiate."""
        self.constructor = constructor

    def __call__(
        self,
        tracker: "Tracker",
        command: str,
        envvars: dict[str, str],
        paths: str | list[str] = "",
        log: bool = False,
        timeout: int = 30,
        linkpaths: str | list[str] | None = None,
        arguments: dict[str, Any] | None = None,
        matlab: str | None = None,
        **kwargs: Any,
    ) -> "TrackerRuntime":
        """Builds the Matlab command line for the tracker and returns the runtime.

        :param tracker: The tracker to create the adapter for.
        :param command: The command to run the tracker.
        :param envvars: The environment variables to set.
        :param paths: The paths to add to the Matlab path.
        :param log: Whether to log the tracker output.
        :param timeout: The timeout in seconds.
        :param linkpaths: The paths to link.
        :param arguments: The arguments to pass to the tracker.
        :param matlab: The Matlab executable to use.
        :param kwargs: Additional keyword arguments for constructor.

        :returns: The tracker runtime object."""
        paths = split_paths(paths)

        pathimport = " ".join(["addpath('{}');".format(escape_matlab_path(x)) for x in normalize_paths(paths, tracker)])

        if sys.platform.startswith("win"):
            matlabname = "matlab.exe"
        else:
            matlabname = "matlab"

        if matlab is None:
            matlabroot = find_executable_root(
                "MATLAB_ROOT", matlabname,
                "Matlab executable not found, set MATLAB_ROOT environmental variable manually.")
            matlab_executable = os.path.join(matlabroot, 'bin', matlabname)
        else:
            matlab_executable = matlab

        if sys.platform.startswith("win"):
            matlab_executable = '"' + matlab_executable + '"'
            matlab_flags: list[str] = ['-nodesktop', '-nosplash', '-wait', '-minimize']
        else:
            matlab_flags = ['-nodesktop', '-nosplash']

        # The script is a double-quoted Python literal so the single quotes inside
        # reach Matlab verbatim; a single-quoted literal would silently concatenate
        # the ``''...''`` fragments and strip the quotes.
        matlab_script = "try; diary('runtime.log'); {}{}; catch ex; disp(getReport(ex)); end; quit;".format(pathimport, command)

        command = '{} {} -r "{}"'.format(matlab_executable, " ".join(matlab_flags), matlab_script)

        return self.constructor(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, **kwargs)

class OctaveAdapter():
    """Builds the command line for a tracker integrated through the Octave TraX wrapper."""

    def __init__(self, constructor: Callable[..., "TrackerRuntime"]) -> None:
        """Stores the runtime constructor invoked once the command line is built.

        :param constructor: The tracker runtime class to instantiate."""
        self.constructor = constructor

    def __call__(
        self,
        tracker: "Tracker",
        command: str,
        envvars: dict[str, str],
        paths: str | list[str] = "",
        log: bool = False,
        timeout: int = 30,
        linkpaths: str | list[str] | None = None,
        arguments: dict[str, Any] | None = None,
        octave: str | None = None,
        **kwargs: Any,
    ) -> "TrackerRuntime":
        """Builds the Octave command line for the tracker and returns the runtime.

        :param tracker: The tracker to create the adapter for.
        :param command: The command to run the tracker.
        :param envvars: The environment variables to set.
        :param paths: The paths to add to the Octave path.
        :param log: Whether to log the tracker output.
        :param timeout: The timeout in seconds.
        :param linkpaths: The paths to link.
        :param arguments: The arguments to pass to the tracker.
        :param octave: The Octave executable to use.
        :param kwargs: Additional keyword arguments for constructor.

        :returns: The tracker runtime object."""

        paths = split_paths(paths)

        pathimport = " ".join(["addpath('{}');".format(escape_matlab_path(x)) for x in normalize_paths(paths, tracker)])

        if sys.platform.startswith("win"):
            octavename = "octave.exe"
        else:
            octavename = "octave"

        if octave is None:
            octaveroot = find_executable_root(
                "OCTAVE_ROOT", octavename,
                "Octave executable not found, set OCTAVE_ROOT environmental variable manually.")
            octave_executable = os.path.join(octaveroot, 'bin', octavename)
        else:
            octave_executable = octave

        if sys.platform.startswith("win"):
            octave_executable = '"' + octave_executable + '"'

        octave_flags: list[str] = ['--no-gui', '--no-window-system']

        # Double-quoted Python literal: see the note in MatlabAdapter — a single-quoted
        # literal would concatenate the ``''...''`` fragments and turn ``disp('filename')``
        # into ``disp(filename)``, which references an undefined variable in the catch.
        octave_script = "try; diary('runtime.log'); {}{}; catch ex; disp(ex.message); for i = 1:size(ex.stack) disp('filename'); disp(ex.stack(i).file); disp('line'); disp(ex.stack(i).line); endfor; end; quit;".format(pathimport, command)

        command = '{} {} --eval "{}"'.format(octave_executable, " ".join(octave_flags), octave_script)

        return self.constructor(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, **kwargs)
