
from typing import Any, Iterable

import subprocess
import shutil
import os
import time
import tempfile
from types import TracebackType

from vot import log_debug
from vot.region import Region, Mask, Rectangle, Point, Polygon
from vot.region.io import parse_region
from vot.tracker import Tracker, TrackerRuntime, ObjectQuery, ObjectStatus, RunQueries, RunResult
from vot.dataset import Frame
from vot.tracker.helpers import convert_region, spawn_process

def make_temporary_folder() -> str:
    """Creates a temporary folder and returns its path."""
    return tempfile.mkdtemp(prefix="tracker_folder_")

def generate_query_file(folder: str, oid: str, query: ObjectQuery, convert: str | None = None) -> None:
    """Generates a query file for the given object query in the specified folder."""
    filename = os.path.join(folder, f"query_{oid}.txt")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{query.offset}\n")
        f.write(f"{str(convert_region(query.state, convert))}\n")
        for key, value in query.properties.items():
            f.write(f"{key}={value}\n")

def generate_input_data(folder: str, frames: list[Frame], queries: dict[str, ObjectQuery], convert: str | None = None) -> None:
    """Generates the input data for the tracker in the specified folder."""
    # Generate frames file
    channels = set()
    for frame in frames:
        channels.update(frame.channels())
    
    for channel in channels:
        filename = os.path.join(folder, f"frames_{channel}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            for frame in frames:
                if channel in frame.channels():
                    f.write(f"{frame.filename(channel)}\n")
    
    # Generate query files
    for oid, query in queries.items():
        generate_query_file(folder, oid, query, convert=convert)

def parse_output(folder: str, oid: str) -> list[ObjectStatus]:
    """Parses the output file for the given object id in the specified folder."""
    filename = os.path.join(folder, f"output_{oid}.txt")
    if not os.path.exists(filename):
        raise RuntimeError(f"Output file for object {oid} not found")
    
    with open(filename, "r", encoding="utf-8") as f:
        trajectory = [parse_region(line.strip()) for line in f]

    # Search for additional properties
    properties = {}
    for file in os.listdir(folder):
        if file.startswith(f"output_{oid}_") and file.endswith(".txt"):
            key = file[len(f"output_{oid}_"):-len(".txt")]
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                properties[key] = [line.strip() for line in f]

    # Combine trajectory with properties into object status list
    status = []
    for i in range(len(trajectory)):
        obj_status = ObjectStatus(trajectory[i], {key: properties[key][i] for key in properties})
        status.append(obj_status)

    return status

def parse_outputs(folder: str, oids: list[str]) -> dict[str, list[ObjectStatus]]:
    """Parses the output files for the given object ids in the specified folder."""
    return {oid: parse_output(folder, oid) for oid in oids}

class TrackerFolderRuntime(TrackerRuntime):
    """A tracker runtime that executes a tracker process in a temporary folder. The
    folder is deleted after the tracker process finishes.

    The core idea is that a tracker runs as a batch program over a directory of structured data.
    The tracker then outputs the required output in the specified files.

    Sequence specification

    The sequence is specified as a sequence of image files.
    Multiple input channels are supported, depending on the sequence.
    The file is “frames_<CHANNEL>.txt”, and each line contains a single path to a frame file.
    This can be an absolute or a relative path (in this case, relative to the working directory).

    Query specification

    A query is specified in a single file per object.
    All files follow the naming pattern “query_<ID>.txt”, where <ID> denotes the string identifier of an object (alphanumeric sequence). The file contains the following lines:

    * Offset - a single value for the temporal location of the query; frames start with number 0.
    * State - Initialization state, a comma-separated sequence of numbers. Can contain various state formats:
        * No state - single 0 (can be used for referral and specify description via a text argument)
        * Point - two numbers
        * Rectangle - four numbers
        * Polygon - six or more (even) numbers
        * Mask - using the same format as the toolkit is using (code from the toolkit can be used)
    * Additional lines contain optional arguments in the form of key=value

    Output trajectories

    At the end of the tracking process, the tracker should output a sequence of files.
    For each query object, the mandatory file is “output_<ID>.txt”, which contains the object states, one line per frame in the sequence.
    This means that if the object is not present in a given frame or was even queried at a later frame, the tracker should output “0” for that frame.

    Additionally, the tracker may return optional values for the object in the form of “output_<ID>_<VALUE>.txt”, where each frame's values are provided.
    """
    
    def __init__(self, tracker: Tracker, command: str, log: bool = False, timeout: int = 30, linkpaths: str | list[str] | None = None, envvars: dict[str, str] | None = None, arguments: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initializes the tracker runtime.

        :param tracker: The tracker to run.
        :param command: The command to run the tracker.
        :param log: Whether to log the tracker output.
        :param timeout: The timeout for the tracker process in seconds.
        :param linkpaths: The paths to link to the tracker process.
        :param envvars: The environment variables to set for the tracker process.
        :param arguments: The additional arguments.
        :param kwargs: Additional options; ``convert`` sets the target region type passed to ``convert_region``.
        """
        super().__init__(tracker)
        self.folder = None
        self._command = command
        self._log = log
        self._timeout = timeout
        self._linkpaths = linkpaths
        self._envvars = envvars
        self._arguments = arguments
        self._convert = kwargs.get("convert", None)
        
    def __enter__(self) -> "TrackerFolderRuntime":
        return self
    
    def stop(self) -> None:
        if self.folder is not None:
            shutil.rmtree(self.folder)
            self.folder = None
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.stop()
        
    def run(self, frames: Iterable[Frame], queries: RunQueries) -> RunResult:

        if self.folder is not None:
            raise RuntimeError("Tracker is already running")

        self.folder = make_temporary_folder()

        # Materialize the iterable so we can index and ``len()`` it below.
        frames_list = list(frames)

        # Generate unique query ids for each query object. The folder runtime
        # internally addresses queries by string id (per-object files); the public
        # interface accepts a ``RunQueries`` list, so we project to a dict here.
        query_keys = [f"obj{i}" for i in range(len(queries))]
        queries_by_id: dict[str, ObjectQuery] = {key: query for query, key in zip(queries, query_keys)}

        generate_input_data(self.folder, frames_list, queries_by_id, convert=self._convert)

        # For consistency with the online tracker runtime,
        # we set the timeout to be the specified timeout 
        # multiplied by the number of frames, if the timeout 
        # is specified and greater than 0. Otherwise, we set it to None (no timeout).
        timeout = None if self._timeout is None or self._timeout <= 0 else self._timeout * len(frames_list)

        environment = dict(os.environ)
        if self._envvars is not None:
            environment.update(self._envvars)
        
        start_time = time.time()
        
        log_debug(f"Running tracker with command: {self._command} in folder: {self.folder}")
        
        process = spawn_process(self._command, self.folder, environment)
        try:
            
            output_data, _ = process.communicate(timeout=timeout)

            total_time = time.time() - start_time
            frame_time = total_time / len(frames_list)

        except subprocess.TimeoutExpired as te:
            process.kill()
            process.communicate()
            self.stop()
            raise RuntimeError("Tracker process timed out") from te
        
        if process.returncode != 0:
            output_data = output_data.decode("utf-8") if output_data is not None else ""

            log_debug(f"Tracker process output:\n{output_data}")
            self.stop()
            
            raise RuntimeError(f"Tracker process exited with code {process.returncode}. Output:\n{output_data}")
        
        try:
            
            objects = parse_outputs(self.folder, list(queries_by_id.keys()))
            
            objects = [objects[key] for key in query_keys]
            
            # Verify the output trajectories have the same length as the number of frames
            for obj in objects:
                if len(obj) != len(frames_list):
                    self.stop()
                    raise RuntimeError(f"Output trajectory length {len(obj)} does not match number of frames {len(frames_list)}")
            
            return RunResult(objects, [frame_time] * len(frames_list))
        except Exception as e:
            self.stop()
            raise RuntimeError("Failed to parse tracker output") from e
    
    @property      
    def multiobject(self) -> bool:
        return True
    
from vot.tracker import register_runtime_protocol
    
from vot.tracker.adapters import PythonAdapter, MatlabAdapter, OctaveAdapter

FolderPython = PythonAdapter(TrackerFolderRuntime)
FolderMatlab = MatlabAdapter(TrackerFolderRuntime)
FolderOctave = OctaveAdapter(TrackerFolderRuntime)

register_runtime_protocol("folder", TrackerFolderRuntime)
register_runtime_protocol("folderpython", FolderPython)
register_runtime_protocol("foldermatlab", FolderMatlab)
register_runtime_protocol("folderoctave", FolderOctave)
