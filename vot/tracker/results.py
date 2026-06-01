"""Results module for storing and retrieving tracker results."""

import os
import fnmatch
from collections.abc import Iterator
from typing import IO, Any, TYPE_CHECKING, overload
from copy import copy
from vot.region import Region, Special, SpecialCode, calculate_overlap, is_special
from vot.region.io import write_trajectory, read_trajectory
from vot.utilities import to_string

if TYPE_CHECKING:
    from vot.workspace.storage import Storage


class Results(object):
    """Generic results interface for storing and retrieving results."""

    def __init__(self, storage: "Storage") -> None:
        """Creates a new results interface.

        :param storage: Storage interface
        """
        self._storage = storage

    def exists(self, name: str) -> bool:
        """Returns true if the given file exists in the results storage.

        :param name: File name
        :type name: str

        :returns: True if the file exists
        :rtype: bool"""
        return self._storage.isdocument(name)

    def read(self, name: str) -> IO[Any] | None:
        """Returns a file handle for reading the given file from the results storage.

        :param name: File name
        :type name: str

        :returns: File handle
        :rtype: file"""
        if name.endswith(".bin"):
            return self._storage.read(name, binary=True)
        return self._storage.read(name)

    def write(self, name: str) -> IO[Any]:
        """Returns a file handle for writing the given file to the results storage.

        :param name: File name
        :type name: str

        :returns: File handle
        :rtype: file"""
        if name.endswith(".bin"):
            return self._storage.write(name, binary=True)
        return self._storage.write(name)

    def find(self, pattern: str) -> list[str]:
        """Returns a list of files matching the given pattern in the results storage.

        :param pattern: Pattern
        :type pattern: str

        :returns: List of files
        :rtype: list"""

        return fnmatch.filter(self._storage.documents(), pattern)
    
class Trajectory(object):
    """Trajectory class for storing and retrieving tracker trajectories."""

    @classmethod
    def exists(cls, results: Results, name: str) -> bool:
        """Returns true if the trajectory exists in the results storage.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str

        :returns: True if the trajectory exists
        :rtype: bool"""
        return results.exists(name + ".bin") or results.exists(name + ".txt")

    @classmethod
    def gather(cls, results: Results, name: str) -> list:
        """Returns a list of files that are part of the trajectory.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str

        :returns: List of files
        :rtype: list"""

        if results.exists(name + ".bin"):
            files = [name + ".bin"]
        elif results.exists(name + ".txt"):
            files = [name + ".txt"]
        else:
            return []

        for propertyfile in results.find(name + "_*.value"):
            files.append(propertyfile)

        return files

    @classmethod
    def read(cls, results: Results, name: str) -> 'Trajectory':
        """Reads a trajectory from the results storage.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str

        :returns: Trajectory
        :rtype: Trajectory"""

        def parse_float(line: str) -> float | None:
            """Parses a float from a line.

            :param line: Line
            :type line: str

            :returns: Float value
            :rtype: float"""
            if not line.strip():
                return None
            return float(line.strip())

        if results.exists(name + ".bin"):
            fp = results.read(name + ".bin")
            assert fp is not None, "read() returned None despite exists() returning True"
            with fp:
                regions = read_trajectory(fp)
        elif results.exists(name + ".txt"):
            fp = results.read(name + ".txt")
            assert fp is not None, "read() returned None despite exists() returning True"
            with fp:
                regions = read_trajectory(fp)
        else:
            raise FileNotFoundError("Trajectory data not found: {}".format(name))

        trajectory = Trajectory(len(regions))
        trajectory._regions = regions

        for propertyfile in results.find(name + "_*.value"):
            filehandle = results.read(propertyfile)
            assert filehandle is not None, "read() returned None despite find() reporting the file"
            with filehandle:
                propertyname = os.path.splitext(os.path.basename(propertyfile))[0][len(name)+1:]
                lines = list(filehandle.readlines())
                try:
                    trajectory._properties[propertyname] = [parse_float(line) for line in lines]
                except ValueError:
                    trajectory._properties[propertyname] = [line.strip() for line in lines]

        return trajectory

    def __init__(self, length: int) -> None:
        """Creates a new trajectory of the given length.

        :param length: Trajectory length
        :type length: int
        """
        # ``Special`` is a subclass of ``Region`` — type the list with the
        # general element type so subsequent assignments of other ``Region``
        # subclasses (``Rectangle`` etc.) are accepted.
        self._regions: list[Region] = [Special(SpecialCode.UNKNOWN)] * length
        self._properties: dict = dict()

    def set(self, frame: int, region: Region, properties: dict | None = None) -> None:
        """Sets the region for the given frame.

        :param frame: Frame index
        :param region: Region
        :param properties: Frame properties. Defaults to None.

        :raises IndexError: Frame index out of bounds"""
        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")

        self._regions[frame] = region

        if properties is None:
            properties = dict()

        for k, v in properties.items():
            if k not in self._properties:
                self._properties[k] = [None] * len(self._regions)
            self._properties[k][frame] = v

    def region(self, frame: int) -> Region:
        """Returns the region for the given frame.

        :param frame: Frame index

        :raises IndexError: Frame index out of bounds
        :returns: Region"""
        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")
        return self._regions[frame]

    def regions(self) -> list[Region]:
        """Returns the list of regions."""
        return copy(self._regions)

    def markers(self) -> tuple[list[int], list[int], list[int]]:
        """Returns the ascending frame indices of every INITIALIZATION, FAILURE and
        CRASH marker as ``(init, failure, crash)``.

        Both FAILURE and CRASH terminate a tracking run for EAO purposes; only FAILURE
        is a tracking failure (low-overlap loss) while CRASH is a tracker process
        failure. For the robustness / crash counts use :meth:`failures` / :meth:`crashes`,
        which apply run-pairing semantics rather than reporting raw markers.

        :returns: ``(init_idxs, failure_idxs, crash_idxs)``."""
        init_idxs: list[int] = []
        failure_idxs: list[int] = []
        crash_idxs: list[int] = []
        for i, region in enumerate(self._regions):
            if is_special(region, SpecialCode.INITIALIZATION):
                init_idxs.append(i)
            elif is_special(region, SpecialCode.FAILURE):
                failure_idxs.append(i)
            elif is_special(region, SpecialCode.CRASH):
                crash_idxs.append(i)
        return init_idxs, failure_idxs, crash_idxs

    def failures(self) -> list[int]:
        """Returns the frame indices of counted tracking failures.

        A counted failure is a FAILURE marker that terminates an open run, i.e. one
        preceded by an INITIALIZATION not already paired with an earlier failure, so
        ``len(trajectory.failures())`` is the VOT robustness count. A CRASH does not
        close a run here; crashes are reported separately by :meth:`crashes`.

        :returns: Ascending frame indices of the counted tracking failures."""
        frames: list[int] = []
        in_run = False
        for i, region in enumerate(self._regions):
            if is_special(region, SpecialCode.INITIALIZATION):
                in_run = True
            elif is_special(region, SpecialCode.FAILURE) and in_run:
                frames.append(i)
                in_run = False
        return frames

    def crashes(self) -> list[int]:
        """Returns the frame indices of every CRASH marker.

        A crash is written when the runtime raises a tracker exception (a process
        crash or timeout) during initialize or update. Orphan crashes during a reinit
        attempt count too — the tracker failed to produce output in either case.

        :returns: Ascending frame indices of the CRASH markers."""
        return [i for i, region in enumerate(self._regions) if is_special(region, SpecialCode.CRASH)]

    @overload
    def properties(self, frame: None = ...) -> tuple[str, ...]: ...
    @overload
    def properties(self, frame: int) -> dict[str, Any]: ...
    def properties(self, frame: int | None = None) -> tuple[str, ...] | dict[str, Any]:
        """Returns the properties for the given frame, or a tuple of property names
        when ``frame`` is ``None``.

        :param frame: Frame index. Defaults to None.

        :raises IndexError: Frame index out of bounds
        :returns: Properties dict for a frame, or tuple of property names."""

        if frame is None:
            return tuple(self._properties.keys())

        if frame < 0 or frame >= len(self._regions):
            raise IndexError("Frame index out of bounds")

        return {k: v[frame] for k, v in self._properties.items() if v[frame] is not None}

    def __len__(self) -> int:
        """Returns the length of the trajectory.

        :returns: Length
        :rtype: int"""
        return len(self._regions)
    
    def __iter__(self) -> Iterator[Region]:
        """Returns an iterator over the regions.

        :returns: Iterator
        :rtype: Iterator"""
        return iter(self._regions)

    def write(self, results: Results, name: str) -> None:
        """Writes the trajectory to the results storage.

        :param results: Results storage
        :type results: Results
        :param name: Trajectory name (without extension)
        :type name: str
        """
        from vot import config

        if config.results_binary:
            with results.write(name + ".bin") as fp:
                write_trajectory(fp, self._regions)
        else:
            with results.write(name + ".txt") as fp:
                # write_trajectory_file(fp, self._regions)
                write_trajectory(fp, self._regions)

        for k, v in self._properties.items():
            with results.write(name + "_" + k + ".value") as fp:
                fp.writelines([to_string(e) + "\n" for e in v])


    def equals(self, trajectory: 'Trajectory', check_properties: bool = False, overlap_threshold: float = 0.99999) -> bool:
        """Returns true if the trajectories are equal.

        :param trajectory: The other trajectory to compare against.
        :type trajectory: Trajectory
        :param check_properties: Also require per-frame properties to match. Defaults to False.
        :type check_properties: bool, optional
        :param overlap_threshold: Minimum per-region overlap to treat two regions as equal. Defaults to 0.99999.
        :type overlap_threshold: float, optional

        :returns: True if the trajectories are equal, False otherwise.
        :rtype: bool"""
        if not len(self) == len(trajectory):
            return False

        for r1, r2 in zip(self.regions(), trajectory.regions()):
            if calculate_overlap(r1, r2) < overlap_threshold and not (is_special(r1) and is_special(r2)):
                return False

        if check_properties:
            if not set(self._properties.keys()) == set(trajectory._properties.keys()):
                return False
            for name, _ in self._properties.items():
                for p1, p2 in zip(self._properties[name], trajectory._properties[name]):
                    if not p1 == p2 and not (p1 is None and p2 is None):
                        return False
        return True
