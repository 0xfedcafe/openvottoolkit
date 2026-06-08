"""Storage abstraction for the workspace."""

import os
import pickle

from abc import ABC, abstractmethod
import typing

import cachetools

from attributee.object import class_fullname

from ..experiment import Experiment
from ..dataset import Sequence
from ..tracker import Tracker, Results

CacheKey = str | tuple[typing.Any, ...]


def _storage_path_segment(value: typing.Any) -> str | None:
    """Convert internal cache/directory keys to storage path segments."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return class_fullname(value)


class Storage(ABC):
    """Abstract superclass for workspace storage abstraction."""

    @abstractmethod
    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence) -> Results:
        """Returns results object for the given tracker, experiment, sequence
        combination.

        :param tracker: Selected tracker
        :type tracker: Tracker
        :param experiment: Selected experiment
        :type experiment: Experiment
        :param sequence: Selected sequence
        :type sequence: Sequence
        """
        pass

    @abstractmethod
    def documents(self) -> list[str]:
        """Lists documents in the storage."""
        pass

    @abstractmethod
    def folders(self) -> list[str]:
        """Lists folders in the storage."""
        pass

    @abstractmethod
    def write(self, name: str, binary: bool = False) -> typing.IO:
        """Opens the given file entry for writing, returns opened handle.

        :param name: File name.
        :type name: str
        :param binary: Open file in binary mode. Defaults to False.
        :type binary: bool, optional
        """
        pass

    @abstractmethod
    def read(self, name: str, binary: bool = False) -> typing.IO | None:
        """Opens the given file entry for reading, returns opened handle.

        :param name: File name.
        :type name: str
        :param binary: Open file in binary mode. Defaults to False.
        :type binary: bool, optional
        """
        pass

    @abstractmethod
    def isdocument(self, name: str) -> bool:
        """Checks if given name is a document/file in this storage.

        :param name: Name of the entry to check
        :type name: str

        :returns: Returns True if entry is a document, False otherwise.
        :rtype: bool"""

    @abstractmethod
    def isfolder(self, name: str) -> bool:
        """Checks if given name is a folder in this storage.

        :param name: Name of the entry to check
        :type name: str

        :returns: Returns True if entry is a folder, False otherwise.
        :rtype: bool"""

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Deletes a given document.

        :param name: File name.
        :type name: str


        :returns: Returns True if successful, False otherwise.
        :rtype: bool"""

    @abstractmethod
    def substorage(self, name: str) -> "Storage":
        """Returns a substorage, storage object with root in a subfolder.

        :param name: Name of the entry, must be a folder
        :type name: str

        :returns: Storage object
        :rtype: Storage"""

    @abstractmethod
    def copy(self, localfile: str, destination: str) -> None:
        """Copy a document to another location.

        :param localfile: Original location
        :type localfile: str
        :param destination: New location
        :type destination: str
        """


class FilesystemStorage(Storage):
    """Storage that can expose local filesystem directories.

    The base Storage contract only deals with documents and folders. Some parts of
    the toolkit need an actual local directory path for APIs that do not speak the
    Storage abstraction, so that capability is modeled separately.
    """

    @abstractmethod
    def substorage(self, name: str) -> "FilesystemStorage":
        """Returns a filesystem-backed substorage."""

    @abstractmethod
    def directory(self, *args: typing.Any) -> str:
        """Returns a local filesystem directory path in this storage."""


class NullStorage(Storage):
    """An implementation of dummy storage that does not save anything."""

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence) -> Results:
        """Returns results object for the given tracker, experiment, sequence
        combination."""
        return Results(self)

    def __repr__(self) -> str:
        """Returns a string representation of the storage object."""
        return "<Null storage>"

    def write(self, name: str, binary: bool = False) -> typing.IO:
        """Opens the given file entry for writing, returns opened handle."""
        if binary:
            return open(os.devnull, "wb")
        else:
            return open(os.devnull, "w", encoding="utf-8")

    def documents(self) -> list[str]:
        """Lists documents in the storage."""
        return []

    def folders(self) -> list[str]:
        """Lists folders in the storage. Reuturns an empty list.

        :returns: Empty list
        :rtype: list"""
        return []

    def read(self, name: str, binary: bool = False) -> typing.IO | None:
        """Opens the given file entry for reading, returns opened handle.

        :returns: Returns None.
        :rtype: None"""
        return None

    def isdocument(self, name: str) -> bool:
        """Checks if given name is a document/file in this storage.

        :returns: Returns False.
        :rtype: bool"""
        return False

    def isfolder(self, name: str) -> bool:
        """Checks if given name is a folder in this storage.

        :returns: Returns False.
        :rtype: bool"""
        return False

    def delete(self, name: str) -> bool:
        """Deletes a given document.

        :returns: Returns False since nothing is deleted.
        :rtype: bool"""
        return False

    def substorage(self, name: str) -> "Storage":
        """Returns a substorage, storage object with root in a subfolder."""
        return NullStorage()

    def copy(self, localfile: str, destination: str) -> None:
        """Copy a document to another location.

        Does nothing.
        """
        return


class LocalStorage(FilesystemStorage):
    """Storage backed by the local filesystem.

    This is the default real storage implementation.
    """

    def __init__(self, root: str) -> None:
        """Creates a new local storage object.

        :param root: Root path of the storage.
        :type root: str
        """
        self._root = root
        self._results = os.path.join(root, "results")

    def __repr__(self) -> str:
        """Returns a string representation of the storage object."""
        return "<Local storage: {}>".format(self._root)

    @property
    def base(self) -> str:
        """Returns the base path of the storage."""
        return self._root

    def results(self, tracker: Tracker, experiment: Experiment, sequence: Sequence) -> Results:
        """Returns results object for the given tracker, experiment, sequence
        combination.

        :param tracker: Selected tracker
        :type tracker: Tracker
        :param experiment: Selected experiment
        :type experiment: Experiment
        :param sequence: Selected sequence
        :type sequence: Sequence

        :returns: Results object
        :rtype: Results"""
        storage = LocalStorage(os.path.join(self._results, tracker.reference, experiment.identifier, sequence.name))
        return Results(storage)

    def _list_entries(self, predicate: typing.Callable[[str], bool]) -> list[str]:
        """Names of entries directly under the root whose full path satisfies ``predicate``."""
        if not os.path.isdir(self._root):
            return []
        return [name for name in os.listdir(self._root) if predicate(os.path.join(self._root, name))]

    def documents(self) -> list[str]:
        """Lists documents in the storage.

        :returns: List of document names.
        :rtype: list"""
        return self._list_entries(os.path.isfile)

    def folders(self) -> list[str]:
        """Lists folders in the storage.

        :returns: List of folder names.
        :rtype: list"""
        return self._list_entries(os.path.isdir)

    def write(self, name: str, binary: bool = False) -> typing.IO:
        """Opens the given file entry for writing, returns opened handle.

        :param name: File name.
        :type name: str
        :param binary: Open file in binary mode. Defaults to False.
        :type binary: bool, optional

        :returns: Opened file handle.
        :rtype: file"""
        full = os.path.join(self.base, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        if binary:
            return open(full, mode="wb")
        else:
            return open(full, mode="w", newline="", encoding="utf-8")

    def read(self, name: str, binary: bool = False) -> typing.IO | None:
        """Opens the given file entry for reading, returns opened handle.

        :param name: File name.
        :type name: str
        :param binary: Open file in binary mode. Defaults to False.
        :type binary: bool, optional

        :returns: Opened file handle.
        :rtype: file"""
        full = os.path.join(self.base, name)

        if binary:
            return open(full, mode="rb")
        else:
            return open(full, mode="r", newline="")

    def delete(self, name: str) -> bool:
        """Deletes a given document. Returns True if successful, False otherwise.

        :param name: File name.
        :type name: str

        :returns: Returns True if successful, False otherwise.
        :rtype: bool"""
        full = os.path.join(self.base, name)
        if os.path.isfile(full):
            os.unlink(full)
            return True
        return False

    def isdocument(self, name: str) -> bool:
        """Checks if given name is a document/file in this storage.

        :param name: Name of the entry to check
        :type name: str

        :returns: Returns True if entry is a document, False otherwise.
        :rtype: bool"""
        return os.path.isfile(os.path.join(self._root, name))

    def isfolder(self, name: str) -> bool:
        """Checks if given name is a folder in this storage.

        :param name: Name of the entry to check
        :type name: str

        :returns: Returns True if entry is a folder, False otherwise.
        :rtype: bool"""
        return os.path.isdir(os.path.join(self._root, name))

    def substorage(self, name: str) -> "LocalStorage":
        """Returns a substorage, storage object with root in a subfolder.

        :param name: Name of the entry, must be a folder
        :type name: str

        :returns: Storage object
        :rtype: Storage"""
        return LocalStorage(os.path.join(self.base, name))

    def copy(self, localfile: str, destination: str) -> None:
        """Copy a document to another location in the storage.

        :param localfile: Original location
        :type localfile: str
        :param destination: New location
        :type destination: str

        :raises IOError: If the destination is an absolute path."""
        import shutil
        if os.path.isabs(destination):
            raise IOError("Only relative paths allowed")

        full = os.path.join(self.base, destination)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        shutil.copyfile(localfile, full)

    def directory(self, *args: typing.Any) -> str:
        """Returns a path to a directory in the storage.

        :param *args: Path segments.

        :returns: Path to the directory.
        :rtype: str
        :raises ValueError: If the path is not a directory."""
        segments = [_storage_path_segment(arg) for arg in args]
        segments = [segment for segment in segments if segment is not None]

        path = os.path.join(self._root, *segments)
        os.makedirs(path, exist_ok=True)

        return path


class Cache(cachetools.Cache):
    """Persistent cache, extends the cache from cachetools package by storing cached
    objects (using picke serialization) to the underlying storage."""

    def __init__(self, storage: Storage) -> None:
        """Creates a new cache backed by the given storage.

        :param storage: The storage used to save objects.
        :type storage: Storage
        """
        super().__init__(10000)
        self._storage = storage

    def _filename(self, key: CacheKey) -> str:
        """Generates a filename for the given object key.

        :param key: Cache key, either tuple or a single string
        :type key: tuple | str

        :returns: Relative path as a string
        :rtype: str"""
        if isinstance(key, tuple):
            segments = [_storage_path_segment(item) for item in key]
        else:
            segments = [_storage_path_segment(key)]

        segments = [segment for segment in segments if segment is not None]
        if not segments:
            raise ValueError("Cache key must contain at least one path segment")

        return os.path.join(*segments)

    def __getitem__(self, key: CacheKey) -> typing.Any:
        """Retrieves an image from cache. If it does not exist, a KeyError is raised.

        :param key: Key of the item
        :type key: str

        :raises KeyError: Entry does not exist or cannot be retrieved
        :raises PickleError: Unable to
        :returns: item value
        :rtype: typing.Any"""
        try:
            return super().__getitem__(key)
        except KeyError as e:
            filename = self._filename(key)
            if not self._storage.isdocument(filename):
                raise e
            try:
                filehandle = self._storage.read(filename, binary=True)
                if filehandle is None:
                    raise KeyError(filename)
                with filehandle:
                    data = pickle.load(filehandle)
                    super().__setitem__(key, data)
                    return data
            except (pickle.PickleError, EOFError, AttributeError, ImportError, ValueError) as pe:
                raise KeyError(pe) from e
            except IOError as ie:
                raise KeyError(ie) from e

    def __setitem__(self, key: CacheKey, value: typing.Any) -> None:
        """Sets an item for given key.

        :param key: Item key
        :type key: str
        :param value: Item value
        :type value: typing.Any
        """
        super().__setitem__(key, value)

        try:
            payload = pickle.dumps(value)
            filename = self._filename(key)
            with self._storage.write(filename, binary=True) as filehandle:
                filehandle.write(payload)
        except (pickle.PickleError, TypeError):
            pass

    def __delitem__(self, key: CacheKey) -> None:
        """Operator for item deletion.

        :param key: Key of object to remove
        :type key: str
        """
        filename = self._filename(key)

        if super().__contains__(key):
            super().__delitem__(key)

        try:
            self._storage.delete(filename)
        except IOError:
            pass

    def __contains__(self, key: object) -> bool:
        """Magic method, does the cache include an item for a given key.

        :param key: Item key
        :type key: CacheKey

        :returns: True if object exists for a given key
        :rtype: bool"""
        if not isinstance(key, (str, tuple)):
            return False
        filename = self._filename(key)
        return super().__contains__(key) or self._storage.isdocument(filename)
