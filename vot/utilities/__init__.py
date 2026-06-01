"""This module contains various utility functions and classes used throughout the
toolkit."""

import os
import sys
import csv
import re
import hashlib
import logging
import inspect
import time
import concurrent.futures as futures
from logging import Formatter, LogRecord
from types import TracebackType

from numbers import Number
from typing import Any, Mapping
import typing

import colorama

from class_registry import ClassRegistry

def import_class(classpath: str) -> typing.Type:
    """Import a class from a string by importing parent packages.

    :param classpath: String representing a canonical class name with all parent packages.
    :type classpath: str

    :raises ImportError: If ``classpath`` has no package qualifier (no ``.``).
    :returns: The imported class.
    :rtype: type"""
    delimiter = classpath.rfind(".")
    if delimiter == -1:
        raise ImportError("Class alias '{}' not found".format(classpath))
    else:
        classname = classpath[delimiter+1:len(classpath)]
        module = __import__(classpath[0:delimiter], globals(), locals(), [classname])
        return getattr(module, classname)

def class_fullname(o: object) -> str:
    """Returns the full name of the class of the given object.

    :param o: The object to get the class name from.

    :returns: The full name of the class of the given object."""
    return class_string(o.__class__)

def class_string(kls: type) -> str:
    """Returns the full name of the given class.

    :param kls: The class to get the name from.

    :returns: The full name of the given class."""
    assert inspect.isclass(kls)
    module = kls.__module__
    if module is None or module == str.__class__.__module__:
        return kls.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + kls.__name__

def flip(size: tuple[Number, Number]) -> tuple[Number, Number]:
    """Flips the given size tuple.

    :param size: The size tuple to flip.

    :returns: The flipped size tuple."""
    return (size[1], size[0])

def flatten(nested_list: list) -> list:
    """Flattens a nested list.

    :param nested_list: The nested list to flatten.

    :returns: The flattened list."""
    return [item for sublist in nested_list for item in sublist]


class Progress(object):
    """Wrapper around tqdm progress bar, enables silecing the progress output and some
    more costumizations."""

    def __init__(self, description: str ="Processing", total: int = 100) -> None:
        """Creates a new progress bar.

        :param description: The description of the progress bar.
        :param total: The total number of steps.
        """
        
        from vot.utilities.notebook import is_notebook
        
        from vot import get_logger
        
        if is_notebook():
            try:
                from tqdm._tqdm_notebook import tqdm_notebook as tqdm
            except ImportError:
                from tqdm import tqdm
        else:
            from tqdm import tqdm
        
        silent = get_logger().level > logging.INFO

        if not silent:
            self._bar = tqdm(disable=None, # type: ignore[arg-type]
                bar_format=" {desc:20.20} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]", file=sys.stdout, leave=False)
            self._bar.desc = description
            self._bar.total = total
        if silent or (self._bar and self._bar.disable):
            self._bar = None
            self._value = 0
            self._total = total if not silent else 0

    def _percent(self, n: int) -> int:
        """Returns the percentage of the given value.

        :param n: The value to compute the percentage of.

        :returns: The percentage of the given value."""
        return int((n * 100) / self._total)

    def absolute(self, value: int) -> None:
        """Sets the progress to the given value.

        :param value: The value to set the progress to.
        """
        if self._bar is None:
            if self._total == 0:
                return
            prev = self._value
            self._value = max(0, min(value, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._bar.update(value - self._bar.n)  # will also set self.n = b * bsize
        
    def relative(self, n: int) -> None:
        """Increments the progress by the given value.

        :param n: The value to increment the progress by.
        """
        if self._bar is None:
            if self._total == 0:
                return
            prev = self._value
            self._value = max(0, min(self._value + n, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._bar.update(n)  # will also set self.n = b * bsize 

    def total(self, t: int) -> None:
        """Sets the total number of steps.

        :param t: The total number of steps.
        """
        if self._bar is None:
            if self._total == 0:
                return
            self._total = t
        else:
            if self._bar.total == t:
                return
            self._bar.total = t
            self._bar.refresh()

    def close(self) -> None:
        """Closes the progress bar."""
        if self._bar:
            self._bar.close()
            
    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()
        

def extract_files(archive: str, destination: str, callback: typing.Callable[[int, int], None] | None = None) -> None:
    """Extracts all files from the given archive to the given destination.

    :param archive: The archive to extract the files from.
    :param destination: The destination to extract the files to.
    :param callback: An optional callback function that is called after each file is extracted.
    """
    from zipfile import ZipFile

    with ZipFile(file=archive) as zip_file:
        # Loop over each file
        total=len(zip_file.namelist())
        for file in zip_file.namelist():

            # Extract each file to another directory
            # If you want to extract to current working directory, don't specify path
            zip_file.extract(member=file, path=destination)
            if callback:
                callback(1, total)

def read_properties(filename: str, delimiter: str = '=') -> dict[str, str]:
    """Reads a given properties file with each line of the format key=value. Returns a
    dictionary containing the pairs.

    :param filename: The name of the file to be read.
    :type filename: str
    :param delimiter: Key-value delimiter. Defaults to '='.
    :type delimiter: str, optional

    :returns: Resuting properties as a dictionary
    :rtype: [dict[str, str]]"""
    if not os.path.exists(filename):
        return {}
    matcher = re.compile("^([a-zA-Z0-9_\\-.]+) *{} *(.*)$".format(delimiter))
    with open(filename, mode='r', newline='') as pfile:
        properties = dict()
        for line in pfile.readlines():
            groups = matcher.match(line.strip())
            if not groups:
                continue
            properties[groups.group(1)] = groups.group(2)
        return properties

def write_properties(filename: str, dictionary: Mapping[str, Any], delimiter: str = '=') -> None:
    """Writes the provided dictionary in key sorted order to a properties
        file with each line in the format: key<delimiter>value

    :param filename: the name of the file to be written
    :type filename: str
    :param dictionary: a dictionary containing the key/value pairs.
    :type dictionary: Mapping[str, str]
    :param delimiter: _description_. Defaults to '='.
    :type delimiter: str, optional
    """

    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, escapechar='\\',
                            quoting=csv.QUOTE_NONE)
        writer.writerows(sorted(dictionary.items()))

def file_hash(filename: str) -> tuple[str, str]:
    """Calculates MD5 and SHA1 hashes based on file content.

    :param filename: Filename of the file to open and analyze
    :type filename: str

    :returns: MD5 and SHA1 hashes as hexadecimal strings.
    :rtype: tuple[str, str]"""
    
    bufsize = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()
    sha1 = hashlib.sha1()

    with open(filename, 'rb') as f:
        while True:
            data = f.read(bufsize)
            if not data:
                break
            md5.update(data)
            sha1.update(data)

    return md5.hexdigest(), sha1.hexdigest()

def arg_hash(*args: typing.Any, **kwargs: typing.Any) -> str:
    """Computes hash based on input positional and keyword arguments.

    The algorithm tries to convert all arguments to string, then enclose them with delimiters. The
    positonal arguments are listed as is, keyword arguments are sorted and encoded with their keys as
    well as values.

    :returns: SHA1 hash as hexadecimal string
    :rtype: str"""
    sha1 = hashlib.sha1()

    for arg in args:
        sha1.update(("(" + str(arg) + ")").encode("utf-8"))

    for (key, val) in sorted(kwargs.items()):
        sha1.update(("(" + str(key) + ":" + str(val) + ")").encode("utf-8"))

    return sha1.hexdigest()

def which(program: str) -> str | None:
    """Locates an executable in system PATH list by its name.

    :param program: Name of the executable

    :returns: Full path or None if not found."""

    def is_exe(fpath: str) -> bool:
        """Checks if the given path is an executable file.

        :param fpath: Path to check
        :type fpath: str

        :returns: True if the path is an executable file
        :rtype: bool"""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def normalize_path(path: str, root: str | None = None) -> str:
    """Normalizes the given path by making it absolute and removing redundant parts.

    :param path: Path to normalize
    :type path: str
    :param root: Root path to use if the given path is relative. Defaults to None.
    :type root: str, optional

    :returns: Normalized path
    :rtype: str"""
    if os.path.isabs(path):
        return path
    if not root:
        root = os.getcwd()
    return os.path.normpath(os.path.join(root, path))

def localize_path(path: str) -> str:
    """Converts path to local format (backslashes on Windows, slashes on Linux)

    :param path: Path to convert
    :type path: str

    :returns: Converted path
    :rtype: str"""
    if sys.platform.startswith("win"):
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")

def to_string(n: Any) -> str:
    """Converts object to string, returs empty string if object is None (so a bit
    different behaviour than the original string conversion).

    :param n: Object of any kind
    :type n: Any

    :returns: String representation (using built-in conversion)
    :rtype: str"""
    if n is None:
        return ""
    else:
        return str(n)

def to_number(
    val: typing.Any,
    max_n: float | None = None,
    min_n: float | None = None,
    conversion: typing.Callable[[typing.Any], typing.Any] = int,
) -> typing.Any:
    """Converts the given value to a number and checks if it is within the given range.
    If the value is not a number, a RuntimeError is raised.

    :param val: Value to convert
    :param max_n: Maximum allowed value
    :param min_n: Minimum allowed value
    :param conversion: Conversion function (e.g. ``int`` or ``float``).

    :returns: Converted value (same type as ``conversion``'s return)."""
    try:
        n = conversion(val)

        if max_n is not None and n > max_n:
            raise RuntimeError("Parameter higher than maximum allowed value ({}>{})".format(n, max_n))
        if min_n is not None and n < min_n:
            raise RuntimeError("Parameter lower than minimum allowed value ({}<{})".format(n, min_n))

        return n
    except ValueError as ve:
        raise RuntimeError("Number conversion error") from ve

def to_logical(val: typing.Any) -> bool:
    """Converts the given value to a logical value (True/False).

    Strings are matched case-insensitively against ``true``/``1``/``t``/``y``/``yes``
    (anything else is False); non-string values use the built-in ``bool``.

    :param val: Value to convert
    :type val: Any

    :returns: Converted value
    :rtype: bool"""
    if isinstance(val, str):
        return val.lower() in ['true', '1', 't', 'y', 'yes']
    return bool(val)

def format_size(num: float, suffix: str = "B") -> str:
    """Formats the given number as a human-readable size string.

    :param num: Number to format
    :type num: float
    :param suffix: Suffix to use. Defaults to "B".
    :type suffix: str, optional

    :returns: Formatted string
    :rtype: str"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def singleton(class_: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Singleton decorator for classes.

    :param class_: Class to decorate
    :type class_: class

    :returns: Decorated class
    :rtype: class
    Example:
        @singleton
        class MyClass:
            pass

        a = MyClass()
    """
    instances: dict[typing.Callable[..., typing.Any], typing.Any] = {}
    def getinstance(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Returns the singleton instance of the class.

        If the instance does not exist, it is created.
        """
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class ColoredFormatter(Formatter):
    """Colored log formatter using colorama package."""

    def __init__(self, **kwargs: typing.Any) -> None:
        """Initializes the formatter.

        :param **kwargs: Keyword arguments passed to the base class
        """
        super().__init__(**kwargs)
        colorama.init()

        self._styles = dict(
            debug=colorama.Fore.GREEN,
            verbose=colorama.Fore.BLACK,
            info="",
            notice=colorama.Fore.MAGENTA,
            warning=colorama.Fore.YELLOW,
            error=colorama.Fore.RED,
            critical=colorama.Fore.RED + colorama.Style.BRIGHT,
        )

    def format(self, record: LogRecord) -> str:
        """Formats message by injecting colorama terminal codes for text coloring.

        :param record: Input log record
        :type record: LogRecord

        :returns: Formatted string
        :rtype: str"""
        import copy as _copy

        style = self._styles[record.levelname.lower()]

        record_copy = _copy.copy(record)
        msg = record.msg if isinstance(record.msg, str) else str(record.msg)
        record_copy.msg = style + msg + colorama.Style.RESET_ALL
        # Delegate the remaining formatting to the base formatter.
        return Formatter.format(self, record_copy)


def arm_parent_watchdog(poll_seconds: float = 2.0) -> None:
    """Arm a daemon-thread watchdog that terminates this worker when its parent dies.

    Captures the current parent PID and polls ``os.getppid()`` every
    ``poll_seconds``. When the PID changes — i.e. the parent died and the OS
    reparented the worker to init/launchd — the worker calls ``os._exit(0)``
    instead of waiting on a now-broken pipe.

    Why: :class:`concurrent.futures.ProcessPoolExecutor` workers are not
    daemons, so when the parent is SIGKILL'd or its controlling terminal is
    closed, the workers keep running a CPU-bound chunk (overlap rasterization,
    EAO subcompute, etc.) until they try to send the result back. That can
    take many minutes per chunk and produces orphaned 100%-CPU workers
    visible in ``htop`` long after the user thinks ``vot`` is done.

    Safe to call multiple times; each call starts an additional daemon thread
    but they share the same exit condition.
    """
    import threading

    parent_pid = os.getppid()

    def _watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                os._exit(0)
            time.sleep(poll_seconds)

    threading.Thread(target=_watch, name="parent-watchdog", daemon=True).start()


class ThreadPoolExecutor(futures.ThreadPoolExecutor):
    """Thread pool executor with a shutdown method that waits for all threads to
    finish."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Initializes the thread pool executor."""
        super().__init__(*args, **kwargs)
        #self._work_queue = Queue.Queue(maxsize=maxsize)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Shuts down the thread pool executor. If wait is True, waits for all threads
        to finish.

        :param wait: Wait for all threads to finish. Defaults to True.
        :param cancel_futures: Cancel pending futures. Accepted for parent-class
            compatibility; pending work items are always cancelled here.
        """
        import queue
        del cancel_futures
        with self._shutdown_lock:
            self._shutdown = True
            try:
                while True:
                    item = self._work_queue.get_nowait()
                    item.future.cancel()
            except queue.Empty:
                pass
            self._work_queue.put(None)  # type: ignore[arg-type]
        if wait:
            for t in self._threads:
                t.join()

class Timer(object):
    """Simple timer class for measuring elapsed time."""

    def __init__(self, name: str | None = None) -> None:
        """Initializes the timer.

        :param name: Name of the timer. Defaults to None.
        :type name: str, optional
        """
        self.name = name

    def __enter__(self) -> None:
        """Starts the timer."""
        self._tstart = time.time()

    def __exit__(self, type: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        """Stops the timer and prints the elapsed time."""
        elapsed = time.time() - self._tstart
        if self.name:
            print('[%s]: %.4fs' % (self.name, elapsed))
        else:
            print('Elapsed: %.4fs' % elapsed)

class Registry(ClassRegistry):
    """A class registry for storing classes with a fallback to entry point registry."""

    def __init__(self, group: str, attr_name: str | None = None) -> None:
        """Initializes the registry.

        :param group: The name of the entry point group that will be used to load new classes.
        :type group: str
        :param attr_name: If set, the registry will "brand" each class with its corresponding registry key. Defaults to None.
        :type attr_name: str | None, optional
        """
        from class_registry.entry_points import EntryPointClassRegistry
        super(Registry, self).__init__(attr_name=attr_name)
        
        import vot
        import json
        from vot import get_logger
    
        # Handle development mode where alias entries are in the data directory
        root = os.path.dirname(os.path.dirname(os.path.abspath(vot.__file__)))
        alias_file = os.path.join(root, "data", "aliases", group + ".json")
        if os.path.exists(alias_file):
            try:
                with open(alias_file, "r") as f:
                    aliases = json.load(f)
                    for key, value in aliases.items():
                        try:
                            self.register(key)(import_class(value))
                        except Exception as e:
                            get_logger().warning("Failed to import class '{}' for alias '{}': {}".format(value, key, e))
            except Exception:
                pass
        
        # By default use the entry point mechanism
        self._entry_point = EntryPointClassRegistry(group="vot_" + group, attr_name=attr_name)

    def __missing__(self, key: typing.Hashable) -> type:
        """Attempts to load a class from the entry point registry if it is not found in
        the local registry.

        :param key: Key of the class to load

        :raises KeyError: If the key is not found in the entry point registry.
        :returns: Loaded class."""
        return self._entry_point.get_class(key)
 
    def get_class(self, key: typing.Hashable) -> type:
        """Returns the class associated with the specified key. If the class is not
        found in the local registry, it is loaded from the entry point registry.

        :param key: Key of the class to load
        :type key: typing.Hashable

        :returns: Loaded class
        :rtype: typing.Type[T]"""
        try:
            return super().get_class(key)
        except KeyError:
            return self._entry_point.get_class(key)
 
    def keys(self) -> set[typing.Hashable]:
        """Returns the set of registered keys (local registry and entry points).

        :returns: Set of registered keys
        :rtype: set"""
        items = {key for key in self._entry_point.keys()}
        items.update({key for key in super().keys()})

        return items
 
    def classes(self) -> typing.Iterator[type]:
        """Returns an iterator over the registered classes.

        :returns: Iterator over the registered classes
        :rtype: Iterator"""

        items: dict[typing.Hashable, type] = {}
        for key in self._entry_point.keys():
            items[key] = self._entry_point.get_class(key)
            
        inherited_keys = super().keys()
        for key in inherited_keys:
            if key not in items:
                items[key] = super().get_class(key)
     
        return iter(items.values())

    def items(self) -> typing.Iterator[tuple[typing.Hashable, type]]:
        return zip(self.keys(), self.classes())

class ObjectResolver(object):
    """Resolves objects from a registry by name, instantiating them with optional checks.

    A name is looked up in the registry; if absent it is imported as a class path. Optional
    ``class_check`` / ``object_check`` callables validate the class and the instance against
    the resolution context."""

    def __init__(self, registry: ClassRegistry, extra_arguments: typing.Callable | None = None,
                class_check: typing.Callable | None = None, object_check: typing.Callable | None = None) -> None:
        """Initializes the object resolver.

        :param registry: Registry of classes
        :type registry: ClassRegistry
        :param extra_arguments: Extra arguments to pass to the class constructor
        :type extra_arguments: typing.Callable | None, optional
        :param class_check: Function to check if the class is compatible with the purpose
        :type class_check: typing.Callable | None, optional
        :param object_check: Function to check if the object is compatible with the purpose
        :type object_check: typing.Callable | None, optional
        """
        self._registry = registry
        self._extra_arguments = extra_arguments
        self._class_check = class_check
        self._object_check = object_check
    
    
    
    def __call__(self, typename: str, context: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """Resolve an object from a string. If the object is not registered, it is
        imported as a class and instantiated with the provided arguments.

        :param typename: Registry key or importable class path of the object.
        :param context: Context of the resolver, passed to the optional checks.

        :returns: The resolved object."""

        if self._extra_arguments:
            kwargs.update(self._extra_arguments(context))

        if typename in self._registry:
            obj = self._registry.get(typename, **kwargs)
            if self._class_check:
                assert self._class_check(obj, context)
        else:
            obj_class = import_class(typename)
            if self._class_check:
                assert self._class_check(obj_class, context)
            obj = obj_class(**kwargs)

        if self._object_check:
            assert self._object_check(obj, context)

        return obj