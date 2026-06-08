from collections.abc import Callable
from typing import Any

import importlib
import logging
import multiprocessing
import os
import queue
import sys
import traceback
import inspect
import time

from vot.dataset import Frame
from vot.tracker import Tracker, OnlineTrackerRuntime, FrameObjects, ObjectStatus, TrackerException, FrameResult
from vot.tracker.helpers import encode_region, decode_region, convert_region
from vot.utilities import to_number

logger = logging.getLogger("vot")


def _resolve_factory(command: str) -> Any:
    if ":" in command:
        module_name, object_name = command.split(":", 1)
    elif "." in command:
        module_name, object_name = command.rsplit(".", 1)
    else:
        module_name, object_name = command, None

    module = importlib.import_module(module_name)

    if object_name is None:
        if hasattr(module, "create_tracker"):
            return getattr(module, "create_tracker")
        if hasattr(module, "Tracker"):
            return getattr(module, "Tracker")
        raise RuntimeError("Unable to resolve tracker factory from command '{}'".format(command))

    if not hasattr(module, object_name):
        raise RuntimeError("Object '{}' not found in module '{}'".format(object_name, module_name))

    return getattr(module, object_name)


def _call_tracker_method(method: Callable[..., Any], frame: Any, new: Any,
                         properties: dict[str, Any] | None) -> Any:
    signature = inspect.signature(method)
    arity = len(signature.parameters)

    if arity >= 3:
        return method(frame, new, properties)
    if arity == 2:
        return method(frame, new)
    if arity == 1:
        return method(frame)
    return method()


def _worker_main(command: str, task_queue: Any, result_queue: Any,
                 arguments: dict[str, Any] | None, initialize_method: str, update_method: str,
                 paths: list[str] | None = None, envvars: dict[str, str] | None = None) -> None:
    try:
        # Self-cleanup: if the parent dies (Ctrl+C, killed terminal, SIGKILL), exit
        # instead of running the tracker indefinitely on an orphaned pipe.
        from vot.utilities import arm_parent_watchdog
        arm_parent_watchdog()
        # The worker runs in a spawned process, which does not inherit ``sys.path``
        # edits or environment changes from the parent. Apply the tracker source
        # paths and configured environment variables before importing the tracker.
        for path in paths or []:
            if path and path not in sys.path:
                sys.path.insert(0, path)
        if envvars:
            os.environ.update({str(k): str(v) for k, v in envvars.items()})
        factory = _resolve_factory(command)
        tracker = factory(**(arguments or {})) if callable(factory) else factory

        if hasattr(tracker, initialize_method):
            init_name = initialize_method
        elif hasattr(tracker, "init"):
            init_name = "init"
        elif hasattr(tracker, "initialize"):
            init_name = "initialize"
        else:
            raise RuntimeError("Tracker object does not expose an initialization method")

        if not hasattr(tracker, update_method):
            raise RuntimeError(f"Tracker object does not expose '{update_method}' method")

        init_fn = getattr(tracker, init_name)
        update_fn = getattr(tracker, update_method)

        multiobject = bool(getattr(tracker, "multiobject", False))

        result_queue.put({
            "ok": True,
            "event": "ready",
            "multiobject": multiobject,
            "initialize_method": init_name,
            "update_method": update_method
        })
        
        while True:
            task = task_queue.get()
            task_type = task.get("type")

            if task_type == "stop":
                result_queue.put({"ok": True, "event": "stopped"})
                break

            frame = task.get("frame")
            new = task.get("new")
            properties = task.get("properties")

            start = time.time()

            if task_type == "initialize":
                output = _call_tracker_method(init_fn, frame, new, properties)
            elif task_type == "update":
                output = _call_tracker_method(update_fn, frame, new, properties)
            else:
                raise RuntimeError(f"Unknown task type '{task_type}'")

            elapsed = time.time() - start

            if multiobject:
                assert isinstance(output, (list, tuple)), "Expected tracker output to be a list or tuple of object statuses for multi-object tracker"
                status = []
                for item in output:
                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (dict)):
                        status.append(item)
                    else:
                        status.append((item, {}))
            else:
                if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
                    status = output
                else:
                    status = (output, {})
            result_queue.put({"ok": True, "event": task_type, "status": status, "time": elapsed})

    except Exception as e:
        result_queue.put({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


class PythonRuntime(OnlineTrackerRuntime):
    """Multiprocessing runtime for Python-native trackers.

    The command is interpreted as an import path to a tracker factory/class,
    for example: ``mypackage.mytracker:Tracker``.
    """

    def __init__(self, tracker: Tracker, command: str, log: bool = False, timeout: int = 30,
                 linkpaths: list[str] | str | None = None, envvars: dict[str, str] | None = None,
                 arguments: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(tracker)

        self._command = command
        self._timeout = to_number(timeout)
        self._log = log
        self._linkpaths = linkpaths
        self._envvars = envvars
        self._arguments = arguments if arguments is not None else {}
        self._initialize_method = kwargs.get("initialize_method", "init")
        self._update_method = kwargs.get("update_method", "update")
        self._convert = kwargs.get("convert", None)
        # ``paths`` from the tracker config (registry key ``paths``). The worker is a
        # spawned process and does not inherit the parent's ``sys.path``, so these
        # tracker source directories must be forwarded to it explicitly.
        self._paths = kwargs.get("paths", linkpaths)

        # Surface configuration keys this runtime does not understand instead of
        # silently dropping them (a misspelled option would otherwise just vanish).
        _recognized = {"initialize_method", "update_method", "convert", "paths", "multiobject"}
        unknown = sorted(set(kwargs) - _recognized)
        if unknown:
            logger.warning(
                "PythonRuntime for tracker '%s' ignoring unrecognized option(s): %s",
                tracker.identifier, ", ".join(unknown)
            )

        # ``multiprocessing.Queue``/``multiprocessing.Process`` annotations in
        # the stdlib stubs are version-dependent; the simplest portable choice is
        # ``Any | None`` here — the runtime types are still ``mp.Queue`` etc.
        self._task_queue: Any | None = None
        self._result_queue: Any | None = None
        self._process: Any | None = None
        self._multiobject = kwargs.get("multiobject", True)

    def _timeout_value(self) -> float | None:
        return None if self._timeout is None or self._timeout <= 0 else self._timeout

    def _wait_message(self) -> dict:
        assert self._result_queue is not None, "Worker queues not initialized; call _ensure_started() first"
        try:
            return self._result_queue.get(timeout=self._timeout_value())
        except queue.Empty as e:
            raise TrackerException(
                f"Python tracker runtime timed out after {self._timeout} seconds",
                tracker=self.tracker
            ) from e

    def _raise_worker_error(self, message: dict) -> None:
        details = message.get("traceback")
        error = message.get("error", "Unknown worker error")
        raise TrackerException(
            f"Python tracker worker failed: {error}",
            tracker=self.tracker,
            tracker_log=details
        )

    def _resolved_paths(self) -> list[str]:
        """Tracker source paths as absolute paths for the worker's ``sys.path``."""
        if not self._paths:
            return []
        raw = self._paths.split(os.pathsep) if isinstance(self._paths, str) else list(self._paths)
        return [os.path.abspath(p) for p in raw if p]

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        context = multiprocessing.get_context("spawn")
        self._task_queue = context.Queue()
        self._result_queue = context.Queue()
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._command,
                self._task_queue,
                self._result_queue,
                self._arguments,
                self._initialize_method,
                self._update_method,
                self._resolved_paths(),
                self._envvars,
            )
        )
        self._process.start()

        message = self._wait_message()
        if not message.get("ok", False):
            self.stop()
            self._raise_worker_error(message)

        if message.get("event") != "ready":
            self.stop()
            raise TrackerException("Python tracker worker did not enter ready state", tracker=self.tracker)

        self._multiobject = bool(message.get("multiobject", True))

    def _send_task(self, task_type: str, frame: Frame, new: FrameObjects | None = None, properties: dict | None = None) -> tuple[FrameObjects, float]:
        if new is None:
            converted_new = []
        else:
            try:
                if not isinstance(new, (list)):
                    converted_new = (encode_region(convert_region(new.region, self._convert)), new.properties)
                else:
                    converted_new = [(encode_region(convert_region(status.region, self._convert)), status.properties) for status in new]
            except ValueError as e:
                raise TrackerException(str(e), tracker=self.tracker) from e
        if len(frame.channels()) > 1:
            frame_payload = {channel: frame.filename(channel) for channel in frame.channels()}
        else:
            frame_payload = frame.filename()

        payload = {
            "type": task_type,
            "frame": frame_payload,
            "new": converted_new,
            "properties": {} if properties is None else properties
        }

        assert self._task_queue is not None, "Worker queues not initialized; _ensure_started() must run first"
        self._task_queue.put(payload)
        message = self._wait_message()

        if not message.get("ok", False):
            self.stop()
            self._raise_worker_error(message)

        if message.get("event") != task_type:
            self.stop()
            raise TrackerException(
                f"Unexpected worker event '{message.get('event')}' while waiting for '{task_type}'",
                tracker=self.tracker
            )
        if not self._multiobject:
            return FrameResult(ObjectStatus(decode_region(message.get("status", (None, {}))[0]), message.get("status", (None, {}))[1]), float(message.get("time", 0.0)))

        return FrameResult([ObjectStatus(decode_region(status[0]), status[1]) for status in message.get("status", [])], float(message.get("time", 0.0)))

    @staticmethod
    def _mirror_output_shape(input_is_list: bool, status: FrameObjects) -> FrameObjects:
        """Match the result shape to the input: a query-based caller passes a list and
        indexes the result per query, while a legacy per-frame caller passes a single
        ObjectStatus and expects one back. ``_send_task`` returns a bare ObjectStatus for
        single-object trackers, so wrap/unwrap to mirror the input."""
        if input_is_list and not isinstance(status, list):
            return [status]
        if not input_is_list and isinstance(status, list):
            return status[0]
        return status

    def initialize(self, frame: Frame, new: FrameObjects | None = None, properties: dict | None = None) -> tuple[FrameObjects, float]:
        self._ensure_started()
        # ``new`` is either a single ObjectStatus (legacy per-frame caller) or a
        # list[ObjectStatus] (query-based caller). Return shape mirrors input shape so
        # both call styles work without a dedicated wrapper.
        input_is_list = isinstance(new, list)
        if not self.multiobject:
            if new is None:
                raise TrackerException(
                    "Initialization frame must be provided for single-object tracker",
                    tracker=self.tracker
                )
            if input_is_list:
                if len(new) == 0:
                    raise TrackerException(
                        "Initialization frame must contain exactly one object for single-object tracker, but got an empty list",
                        tracker=self.tracker
                    )
                if len(new) > 1:
                    raise TrackerException(
                        "Initialization frame must contain exactly one object for single-object tracker, but got multiple objects",
                        tracker=self.tracker
                    )
                new = new[0]

        status, elapsed = self._send_task("initialize", frame, new, properties)
        return self._mirror_output_shape(input_is_list, status), elapsed

    def update(self, frame: Frame, new: FrameObjects | None = None, properties: dict | None = None) -> tuple[FrameObjects, float]:
        self._ensure_started()
        input_is_list = isinstance(new, list)
        if not self.multiobject and input_is_list:
            if len(new) > 1:
                raise TrackerException(
                    "Tracker does not support multiple objects, but multiple objects were provided for update",
                    tracker=self.tracker,
                )
            new = new[0] if len(new) == 1 else None

        status, elapsed = self._send_task("update", frame, new, properties)
        return self._mirror_output_shape(input_is_list, status), elapsed

    def restart(self) -> None:
        self.stop()
        self._ensure_started()

    def stop(self) -> None:
        if self._process is None:
            return

        try:
            if self._process.is_alive() and self._task_queue is not None:
                self._task_queue.put({"type": "stop"})
                self._process.join(timeout=2)
        except (OSError, ValueError, EOFError, BrokenPipeError):
            pass

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)

        try:
            if self._task_queue is not None:
                self._task_queue.close()
            if self._result_queue is not None:
                self._result_queue.close()
        except (OSError, ValueError):
            pass

        self._task_queue = None
        self._result_queue = None
        self._process = None

    @property
    def multiobject(self) -> bool:
        self._ensure_started()
        return self._multiobject

from vot.tracker import register_runtime_protocol

register_runtime_protocol("python", PythonRuntime)
