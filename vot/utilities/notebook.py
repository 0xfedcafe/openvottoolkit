"""Utilities for running and visualizing VOT workflows in Jupyter notebooks."""

import io
import typing
from collections.abc import Iterable
from datetime import datetime
from threading import Condition, Thread

if typing.TYPE_CHECKING:
    import numpy.typing as npt
    from vot.experiment import Experiment
    from vot.tracker import Tracker
    from vot.workspace import Workspace
    from vot.dataset import Sequence, Frame
    from vot.region import Region


def is_notebook() -> bool:
    """Return True when executed inside an IPython kernel notebook."""
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        shell = get_ipython()
        if shell is None:
            return False

        return "IPKernelApp" in shell.config
    except (ImportError, AttributeError):
        return False


def _require_notebook_dependencies() -> tuple[typing.Any, typing.Any, typing.Any]:
    """Load notebook-only dependencies or raise a clear import error."""
    if not is_notebook():
        raise ImportError("The Jupyter notebook environment is required for visualization.")

    try:
        from IPython.display import display  # type: ignore[import-not-found]
        from ipywidgets import widgets  # type: ignore[import-not-found]
        from vot.utilities.draw import ImageDrawHandle
    except ImportError as exc:
        raise ImportError("The IPython and ipywidgets packages are required for visualization.") from exc

    return display, widgets, ImageDrawHandle


def _encode_image(handle) -> bytes:
    with io.BytesIO() as output:
        handle.snapshot.save(output, format="PNG")
        return output.getvalue()


def _frame_image(frame: "Frame") -> "npt.NDArray":
    """Return the image array for a frame, raising if it can't be loaded."""
    image = frame.image()
    if image is None:
        raise RuntimeError("Unable to load image for frame")
    return image


_T = typing.TypeVar("_T")


def _as_list(value: None | _T | typing.Iterable[_T]) -> list[_T]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]  # type: ignore[list-item]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _extract_region(objects: typing.Any) -> "Region | None":
    """Extract first region from runtime frame output."""
    if objects is None:
        return None

    if hasattr(objects, "objects"):
        objects = objects.objects

    if hasattr(objects, "region"):
        return objects.region

    if isinstance(objects, (list, tuple)) and objects:
        first = objects[0]
        if hasattr(first, "region"):
            return first.region

    return None


class SequenceView(object):
    """A compact widget for showing sequence frames and regions."""

    def __init__(self, sequence: "Sequence") -> None:
        display, widgets, ImageDrawHandle = _require_notebook_dependencies()

        self._display = display
        self._sequence = sequence
        self._handle = ImageDrawHandle(_frame_image(sequence.frame(0)))

        self.frame = widgets.Label(value="Frame: 0")
        self.image = widgets.Image(
            value=_encode_image(self._handle),
            format="png",
            width=sequence.size[0] * 2,
            height=sequence.size[1] * 2,
        )
        self.widget = widgets.VBox(children=(self.image, self.frame))

    def set_frame(self, index: int, region: "Region | None" = None) -> None:
        index = max(0, min(index, len(self._sequence) - 1))
        frame = self._sequence.frame(index)

        self._handle.image(_frame_image(frame))
        self._handle.style(color="green").region(frame.groundtruth())
        if region is not None:
            self._handle.style(color="red").region(region)

        self.image.value = _encode_image(self._handle)
        self.frame.value = "Frame: {}".format(index)

    def show(self) -> None:
        self._display(self.widget)


def visualize_tracker(tracker: "Tracker", sequence: "Sequence") -> None:
    """Visualize tracker outputs on a sequence inside a notebook."""
    display, widgets, ImageDrawHandle = _require_notebook_dependencies()

    handle = ImageDrawHandle(_frame_image(sequence.frame(0)))
    frame_label = widgets.Label(value="")
    frame_label.layout.display = "none"
    mirror_label = widgets.Label(value="")
    image = widgets.Image(
        value=_encode_image(handle),
        format="png",
        width=sequence.size[0] * 2,
        height=sequence.size[1] * 2,
    )

    button_restart = widgets.Button(description="Restart")
    button_next = widgets.Button(description="Next")
    button_play = widgets.Button(description="Run")

    state = {"frame": 0, "auto": False, "alive": True, "region": None, "restart": False}
    condition = Condition()

    def update_image() -> None:
        index = max(0, min(state["frame"], len(sequence) - 1))
        frame = sequence.frame(index)

        handle.image(_frame_image(frame))
        handle.style(color="green").region(frame.groundtruth())
        if state["region"] is not None:
            handle.style(color="red").region(state["region"])

        image.value = _encode_image(handle)
        frame_label.value = "Frame: {}".format(index)

    def run() -> None:
        from vot.tracker import OnlineTrackerRuntime, ObjectStatus
        runtime = tracker.runtime()
        assert isinstance(runtime, OnlineTrackerRuntime), \
            "Visualization requires an online tracker runtime"

        try:
            while state["alive"]:
                if state["restart"]:
                    runtime.stop()
                    runtime = tracker.runtime()
                    assert isinstance(runtime, OnlineTrackerRuntime)
                    state["frame"] = 0
                    state["region"] = None
                    state["restart"] = False

                index = state["frame"]
                if index >= len(sequence):
                    state["alive"] = False
                    break

                if index == 0:
                    initial = sequence.groundtruth(0)
                    initial_objects = ObjectStatus(initial, {}) if initial is not None else None
                    objects, _ = runtime.initialize(sequence.frame(0), initial_objects)
                else:
                    objects, _ = runtime.update(sequence.frame(index))

                state["region"] = _extract_region(objects)
                update_image()

                with condition:
                    if state["frame"] >= len(sequence) - 1:
                        state["auto"] = False

                    if state["auto"]:
                        condition.wait(timeout=0.05)
                    else:
                        condition.wait()

                    if not state["alive"]:
                        break

                    if state["frame"] < len(sequence) - 1:
                        state["frame"] += 1
        finally:
            runtime.stop()

    def on_click(button: typing.Any) -> None:
        with condition:
            if button is button_next:
                state["auto"] = False
                condition.notify()
            elif button is button_restart:
                state["auto"] = False
                state["restart"] = True
                button_play.description = "Run"
                condition.notify()
            elif button is button_play:
                state["auto"] = not state["auto"]
                button_play.description = "Stop" if state["auto"] else "Run"
                condition.notify()

    button_next.on_click(on_click)
    button_restart.on_click(on_click)
    button_play.on_click(on_click)
    widgets.jslink((frame_label, "value"), (mirror_label, "value"))

    controls = widgets.VBox(children=(frame_label, button_restart, button_next, button_play, mirror_label))
    thread = Thread(target=run, daemon=True)

    display(widgets.Box([widgets.HBox(children=(image, controls))]))
    thread.start()

    with condition:
        condition.notify()


def visualize_results(experiment: "Experiment", sequence: "Sequence", trackers: "Tracker | list[Tracker]") -> None:
    """Visualize already computed experiment results for one sequence."""
    display, widgets, ImageDrawHandle = _require_notebook_dependencies()

    trackers = _as_list(trackers)
    if not trackers:
        raise ValueError("At least one tracker must be provided.")

    transformed = experiment.transform([sequence])
    if transformed:
        sequence = transformed[0]

    tracker_trajectories: dict["Tracker", typing.Any] = {}
    gather: typing.Callable[..., typing.Sequence[typing.Any]] | None = getattr(experiment, "gather", None)
    if callable(gather):
        for tracker in trackers:
            trajectories = gather(tracker, sequence)
            tracker_trajectories[tracker] = trajectories[0] if trajectories else None

    handle = ImageDrawHandle(_frame_image(sequence.frame(0)))
    image = widgets.Image(
        value=_encode_image(handle),
        format="png",
        width=sequence.size[0] * 2,
        height=sequence.size[1] * 2,
    )
    frame_label = widgets.Label(value="Frame: 0")

    button_restart = widgets.Button(description="Restart")
    button_next = widgets.Button(description="Next")
    button_play = widgets.Button(description="Run")

    state = {"frame": 0, "auto": False, "alive": True}
    condition = Condition()

    colors = ["red", "blue", "orange", "purple", "yellow"]

    def update_image() -> None:
        index = max(0, min(state["frame"], len(sequence) - 1))
        frame = sequence.frame(index)

        handle.image(_frame_image(frame))
        handle.style(color="green").region(frame.groundtruth())

        for i, tracker in enumerate(trackers):
            trajectory = tracker_trajectories.get(tracker)
            if trajectory is None or index >= len(trajectory):
                continue
            handle.style(color=colors[i % len(colors)]).region(trajectory.region(index))

        image.value = _encode_image(handle)
        frame_label.value = "Frame: {}".format(index)

    def run() -> None:
        while state["alive"]:
            update_image()

            with condition:
                if state["frame"] >= len(sequence) - 1:
                    state["auto"] = False

                if state["auto"]:
                    condition.wait(timeout=0.05)
                else:
                    condition.wait()

                if not state["alive"]:
                    break

                if state["frame"] < len(sequence) - 1:
                    state["frame"] += 1

    def on_click(button: typing.Any) -> None:
        with condition:
            if button is button_next:
                state["auto"] = False
                condition.notify()
            elif button is button_restart:
                state["auto"] = False
                state["frame"] = 0
                button_play.description = "Run"
                condition.notify()
            elif button is button_play:
                state["auto"] = not state["auto"]
                button_play.description = "Stop" if state["auto"] else "Run"
                condition.notify()

    button_next.on_click(on_click)
    button_restart.on_click(on_click)
    button_play.on_click(on_click)

    controls = widgets.HBox(children=(frame_label, button_restart, button_next, button_play))
    thread = Thread(target=run, daemon=True)

    display(widgets.Box([widgets.VBox(children=(image, controls))]))
    thread.start()

    with condition:
        condition.notify()


def run_experiment(
    experiment: "Experiment",
    sequences: list["Sequence"],
    trackers: list["Tracker"],
    force: bool = False,
    persist: bool = False,
) -> bool:
    """Run an experiment for one or more trackers from a notebook."""
    display, widgets, _ImageDrawHandle = _require_notebook_dependencies()

    sequences = _as_list(sequences)
    trackers = _as_list(trackers)

    from vot.tracker import TrackerException

    # Pre-transform sequences so progress total is accurate.
    transformed = []
    for sequence in sequences:
        transformed.extend(experiment.transform(sequence))
    sequences = transformed

    n_sequences = max(len(sequences), 1)
    total = len(trackers) * n_sequences

    progress_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=total,
        description="",
        bar_style="info",
        layout=widgets.Layout(width="100%"),
    )
    label = widgets.Label(value="Starting...")
    display(widgets.VBox([label, progress_bar]))

    try:
        for i, tracker in enumerate(trackers):
            for j, sequence in enumerate(sequences):
                base = i * n_sequences + j
                label.value = "{} — {} ({}/{})".format(
                    tracker.identifier, sequence.name, j + 1, n_sequences
                )

                def _callback(p: float, _base: int = base) -> None:
                    progress_bar.value = _base + min(1.0, max(0.0, p))

                try:
                    experiment.execute(tracker, sequence, force=force, callback=_callback)
                except TrackerException as te:
                    if not persist:
                        label.value = "Error on {}: {}".format(sequence.name, te)
                        progress_bar.bar_style = "danger"
                        raise

                progress_bar.value = base + 1

    except InterruptedError:
        label.value = "Interrupted."
        progress_bar.bar_style = "warning"
        return False

    label.value = "Done."
    progress_bar.bar_style = "success"
    progress_bar.value = total
    return True


def run_analysis(
    workspace: "Workspace",
    trackers: list["Tracker"] | list[str],
    sequences: list[str] | None = None,
    experiments: list[str] | None = None,
    output_format: str | None = None,
    name: str | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
    """Run stack analyses from a notebook and optionally serialize outputs."""
    if not is_notebook():
        raise ImportError("The Jupyter notebook environment is required for visualization.")

    if "format" in kwargs and output_format is None:
        output_format = kwargs.pop("format")
    if kwargs:
        raise TypeError("Unexpected keyword arguments: {}".format(", ".join(kwargs.keys())))

    tracker_list = _as_list(trackers)
    sequence_names = _as_list(sequences) if sequences is not None else None
    experiment_names = _as_list(experiments) if experiments is not None else None

    if tracker_list and isinstance(tracker_list[0], str):
        str_trackers: list[str] = [t for t in tracker_list if isinstance(t, str)]
        assert len(str_trackers) == len(tracker_list), "Tracker list must be all strings or all Tracker instances"
        resolved_trackers: list["Tracker"] = workspace.registry.resolve(
            *str_trackers,
            storage=workspace.storage.substorage("results"),
            skip_unknown=False,
        )
    else:
        from vot.tracker import Tracker as _Tracker
        resolved_trackers = [t for t in tracker_list if isinstance(t, _Tracker)]
        assert len(resolved_trackers) == len(tracker_list), "Tracker list must be all Tracker instances or all strings"

    from vot.analysis import process_stack_analyses
    from vot.report import generate_serialized

    try:
        results = process_stack_analyses(workspace, resolved_trackers, sequence_names, experiment_names)

        if results is None:
            return None

        if output_format is not None:
            if output_format not in ("json", "yaml"):
                raise ValueError("Unknown format '{}'".format(output_format))

            if name is None:
                name = "{:%Y-%m-%dT%H-%M-%S.%f%z}".format(datetime.now())

            selected = list(workspace.dataset) if sequence_names is None else [s for s in workspace.dataset if s.name in sequence_names]
            storage = workspace.storage.substorage("analysis")
            generate_serialized(resolved_trackers, selected, results, storage, output_format, name)

            return {"results": results, "name": name, "format": output_format}

        return results

    except InterruptedError:
        return False
