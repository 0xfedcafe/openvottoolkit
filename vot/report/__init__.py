"""This module contains classes for generating reports and visualizations."""

import typing
import json
import inspect
import threading
import datetime
import collections
import collections.abc
import types
from asyncio import wait, ensure_future
from asyncio.futures import wrap_future

import numpy as np
import numpy.typing as npt
import yaml

from matplotlib.figure import Figure
from matplotlib.axes import Axes as PlotAxes
import matplotlib.colors as colors

from attributee import Attributee, Object, Nested, String, Callable, Integer, List

from vot import __version__ as version
from vot import get_logger
from vot.dataset import Dataset, Sequence, FrameList
from vot.tracker import Tracker
from vot.utilities.draw import Color
from vot.utilities import class_fullname
from vot.utilities.data import Grid
from vot.utilities import Registry, ObjectResolver

if typing.TYPE_CHECKING:
    from vot.experiment import Experiment
    from vot.stack import Stack
    from vot.workspace import Workspace
    from vot.workspace.storage import Storage
    from vot.analysis import Analysis

Table = collections.namedtuple("Table", ["header", "data", "order"])

AxisLimit = float | None
AxisLimits = tuple[AxisLimit, AxisLimit] | None
PlotOutput = str | typing.IO[typing.Any]
VideoOutput = str | typing.IO[bytes]
AxesRect = tuple[float, float, float, float]


class Plot(object):
    """Base class for all plots."""

    def __init__(self, identifier: str, xlabel: str | None, ylabel: str | None,
        xlimits: AxisLimits, ylimits: AxisLimits, trait: str | None = None) -> None:
        """Initializes the plot.

        :param identifier: The identifier of the plot.
        :param xlabel: Optional label of the x axis (``None`` leaves the axis unlabeled).
        :param ylabel: Optional label of the y axis (``None`` leaves the axis unlabeled).
        :param xlimits: The limits of the x axis.
        :param ylimits: The limits of the y axis.
        :param trait: The trait of the plot.
        """

        self._identifier = identifier
        self._xlimits = xlimits
        self._ylimits = ylimits

        self._manager = StyleManager.default()

        self._figure, self._axes = self._manager.make_figure(trait)

        if xlabel is not None:
            self._axes.xaxis.set_label_text(xlabel)
        if ylabel is not None:
            self._axes.yaxis.set_label_text(ylabel)

        self._apply_axis_limits()

    @staticmethod
    def _needs_autoscale(limits: AxisLimits) -> bool:
        """Returns true if at least one side of an axis should be auto-scaled."""
        return limits is None or limits[0] is None or limits[1] is None

    def _apply_axis_limit(self, axis: str, limits: AxisLimits) -> None:
        """Apply complete or partial axis limits to a Matplotlib axis."""
        if limits is None:
            return

        lower, upper = limits
        if lower is None and upper is None:
            return

        if axis == "x":
            current_lower, current_upper = self._axes.get_xlim()
            self._axes.set_xlim(
                left=current_lower if lower is None else lower,
                right=current_upper if upper is None else upper,
            )
            if lower is not None and upper is not None:
                self._axes.autoscale(False, axis="x")
        else:
            current_lower, current_upper = self._axes.get_ylim()
            self._axes.set_ylim(
                bottom=current_lower if lower is None else lower,
                top=current_upper if upper is None else upper,
            )
            if lower is not None and upper is not None:
                self._axes.autoscale(False, axis="y")

    def _apply_axis_limits(self) -> None:
        """Apply configured limits while preserving auto-scaling for open bounds."""
        autoscale_x = self._needs_autoscale(self._xlimits)
        autoscale_y = self._needs_autoscale(self._ylimits)

        if autoscale_x:
            self._axes.autoscale(enable=True, axis="x")
        if autoscale_y:
            self._axes.autoscale(enable=True, axis="y")
        if autoscale_x or autoscale_y:
            self._axes.autoscale_view(scalex=autoscale_x, scaley=autoscale_y)

        self._apply_axis_limit("x", self._xlimits)
        self._apply_axis_limit("y", self._ylimits)

    def __call__(self, key: typing.Any, data: typing.Any) -> None:
        """Draws the data on the plot."""
        self.draw(key, data)
        self._apply_axis_limits()

    def draw(self, key: typing.Any, data: typing.Any) -> None:
        """Draws the data on the plot."""
        raise NotImplementedError
    
    @property
    def axes(self) -> PlotAxes:
        """Returns the axes of the plot."""
        return self._axes

    def save(self, output: PlotOutput, fmt: str) -> None:
        """Saves the plot to a file.

        :param output: The output file.
        :type output: str
        :param fmt: The format of the output file.
        :type fmt: str
        """
        self._figure.savefig(output, format=fmt, bbox_inches='tight', transparent=True)

    @property
    def identifier(self) -> str:
        """Returns the identifier of the plot."""
        return self._identifier

class Video(object):
    """Base class for all videos."""

    def __init__(self, identifier: str, frames: FrameList, fps: int = 30, trait: str | None = None) -> None:
        """Initializes the video object.

        :param identifier: The identifier of the video.
        :type identifier: str
        :param frames: The frames of the video.
        :type frames: FrameList
        :param fps: The frames per second of the video.
        :type fps: int
        :param trait: The trait of the video.
        :type trait: str
        """

        self._identifier = identifier
        self._frames = frames
        self._fps = fps
        self._manager = StyleManager.default()

    def __call__(self, frame: int, key: typing.Any, data: typing.Any) -> None:
        """Draws the data on the frame."""
        self.draw(frame, key, data)

    def draw(self, frame: int, key: typing.Any, data: typing.Any) -> None:
        """Draws the data on the plot."""
        raise NotImplementedError

    def render(self, frame: int) -> npt.NDArray[np.uint8]:
        """Renders the frame and returns it as a NumPy array."""
        raise NotImplementedError

    def save(self, output: VideoOutput, fmt: str) -> None:
        import tempfile
        import shutil
        import os
        import importlib
        from .video import VideoWriterScikitH264, VideoWriterOpenCV
        from . import video as _video

        # Ordered preference: try the higher-quality H.264 backend first, fall
        # back to OpenCV (mp4v) when scikit-video / ffmpeg is missing **or**
        # when scikit-video raises at runtime (e.g. unmaintained 1.1.x breaks
        # against numpy >= 2.0 where ``ndarray.tostring`` was removed).
        supported_mappings = {
            "mp4": [
                (VideoWriterScikitH264, "skvideo.io"),
                (VideoWriterOpenCV, "cv2"),
            ],
            "avi": [
                (VideoWriterOpenCV, "cv2"),
            ],
        }

        if not fmt in supported_mappings:
            raise ValueError("Unsupported video format: {}".format(fmt))

        candidates: list[type] = []
        missing: list[str] = []
        for candidate, module in supported_mappings[fmt]:
            # A backend that already failed at runtime in this process is not retried;
            # re-probing it (e.g. broken scikit-video) wastes an ffmpeg spawn per video.
            if candidate in _video._RUNTIME_BROKEN_WRITERS:
                continue
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(module)
                continue
            candidates.append(candidate)

        if not candidates:
            raise ImportError(
                "No usable {} writer available. Install one of: {}.".format(
                    fmt, ", ".join(missing) or "a working video backend")
            )

        if isinstance(output, str):
            tempname = output
            output_handle = None
        else:
            fd, tempname = tempfile.mkstemp(prefix="video_", suffix=".{}".format(fmt))
            os.close(fd)
            output_handle = output

        last_error: BaseException | None = None
        try:
            failed_before_success: list[type] = []
            for attempt, writer_cls in enumerate(candidates):
                # Each attempt writes into a clean file. Remove a partial file
                # left behind by an earlier attempt before trying the next backend.
                if os.path.exists(tempname):
                    os.remove(tempname)
                writer = writer_cls(tempname, self._fps)
                try:
                    try:
                        for i in range(0, len(self._frames)):
                            writer(self.render(i))
                    finally:
                        writer.close()
                except Exception as exc:
                    last_error = exc
                    failed_before_success.append(writer_cls)
                    get_logger().warning(
                        "Video writer %s failed (%s); trying next backend",
                        writer_cls.__name__, exc,
                    )
                    continue
                last_error = None
                # A backend that failed only to be replaced by a working fallback is
                # reliably broken in this environment; skip it for the remaining videos.
                _video._RUNTIME_BROKEN_WRITERS.update(failed_before_success)
                break

            if last_error is not None:
                raise last_error

            if output_handle is None:
                return

            with open(tempname, 'rb') as source:
                shutil.copyfileobj(source, output_handle)
        finally:
            if output_handle is not None and os.path.exists(tempname):
                os.remove(tempname)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def identifier(self) -> str:
        """Returns the identifier of the plot."""
        return self._identifier

class ScatterPlot(Plot):
    """A scatter plot."""

    def draw(self, key, data) -> None:
        """Draws the data on the plot."""
        if data is None or len(data) != 2:
            return

        style = self._manager.plot_style(key)
        self._axes.scatter(data[0], data[1], **style.point_style())

class LinePlot(Plot):
    """A line plot."""

    def draw(self, key, data) -> None:
        """Draws the data on the plot."""
        if data is None or len(data) < 1:
            return

        if isinstance(data[0], tuple):
            # Drawing curve
            if len(data[0]) != 2:
                return
            x, y = zip(*data)
        else:
            y = data
            x = range(len(data))

        style = self._manager.plot_style(key)

        self._axes.plot(x, y, **style.line_style())

class ObjectVideo(Video):

    def __init__(self, identifier: str, frames: FrameList, fps: int = 10, trait: str | None = None) -> None:
        super().__init__(identifier, frames, fps=fps, trait=trait)
        self._regions = {}

    def draw(self, frame: int, key, data) -> None:
        """Draws the data on the frame."""
        from vot.region import Region
        assert isinstance(data, Region)

        if not key in self._regions:
            self._regions[key] = [None] * len(self)

        self._regions[key][frame] = data

    def _load_image(self, frame: int) -> npt.NDArray[np.uint8]:
        """Loads the source image for the given frame index.

        Separated from :meth:`render` so subclasses can cache decoded frames."""
        image = self._frames.frame(frame).image()
        if image is None:
            raise ValueError(f"Frame {frame} has no image data")
        return image

    def render(self, frame: int) -> npt.NDArray[np.uint8]:
        """Renders the frame and returns it as an array."""
        from vot.utilities.draw import ImageDrawHandle

        assert frame >= 0 and frame < len(self)

        handle = ImageDrawHandle(self._load_image(frame))

        for key, regions in self._regions.items():
            if regions[frame] is None:
                continue

            style = self._manager.plot_style(key)

            handle.style(**style.region_style())
            regions[frame].draw(handle)

        return handle.array



def generate_serialized(trackers: list[Tracker], sequences: list[Sequence], results, storage: "Storage", serializer: str, name: str) -> None:
    """Generates a serialized report of the results."""

    from vot.utilities.io import JSONEncoder, YAMLEncoder

    doc: dict[str, typing.Any] = dict()
    doc["toolkit"] = version
    doc["timestamp"] = datetime.datetime.now().isoformat()
    doc["trackers"] = {t.reference : t.describe() for t in trackers}
    doc["sequences"] = {s.name : s.describe() for s in sequences}

    doc["results"] = dict()

    for experiment, analyses in results.items():
        exp: dict[str, typing.Any] = dict(parameters=experiment.dump(), type=class_fullname(experiment))
        exp["results"] = []
        for _, data in analyses.items():
            exp["results"].append(data)
        doc["results"][experiment.identifier] = exp

    if serializer == "json":
        with storage.write(name + "." + serializer) as handle:
            json.dump(doc, handle, indent=2, cls=JSONEncoder)
    elif serializer == "yaml":
        with storage.write(name + "." + serializer) as handle:
            yaml.dump(doc, handle, Dumper=YAMLEncoder)
    else:
        raise RuntimeError("Unknown serializer")

def configure_axes(figure: Figure, rect: AxesRect | None = None, _=None) -> PlotAxes:
    """Configures the axes of the plot."""

    axes = PlotAxes(figure, (0.0, 0.0, 1.0, 1.0) if rect is None else rect)

    figure.add_axes(axes)

    return axes

def configure_figure(traits: str | None = None) -> Figure:
    """Configures the figure of the plot."""

    args: dict[str, typing.Any] = {}
    if traits == "ar":
        args["figsize"] = (5, 5)
    elif traits == "eao":
        args["figsize"] = (7, 5)
    elif traits == "attributes":
        args["figsize"] = (10, 5)

    return Figure(**args)

class PlotStyle(object):
    """A style for a plot."""

    def line_style(self, opacity: float = 1.0) -> dict:
        """Returns the style for a line."""
        raise NotImplementedError

    def point_style(self) -> dict:
        """Returns the style for a point."""
        raise NotImplementedError

    def region_style(self) -> dict:
        """Returns the style for a region, used with DrawHandle."""
        raise NotImplementedError


StyleFactory = typing.Callable[[int], PlotStyle]
AxesFactory = typing.Callable[[Figure, AxesRect | None, str | None], PlotAxes]
FigureFactory = typing.Callable[[str | None], Figure]


def _get_default_colormap() -> colors.Colormap:
    from matplotlib import colormaps
    return colormaps["Set1"].resampled(9)

class DefaultStyle(PlotStyle):
    """The default style for a plot."""

    colormap = _get_default_colormap()
    markers = ["o", "v", "<", ">", "^", "8", "*"]

    def __init__(self, number: int) -> None:
        """Initializes the style.

        :param number: The number of the style.
        :type number: int
        """
        super().__init__()
        self._number = number

    def _color(self) -> Color:
        """Returns this style's colormap entry as a :class:`Color`."""
        return Color.resolve(self.colormap(self._number % self.colormap.N))

    def line_style(self, opacity: float = 1.0) -> dict:
        """Returns the style for a line.

        :param opacity: The opacity of the line.
        :type opacity: float
        """
        color = self._color()
        if opacity < 1:
            color = color.with_alpha(opacity)
        return dict(linewidth=1, c=color.rgba())

    def point_style(self) -> dict:
        """Returns the style for a point."""
        marker = DefaultStyle.markers[self._number % len(DefaultStyle.markers)]
        return dict(marker=marker, c=[self._color().rgba()])

    def region_style(self) -> dict:
        """Returns the style for a region, used with DrawHandle."""
        return dict(color=self._color(), fill=True)

class Legend(object):
    """A legend for a plot."""

    def __init__(self, style_factory: StyleFactory = DefaultStyle) -> None:
        """Initializes the legend.

        :param style_factory: The style factory.
        :type style_factory: PlotStyleFactory
        """
        self._mapping = collections.OrderedDict()
        self._counter = 0
        self._style_factory = style_factory

    def _number(self, key) -> int:
        """Returns the number for a key."""
        if not key in self._mapping:
            self._mapping[key] = self._counter
            self._counter += 1
        return self._mapping[key]

    def __getitem__(self, key) -> PlotStyle:
        """Returns the style for a key."""
        number = self._number(key)
        return self._style_factory(number)

    def _style(self, number: int) -> PlotStyle:
        """Returns the style for a number."""
        raise NotImplementedError

    def keys(self) -> typing.KeysView:
        """Returns the keys of the legend."""
        return self._mapping.keys()

    def figure(self, key) -> Figure:
        """Returns a figure for a key."""
        style = self[key]
        figure = Figure(figsize=(0.1, 0.1))  # TODO: hardcoded
        axes = PlotAxes(figure, (0.0, 0.0, 1.0, 1.0), yticks=[], xticks=[], frame_on=False)
        figure.add_axes(axes)
        axes.patch.set_visible(False)
        marker_style = style.point_style()
        marker_style["s"] = 40 # Reset size
        axes.scatter(0, 0, **marker_style)
        return figure

class StyleManager(Attributee):
    """A manager for styles."""

    plots = Callable(default=DefaultStyle)
    axes = Callable(default=configure_axes)
    figure = Callable(default=configure_figure)

    _context = threading.local()

    def __init__(self, **kwargs):
        """Initializes a new instance of the StyleManager class."""
        super().__init__(**kwargs)
        self._legends = dict()

    def __getitem__(self, key) -> PlotStyle:
        """Gets the style for the given key."""
        return self.plot_style(key)

    def legend(self, key) -> Legend:
        """Gets the legend for a given key."""
        if inspect.isclass(key):
            klass = key
        else:
            klass = type(key)

        if not klass in self._legends:
            self._legends[klass] = Legend(typing.cast(StyleFactory, self.plots))

        return self._legends[klass]

    def plot_style(self, key) -> PlotStyle:
        """Gets the plot style for a given key."""
        return self.legend(key)[key]

    def make_axes(self, figure: Figure, rect: AxesRect | None = None, trait: str | None = None) -> PlotAxes:
        """Makes the axes for a given figure."""
        axes_factory = typing.cast(AxesFactory, self.axes)
        return axes_factory(figure, rect, trait)

    def make_figure(self, trait: str | None = None) -> tuple[Figure, PlotAxes]:
        """Makes the figure for a given trait.

        :param trait: The trait for which to make the figure.
        :type trait: str

        :returns: A tuple containing the figure and the axes."""
        figure_factory = typing.cast(FigureFactory, self.figure)
        figure = figure_factory(trait)
        axes = self.make_axes(figure, trait=trait)

        return figure, axes

    def __enter__(self) -> "StyleManager":
        """Enters the context of the style manager."""

        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            return self

        StyleManager._context.style_manager = self

        return self

    def __exit__(self, exc_type: type, exc_value: Exception, traceback: types.TracebackType) -> None:
        """Exits the context of the style manager."""
        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            StyleManager._context.style_manager = None

    @staticmethod
    def default() -> "StyleManager":
        """Gets the default style manager."""

        manager = getattr(StyleManager._context, 'style_manager', None)
        if manager is None:
            get_logger().info("Creating new style manager", stack_info=True)
            manager = StyleManager()
            StyleManager._context.style_manager = manager

        return manager

class TrackerSorter(Attributee):
    """A sorter for trackers."""

    experiment = String(default=None)
    analysis = String(default=None)
    result = Integer(val_min=0, default=0)

    def __call__(self, experiments: list["Experiment"], trackers: list["Tracker"], sequences: list["Sequence"]) -> list[int]:
        """Sorts the trackers.

        :param experiments: The experiments.
        :type experiments: list[Experiment]
        :param trackers: The trackers.
        :type trackers: list[Tracker]
        :param sequences: The sequences.
        :type sequences: list[Sequence]

        :returns: A list of indices of the trackers in the sorted order."""
        from vot.analysis import AnalysisError

        if self.experiment is None or self.analysis is None:
            return list(range(len(trackers)))

        experiment = next(filter(lambda x: x.identifier == self.experiment, experiments), None)

        if experiment is None:
            raise RuntimeError(f"Experiment not found {self.experiment}")

        analysis = next(filter(lambda x: x.name == self.analysis, experiment.analyses), None)

        if analysis is None:
            raise RuntimeError(f"Analysis not found {self.analysis} in experiment {self.experiment}")

        try:
            sequences = experiment.transform(sequences)
            future = analysis.commit(experiment, trackers, sequences)
            result = future.result()
        except AnalysisError as e:
            raise RuntimeError("Unable to sort trackers", e)

        scores = [x[self.result] for x in result]
        indices = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])]

        return indices

class Report(Attributee):
    """A report generator for various reports.

    Base class for all report generators.
    """

    async def generate(self, experiments: list["Experiment"], trackers: list["Tracker"], sequences: list["Sequence"]) -> dict[str, typing.Any]:
        raise NotImplementedError()

    async def process(self, analyses: list["Analysis"], experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> typing.Iterable[typing.Any]:

        sequences = experiment.transform(sequences)

        if not isinstance(analyses, collections.abc.Iterable):
            analyses = [analyses]

        futures = []

        for analysis in analyses:
            futures.append(wrap_future(analysis.commit(experiment, trackers, sequences)))

        if len(futures) == 0:
            return {}

        await wait(futures)

        return (future.result() for future in futures)

    async def _single_result(self, analysis: "Analysis", experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> typing.Any:
        """Run a single analysis via :meth:`process` and return its sole result, or
        ``None`` when there was nothing to compute (``process`` yielded no futures)."""
        analysis_result = await self.process([analysis], experiment, trackers, sequences)
        if isinstance(analysis_result, dict):
            return None
        return next(iter(analysis_result))

class SeparableReport(Report):
    """A report generator that is separable across experiments.

    Base class for all separable report generators.
    """

    async def perexperiment(self, experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> typing.Any:
        raise NotImplementedError()

    def compatible(self, experiment: "Experiment") -> bool:
        raise NotImplementedError()

    async def generate(self, experiments: list["Experiment"], trackers: list["Tracker"], sequences: list["Sequence"]) -> dict[str, typing.Any]:

        futures = []
        texperiments = []

        for experiment in experiments:

            tsequences = experiment.transform(sequences)

            if self.compatible(experiment):
                futures.append(ensure_future(self.perexperiment(experiment, trackers, tsequences)))
                texperiments.append(experiment)
            else:
                continue

        if len(futures) == 0:
            return {}

        await wait(futures)

        items = dict()

        for experiment, future in zip(texperiments, futures):
            items[experiment.identifier] = future.result()

        return items

report_registry = Registry("report")

class ReportConfiguration(Attributee):
    """A configuration for reports."""

    style = Nested(StyleManager)
    sort = Nested(TrackerSorter)
    index = List(Object(ObjectResolver(report_registry), subclass=Report), default=[], description="The reports to include.")

def generate_document(workspace: "Workspace", trackers: list[Tracker], format: str, name: str, select_sequences: list[str] | None = None, select_experiments: list[str] | None = None) -> None:
    """Generate a report for a one or multiple trackers on an experiment stack and a set
    of sequences.

    :param workspace: The workspace to use for the report.
    :type workspace: Workspace
    :param trackers: The trackers to include in the report.
    :param format: The format of the report.
    :param name: The name of the report.
    """
    import asyncio
    from asyncio import wait

    from vot.analysis import AnalysisProcessor
    from vot.utilities import Progress
    from vot.workspace.storage import Cache
    from vot import config
    from vot.report.common import StackAnalysesTable, StackAnalysesPlots, SequenceSpeedPlots

    def merge_tree(src, dest):

        for key, value in src.items():
            if not key in dest:        
                dest[key] = list()
            dest[key] += value

    logger = get_logger()

    logger.info("Worker pool size: %d", config.worker_pool_size)

    if config.worker_pool_size == 1:

        if config.debug_mode:
            import logging
            from vot.analysis.processor import DebugExecutor
            logging.getLogger("concurrent.futures").setLevel(logging.DEBUG)
            executor = DebugExecutor()
        else:
            from vot.utilities import ThreadPoolExecutor
            executor = ThreadPoolExecutor(1)

    else:
        from concurrent.futures import ProcessPoolExecutor
        from vot.utilities import arm_parent_watchdog
        executor = ProcessPoolExecutor(config.worker_pool_size, initializer=arm_parent_watchdog)

    if not config.persistent_cache:
        from cachetools import LRUCache
        cache = LRUCache(1000)
    else:
        cache = Cache(workspace.storage.substorage("cache").substorage("analysis"))

    index = workspace.report.index
    if len(index) == 0:
        # Default report content
        index = [StackAnalysesTable(), StackAnalysesPlots(), SequenceSpeedPlots()]
        
    with workspace.report.style:

        stack = typing.cast("Stack", workspace.stack)
        stack_experiments = typing.cast(typing.Mapping[str, "Experiment"], stack.experiments)
        experiments: list["Experiment"] = list(stack)
        sequences = list(typing.cast(Dataset, workspace.dataset))
        
        if not select_experiments is None:
            assert isinstance(select_experiments, list)
            experiments = [experiment for name, experiment in stack_experiments.items() if name in select_experiments]
        if not select_sequences is None:
            assert isinstance(select_sequences, list)
            sequences = [sequence for sequence in sequences if sequence.name in select_sequences]


        if len(experiments) == 0:
            logger.warning("No experiments selected")

        if len(sequences) == 0:
            logger.warning("No sequences selected")

        # ``asyncio.get_event_loop()`` no longer auto-creates a loop in Python 3.12+
        # and emits the "There is no current event loop in thread 'MainThread'"
        # warning. Create one explicitly here, install it for the duration of
        # the report generation, and tear it down in the ``finally`` block.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            with AnalysisProcessor(executor, cache) as processor:

                order = workspace.report.sort(experiments, trackers, sequences)

                trackers = [trackers[i] for i in order]

                # Schedule ``report.generate`` coroutines on the explicitly-created loop.
                # ``ensure_future`` / ``create_task`` both bind to the running loop;
                # we use ``create_task`` on ``loop`` so the binding is explicit.
                futures = [
                    loop.create_task(report.generate(experiments, trackers, sequences))
                    for report in index
                ]

                progress = Progress("Processing", processor.total)

                def update() -> None:
                    progress.total(processor.total)
                    progress.absolute(processor.total - processor.pending)
                    loop.call_later(1, update)

                update()

                if len(futures) > 0:
                    loop.run_until_complete(wait(futures))

                progress.close()

                reports = dict()

                for future in futures:
                    merge_tree(future.result(), reports)

        finally:

            executor.shutdown(wait=True)
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

        report_storage = workspace.storage.substorage("reports").substorage(name)

        def only_plots(reports, storage: "Storage"):
            """Filter out all non-plot items from the report and save them to storage.

            :param reports: The reports to filter.
            """
            for key, section in reports.items():
                for item in section:
                    if isinstance(item, Plot):
                        logger.debug("Saving plot %s", item.identifier)
                        with storage.write(key + "_" + item.identifier + '.pdf', binary=True) as out:
                            item.save(out, "PDF")
                        with storage.write(key + "_" + item.identifier + '.png', binary=True) as out:
                            item.save(out, "PNG")
                    if isinstance(item, Video):
                        # The HTML document embeds every video itself (as mp4); writing a
                        # standalone .avi here too would render and encode each video a
                        # second time for no consumer. Only emit standalone videos for
                        # formats that do not embed them.
                        if format == "html":
                            continue
                        logger.debug("Saving video %s", item.identifier)
                        with storage.write(key + "_" + item.identifier + '.avi', binary=True) as out:
                            item.save(out, "avi")

        metadata = {"Stack": stack.title}

        # Prune empty sections
        reports = {key: section for key, section in reports.items() if len(section) > 0}

        if format == "html":
            from .html import generate_html_document
            generate_html_document(trackers, sequences, reports, report_storage, metadata=metadata)
        elif format == "latex":
            from .latex import generate_latex_document
            generate_latex_document(trackers, sequences, reports, report_storage, metadata=metadata)
        elif format == "plots":
            only_plots(reports, report_storage)
        else:
            raise ValueError("Unknown report format %s" % format)
        
import vot.report.common
