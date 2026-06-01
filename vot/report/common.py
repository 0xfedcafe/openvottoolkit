"""Common functions for document generation."""
import os
import math
from collections.abc import Iterator

from attributee import String

from vot.experiment import Experiment
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.report import ScatterPlot, LinePlot, Plot as ReportPlot, Table, SeparableReport, Report
from vot.analysis import Measure, Point, Plot as AnalysisPlot, Curve, Sorting, Axes, Analysis

def read_resource(name: str) -> str:
    """Reads a resource file from the package directory.

    The file is read as a string.
    """
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, "r") as filehandle:
        return filehandle.read()

def per_tracker(a: Analysis) -> bool:
    """Returns true if the analysis is per-tracker."""
    return a.axes == Axes.TRACKERS

def _iter_measures(descriptions: list) -> Iterator[tuple[int, Measure]]:
    """Yields ``(index, description)`` for each non-None :class:`Measure` description."""
    for i, description in enumerate(descriptions):
        if isinstance(description, Measure):
            yield i, description

def extract_measures_table(trackers: list[Tracker], results: dict) -> Table:
    """Extracts a table of measures from the results. The table is a list of lists,
    where each list is a column. The first column is the tracker name, the second column
    is the measure name, and the rest of the columns are the values for each tracker.

    :param trackers: List of trackers.
    :type trackers: list
    :param results: Dictionary of results. It is a dictionary of dictionaries, where the first key is the experiment, and the second key is the analysis. The value is a list of results for each tracker.
    :type results: dict
    """
    table_header = [[], [], []]
    table_data = dict()
    column_order = []

    def safe(value, default):
        return value if not value is None else default

    for experiment, eresults in results.items():
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()

            # Ignore all non per-tracker results
            if not per_tracker(analysis):
                continue

            for i, description in _iter_measures(descriptions):
                table_header[0].append(experiment)
                table_header[1].append(analysis)
                table_header[2].append(description)
                column_order.append(description.direction)

            if aresults is None:
                continue

            for tracker, values in zip(trackers, aresults):
                if not tracker in table_data:
                    table_data[tracker] = list()
                
                for i, description in _iter_measures(descriptions):
                    table_data[tracker].append(values[i] if not values is None else None)

    table_order = []

    for i, order in enumerate(column_order):
        values = [(v[i], k) for k, v in table_data.items()]
        if order == Sorting.ASCENDING:
            values = sorted(values, key=lambda x: safe(x[0], -math.inf), reverse=False)
        elif order == Sorting.DESCENDING:
            values = sorted(values, key=lambda x: safe(x[0], math.inf), reverse=True)
        else:
            table_order.append(None)
            continue
        
        order = dict()
        j = 0
        value = None
    
        # Take into account that some values are the same
        for k, v in enumerate(values):
            j = j if value == v[0] else k + 1
            value = v[0]
            order[v[1]] = j
        table_order.append(order)
 
    return Table(table_header, table_data, table_order)

def extract_plots(trackers: list[Tracker], results: dict, order: list[int] | None = None) -> dict:
    """Extracts plots from the results, grouped by experiment.

    :param trackers: List of trackers.
    :type trackers: list
    :param results: Dictionary of results. It is a dictionary of dictionaries, where the first key is the experiment, and the second key is the analysis. The value is a list of results for each tracker.
    :type results: dict

    :returns: A dict mapping each experiment to a list of ``(title, plot)`` tuples.
    :rtype: dict"""
    plots = dict()
    j = 0

    for experiment, eresults in results.items():
        experiment_plots = list()
        for analysis, aresults in eresults.items():
            descriptions = analysis.describe()

            # Ignore all non per-tracker results
            if not per_tracker(analysis):
                continue

            for i, description in enumerate(descriptions):
                if description is None:
                    continue

                plot_identifier = "%s_%s_%d" % (experiment.identifier, analysis.name, j)
                j += 1

                if isinstance(description, Point) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    xlabel = description.label(0)
                    ylabel = description.label(1)
                    plot = ScatterPlot(plot_identifier, xlabel, ylabel, xlim, ylim, description.trait)
                elif isinstance(description, AnalysisPlot):
                    ylim = (description.minimal, description.maximal)
                    plot = LinePlot(plot_identifier, description.wrt, description.name, None, ylim, description.trait)
                elif isinstance(description, Curve) and description.dimensions == 2:
                    xlim = (description.minimal(0), description.maximal(0))
                    ylim = (description.minimal(1), description.maximal(1))
                    xlabel = description.label(0)
                    ylabel = description.label(1)
                    plot = LinePlot(plot_identifier, xlabel, ylabel, xlim, ylim, description.trait)
                else:
                    continue

                for t in order if order is not None else range(len(trackers)):
                    tracker = trackers[t]
                    values = aresults[t, 0]
                    data = values[i] if not values is None else None
                    plot(tracker, data)

                experiment_plots.append((analysis.title + " - " + description.name, plot))

        plots[experiment] = experiment_plots

    return plots

def format_value(data: object) -> str:
    """Formats a value for display. If the value is a string, it is returned as is. If
    the value is an integer, it is returned as a string. If the value is a float, it is
    returned as a string with 3 decimal places. Otherwise, the value is converted to a
    string.

    :param data: Value to format.

    :returns: Formatted value.
    :rtype: str"""
    if data is None:
        return "N/A"
    if isinstance(data, str):
        return data
    if isinstance(data, int):
        return "%d" % data
    if isinstance(data, float):
        return "%.3f" % data
    return str(data)

def merge_repeats(objects: list[object]) -> list[tuple]:
    """Merges repeated objects in a list into a list of tuples (object, count)."""
    
    if not objects:
        return []

    repeats = []
    previous = objects[0]
    count = 1

    for o in objects[1:]:
        if o == previous:
            count = count + 1
        else:
            repeats.append((previous, count))
            previous = o
            count = 1

    repeats.append((previous, count))

    return repeats

class StackAnalysesPlots(SeparableReport):
    """A document that produces plots for all analyses configures in stack
    experiments."""

    async def perexperiment(self, experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> list[ReportPlot]:

        from vot.report.common import extract_plots

        analyses = experiment.compatible_analyses()

        results = {a: r for a, r in zip(analyses, await self.process(analyses, experiment, trackers, sequences))}

        # Plot in reverse order, with best trackers on top
        z_order = list(reversed(range(len(trackers))))

        return [p for _, p in extract_plots(trackers, {experiment: results}, z_order)[experiment]]

    def compatible(self, experiment: "Experiment") -> bool:
        return True

class StackAnalysesTable(Report):
    """A document that produces plots for all analyses configures in stack
    experiments."""

    async def generate(self, experiments: list["Experiment"], trackers: list["Tracker"], sequences: list["Sequence"]) -> dict:

        from vot.report.common import extract_measures_table

        results = dict()

        for experiment in experiments:
            analyses = experiment.compatible_analyses()
            results[experiment] = {a: r for a, r in zip(analyses, await self.process(analyses, experiment, trackers, sequences))}

        table = extract_measures_table(trackers, results)

        return {"Overview": [table]}

class SequenceSpeedPlots(SeparableReport):
    """Produces a per-sequence per-frame FPS plot for every MultiRun experiment.

    Mirrors :class:`SequenceOverlapPlots`: the report instantiates
    :class:`vot.analysis.speed.SequenceSpeed` itself so the plots show up even when the underlying
    analysis is not declared in the stack."""

    skip_initial = String(default="5", description="Number of leading frames excluded from the FPS curve (init/warmup).")

    async def perexperiment(self, experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> list[ReportPlot]:

        from vot.analysis.speed import SequenceSpeed
        from vot.report import LinePlot

        analysis = SequenceSpeed(skip_initial=int(self.skip_initial))
        results = await self._single_result(analysis, experiment, trackers, sequences)
        if results is None:
            return []

        plots = []
        for s, sequence in enumerate(sequences):
            plot = LinePlot(
                "fps_%s_%s" % (experiment.identifier, sequence.name),
                "Frame", "FPS", (0, None), (0, None), "fps",
            )
            for t, tracker in enumerate(trackers):
                cell = results[t, s]
                if cell is None:
                    continue
                # Cell layout: (avg_fps, avg_time_ms, per_frame_fps, frame_count)
                per_frame_fps = cell[2]
                if not per_frame_fps:
                    continue
                plot(tracker, list(per_frame_fps))
            plots.append(plot)

        return plots

    def compatible(self, experiment):
        from vot.experiment.multirun import MultiRunExperiment
        from vot.experiment.multistart import MultiStartExperiment
        return isinstance(experiment, (MultiRunExperiment, MultiStartExperiment))


class SequenceFailureCurvePlots(SeparableReport):
    """Produces per-sequence cumulative crash / robustness / total curves for every
    supervised experiment.

    Mirrors :class:`SequenceSpeedPlots`: the report instantiates
    :class:`vot.analysis.failures.SequenceFailureCurve` itself so the plots show up even
    when the underlying analysis is not declared in the stack. Each sequence yields three
    plots (crashes, robustness, total), one line per tracker, with the frame index on x."""

    async def perexperiment(self, experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> list[ReportPlot]:

        from vot.analysis.failures import SequenceFailureCurve
        from vot.report import LinePlot

        results = await self._single_result(SequenceFailureCurve(), experiment, trackers, sequences)
        if results is None:
            return []

        # Cell layout: (cumulative_crashes, cumulative_robustness, cumulative_total, frame_count)
        specs = [(0, "Cumulative crashes", "crashes"),
                 (1, "Cumulative robustness", "robustness"),
                 (2, "Cumulative total", "total")]

        plots = []
        for s, sequence in enumerate(sequences):
            for index, ylabel, tag in specs:
                plot = LinePlot(
                    "%s_%s_%s" % (tag, experiment.identifier, sequence.name),
                    "Frame", ylabel, (0, len(sequence)), (0, None), None,
                )
                for t, tracker in enumerate(trackers):
                    cell = results[t, s]
                    if cell is None:
                        continue
                    curve = cell[index]
                    if not curve:
                        continue
                    plot(tracker, list(curve))
                plots.append(plot)

        return plots

    def compatible(self, experiment):
        from vot.experiment.multirun import SupervisedExperiment
        return isinstance(experiment, SupervisedExperiment)


class SequenceOverlapPlots(SeparableReport):
    """A document that produces plots for all analyses configures in stack
    experiments."""

    ignore_masks = String(default="_ignore", description="Object ID used to get ignore masks.")

    async def perexperiment(self, experiment: "Experiment", trackers: list["Tracker"], sequences: list["Sequence"]) -> list[ReportPlot]:

        from vot.analysis.accuracy import Overlaps
        from vot.report import LinePlot

        results = await self._single_result(Overlaps(ignore_masks=self.ignore_masks), experiment, trackers, sequences)
        if results is None:
            return []

        plots = []
        
        for s, sequence in enumerate(sequences):
            plot = LinePlot("overlap_%s_%s" % (experiment.identifier, sequence.name), "Frame", "Overlap", (0, len(sequence)), (0, 1), None)
            
            for t, tracker in enumerate(trackers):
                cell = results[t, s]
                if cell is None:
                    continue
                measurements = cell[0]
                for m in measurements:
                    data = [(i, v) for i, v in zip(m[2], m[1])]
                    plot(tracker, data)

            plots.append(plot)

        return plots

    def compatible(self, experiment):
        from vot.experiment.multirun import MultiRunExperiment
        return isinstance(experiment, MultiRunExperiment)
