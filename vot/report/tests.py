"""Tests for report plotting helpers."""

import unittest

from vot.report import LinePlot


class TestPlots(unittest.TestCase):
    """Tests for plot axis limit handling."""

    def test_partial_axis_limits_keep_explicit_bound(self):
        """Partial limits should pin the explicit side and auto-scale the open side."""

        plot = LinePlot("partial", "Frame", "Value", None, (0, None), None)

        plot("tracker", [1, 2, 3])

        lower, upper = plot.axes.get_ylim()
        self.assertEqual(lower, 0)
        self.assertGreater(upper, 3)

    def test_complete_axis_limits_are_fixed(self):
        """Complete limits should disable autoscaling for that axis."""

        plot = LinePlot("complete", "Frame", "Value", (0, 10), (0, 1), None)

        plot("tracker", [2, 3, 4])

        self.assertEqual(plot.axes.get_xlim(), (0, 10))
        self.assertEqual(plot.axes.get_ylim(), (0, 1))


class TestExtractPlots(unittest.TestCase):
    """Tests for :func:`vot.report.common.extract_plots`."""

    def test_returns_dict(self):
        """The result is a dict keyed by experiment (not a list)."""
        from vot.report.common import extract_plots
        self.assertEqual(extract_plots([], {}), {})


class _FakeRegion:
    """A minimal region exposing ``bounds()`` for diagonal computation."""

    def __init__(self, width, height):
        self._width, self._height = width, height

    def bounds(self):
        return (0, 0, self._width, self._height)


class _FakeSequence:
    """A minimal sequence exposing ``name`` and ``groundtruth()``."""

    def __init__(self, name, boxes=None):
        self.name = name
        self._boxes = boxes or []

    def groundtruth(self, index=None):
        return [_FakeRegion(w, h) for w, h in self._boxes]


class _FakeTracker:
    """A minimal tracker exposing ``reference``, ``label`` and ``metadata()``."""

    def __init__(self, reference, label, category):
        self.reference = reference
        self.label = label
        self._category = category

    def metadata(self, key):
        return self._category


class TestFailureHeatmap(unittest.TestCase):
    """Tests for the failure/crash heatmap report (rendering-independent logic)."""

    def test_size_column_value_rounds_and_merges(self):
        """axis='size' yields the rounded diagonal as the column identity, so near-equal
        slices share one column."""
        from vot.report.heatmap import FailureHeatmap
        report = FailureHeatmap(axis="size", diagonal="mean", decimals=0)

        a = report._column_value(_FakeSequence("a", [(3, 4)]))        # diagonal 5.0
        b = report._column_value(_FakeSequence("b", [(3.1, 4.0)]))    # diagonal ~5.06
        self.assertEqual(a, (5, "5"))
        self.assertEqual(a[0], b[0])  # same column identity

    def test_name_column_value_captures_factor(self):
        """axis='name' captures the numeric column value from the sequence name."""
        from vot.report.heatmap import FailureHeatmap
        report = FailureHeatmap(axis="name", pattern=r"_speedup_(\d+)x_")
        self.assertEqual(report._column_value(_FakeSequence("forth_speedup_4x_428f")), (4, "4"))
        self.assertIsNone(report._column_value(_FakeSequence("no_factor_here")))

    def test_category_order_overrides_first_seen(self):
        """``category_order`` reorders the bands; unlisted categories are appended."""
        from vot.report.heatmap import FailureHeatmap
        report = FailureHeatmap(category_order="Siamese, Correlation")
        trackers = [
            _FakeTracker("cf", "KCF", "Correlation"),
            _FakeTracker("sn", "SiamFC", "Siamese"),
            _FakeTracker("ot", "Weird", "Misc"),
        ]
        ordered = [cat for cat, _ in report._ordered_categories(trackers)]
        self.assertEqual(ordered, ["Siamese", "Correlation", "Misc"])

    def test_vega_spec_structure(self):
        """The Vega spec has one band per category, a cell per (tracker, column) and a
        colour domain matching the maximum value."""
        from vot.report.heatmap import FailureHeatmap
        report = FailureHeatmap()
        t = _FakeTracker("kcf", "KCF", "Correlation")
        categories = [("Correlation", [t])]
        spec = report._vega_spec("id", "Failures", categories, ["5", "8"], {"kcf": [1.0, 3.0]})

        self.assertEqual(len(spec["vconcat"]), 1)
        self.assertEqual(len(spec["data"]["values"]), 2)
        domain = spec["vconcat"][0]["layer"][0]["encoding"]["color"]["scale"]["domain"]
        self.assertEqual(domain, [0, 3])

    def test_region_diagonal_uses_float_extents_for_rectangles(self):
        """Rectangle diagonals come from the float width/height the groundtruth
        stores; ``bounds()`` quantizes to whole pixels, which shifts the size
        axis at px-scale objects (and can relabel heatmap columns)."""
        import math
        from vot.region import Rectangle, Special, SpecialCode
        from vot.report.heatmap import _region_diagonal

        # int-rounded bounds() would give hypot(4, 4) ~= 5.66 here
        diagonal = _region_diagonal(Rectangle(0.6, 0.6, 4.8, 4.8))
        assert diagonal is not None
        self.assertAlmostEqual(diagonal, math.hypot(4.8, 4.8), places=4)
        self.assertIsNone(_region_diagonal(Special(SpecialCode.UNKNOWN)))

    def test_merged_columns_average_not_sum(self):
        """Sequences sharing a column contribute their mean, not a raw sum — a
        sum scales with the column population (e.g. 4 merged slices) rather than
        tracker behaviour."""
        import asyncio
        from vot.report import VegaSpec
        from vot.report.heatmap import FailureHeatmap

        class _Grid:
            def __init__(self, cells):
                self._cells = cells

            def __getitem__(self, key):
                return (self._cells[key],)

        class _Stubbed(FailureHeatmap):
            async def _single_result(self, analysis, experiment, trackers, sequences):
                if type(analysis).__name__ != "FailureCount":
                    return None
                return _Grid({(0, 0): 1.0, (0, 1): 3.0, (0, 2): 5.0})

        class _Experiment:
            identifier = "exp"

            def gather(self, tracker, sequence):
                return [object()]

        report = _Stubbed(axis="name", pattern=r"_(\d+)x")
        sequences = [_FakeSequence("a_1x"), _FakeSequence("b_1x"), _FakeSequence("c_2x")]
        trackers = [_FakeTracker("t0", "T0", "Cat")]

        items = asyncio.run(report.perexperiment(_Experiment(), trackers, sequences))

        spec = next(item for item in items if isinstance(item, VegaSpec)).spec
        cells = {value["Column"]: value["Value"] for value in spec["data"]["values"]}
        self.assertEqual(cells["1"], 2.0)  # mean of 1 and 3, not the sum 4
        self.assertEqual(cells["2"], 5.0)  # singleton column unchanged

    def test_vega_spec_orders_axes_explicitly(self):
        """The Vega spec pins both axis orders: numeric column order (the default
        lexicographic ordinal sort renders '10' before '2') and the registry
        tracker order of the matplotlib twin."""
        from vot.report.heatmap import FailureHeatmap

        report = FailureHeatmap()
        beta = _FakeTracker("b", "Beta", "Cat")
        alpha = _FakeTracker("a", "Alpha", "Cat")
        labels = ["2", "3", "10"]
        grid = {"b": [0.0, 0.0, 0.0], "a": [0.0, 0.0, 0.0]}

        spec = report._vega_spec("id", "Failures", [("Cat", [beta, alpha])], labels, grid)

        encoding = spec["vconcat"][0]["encoding"]
        self.assertEqual(encoding["x"]["sort"], labels)
        self.assertEqual(encoding["y"]["sort"], ["Beta", "Alpha"])

    def test_mpl_plot_saves_svg(self):
        """The matplotlib twin renders to SVG through the Plot interface."""
        import io
        from vot.report.heatmap import FailureHeatmap
        from vot.report import Plot
        report = FailureHeatmap()
        t = _FakeTracker("kcf", "KCF", "Correlation")
        plot = report._mpl_plot("id", "Failures", [("Correlation", [t])], ["5", "8"], {"kcf": [1.0, 3.0]})

        self.assertIsInstance(plot, Plot)
        buffer = io.StringIO()
        plot.save(buffer, "SVG")
        self.assertIn("svg", buffer.getvalue()[:512].lower())


class _SpyExperiment:
    """An experiment stub that counts ``select`` / ``transform`` applications."""

    def __init__(self):
        self.select_calls = 0
        self.transform_calls = 0

    def select(self, sequences):
        self.select_calls += 1
        return list(sequences)

    def transform(self, sequences):
        self.transform_calls += 1
        return list(sequences)


class TestReportTransformOnce(unittest.TestCase):
    """Sequences must be selected+transformed exactly once per report path.

    Regression guard: the ``SeparableReport`` path transforms in ``generate`` before
    ``perexperiment`` and must NOT transform again when committing analyses, otherwise a
    splitting/resizing transformer is applied twice — producing sequences that were never
    evaluated (``MissingResultsException``) and plot ranges misaligned with the results.
    """

    def test_process_selects_and_transforms_once(self):
        """``process`` (the raw-dataset entry point, e.g. the table) transforms once."""
        import asyncio
        from vot.report import Report

        report = Report()
        exp = _SpyExperiment()

        async def fake_commit(analyses, experiment, trackers, sequences):
            return iter([])

        report._commit = fake_commit
        asyncio.run(report.process([], exp, [], ["s1", "s2"]))
        self.assertEqual(exp.select_calls, 1)
        self.assertEqual(exp.transform_calls, 1)

    def test_single_result_does_not_transform(self):
        """``_single_result`` (the SeparableReport path) never re-transforms."""
        import asyncio
        from vot.report import Report

        report = Report()
        exp = _SpyExperiment()

        async def fake_commit(analyses, experiment, trackers, sequences):
            return iter(["result"])

        report._commit = fake_commit
        out = asyncio.run(report._single_result(object(), exp, [], ["s1"]))
        self.assertEqual(out, "result")
        self.assertEqual(exp.transform_calls, 0)
        self.assertEqual(exp.select_calls, 0)


class TestLatexReport(unittest.TestCase):
    """Tests for the LaTeX document generator."""

    def test_does_not_write_stray_pdf_to_cwd(self):
        """Report generation must write only through ``storage``, never to the CWD."""
        import os
        import tempfile

        from vot.report import LinePlot
        from vot.report.latex import generate_latex_document
        from vot.workspace.storage import LocalStorage

        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = os.getcwd()
            os.chdir(directory)
            try:
                storage = LocalStorage(os.path.join(directory, "store"))
                plot = LinePlot("plot1", "Frame", "Value", (0, 1), (0, 1), None)
                generate_latex_document([], [], {"Section": [plot]}, storage, multipart=True)
                stray = [name for name in os.listdir(directory) if name.endswith(".pdf")]
                self.assertEqual(stray, [], "no PDF should be written to the working directory")
            finally:
                os.chdir(previous_cwd)
