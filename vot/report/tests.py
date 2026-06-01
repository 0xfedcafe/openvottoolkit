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
