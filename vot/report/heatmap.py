"""Failure / crash heatmaps for supervised experiments.

The standard AR table reports Robustness/Crashes as a *per-sequence mean* (total
divided by the number of sequences), which hides how many incidents actually happen
and at which point of an experiment's sweep. This report instead lays the raw
per-(tracker, sequence) failure and crash counts out as a heatmap: one row per
tracker (grouped into category bands), one column per sweep value, and the count in
each cell -- exactly the "robustness scores" figure used in the LWIR tracking paper.

The column axis is derived per sequence in one of two ways (``axis`` option):

  * ``size`` -- the groundtruth bounding-box diagonal (pixels), e.g. the size study,
  * ``name`` -- a number captured from the sequence name with ``pattern``, e.g.
    ``_speedup_(\\d+)x_`` for the temporal sub-sampling experiment.

Sequences mapping to the same column value are summed. Tracker categories come from
the ``meta_category`` field in the tracker registry (``category`` metadata key);
trackers without one fall into an "Other" band.

Each compatible experiment yields up to two heatmaps (failures, crashes). Every
heatmap is emitted twice: as a live Vega-Lite spec (:class:`vot.report.VegaSpec`,
also written to ``<id>.vl.json``) and as a static matplotlib figure
(:class:`vot.report.Plot`, so it renders in the LaTeX/PDF report too).
"""

import os
import re
import math
from typing import Any

from attributee import String, Integer, Boolean

from vot.experiment import Experiment
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.report import SeparableReport, Plot as ReportPlot, VegaSpec, Coverage


# Metric name -> (analysis class import path attribute, human label, vega title noun)
_METRICS = [
    ("failures", "FailureCount", "Failures (track loss)"),
    ("crashes", "CrashCount", "Crashes (process)"),
]


def _region_diagonal(region) -> float | None:
    """Bounding-box diagonal (px) of a region, or None for special/empty regions."""
    try:
        x1, y1, x2, y2 = region.bounds()
    except Exception:
        return None
    w, h = float(x2 - x1), float(y2 - y1)
    if w <= 0 and h <= 0:
        return None
    return math.hypot(w, h)


class FailureHeatmap(SeparableReport):
    """Per-(tracker, sweep-value) failure and crash heatmaps, grouped by category."""

    axis = String(default="size", description="How to derive the column value of a "
                  "sequence: 'size' (groundtruth diagonal) or 'name' (regex capture).")
    diagonal = String(default="mean", description="Which groundtruth diagonal to use "
                      "when axis='size': mean, init, min or max.")
    pattern = String(default="", description="Regex with one capture group used to read "
                     "the column value from the sequence name when axis='name'.")
    numeric = Boolean(default=True, description="Treat the column value as a number "
                      "(numeric sort and rounded label) rather than a string.")
    axis_title = String(default="", description="X-axis title; defaults from 'axis'.")
    category = String(default="category", description="Tracker metadata key to group rows by.")
    category_order = String(default="", description="Comma-separated category band order; "
                            "categories not listed are appended in first-seen order.")
    sequences = String(default="", description="Regex; only matching sequence names are included.")
    experiments = String(default="", description="Regex; only matching experiment ids are processed.")
    decimals = Integer(default=0, description="Decimal places for the displayed cell value.")
    scheme = String(default="viridis", description="Vega-Lite colour scheme for the cells.")

    def compatible(self, experiment: Experiment) -> bool:
        """Compatible with supervised experiments whose id matches ``experiments``."""
        from vot.experiment.multirun import SupervisedExperiment
        if not isinstance(experiment, SupervisedExperiment):
            return False
        if self.experiments and not re.search(self.experiments, experiment.identifier):
            return False
        return True

    # -- column-value derivation ------------------------------------------------

    def _column_value(self, sequence: "Sequence"):
        """Return the (sortkey, label) for a sequence, or None to drop it."""
        if self.axis == "name":
            if not self.pattern:
                return None
            m = re.search(self.pattern, sequence.name)
            if not m:
                return None
            raw = m.group(1)
        else:  # size
            groundtruth = sequence.groundtruth()
            diags = [d for d in (_region_diagonal(r) for r in groundtruth) if d is not None]
            if not diags:
                return None
            if self.diagonal == "init":
                raw = diags[0]
            elif self.diagonal == "min":
                raw = min(diags)
            elif self.diagonal == "max":
                raw = max(diags)
            else:
                raw = sum(diags) / len(diags)

        if self.numeric:
            # Round to the displayed precision so it is the column *identity*: slices that
            # land on the same size/factor share one column (their counts are summed),
            # keeping the matplotlib grid and the Vega spec in lock-step.
            value = round(float(raw), self.decimals)
            if self.decimals <= 0:
                value = int(value)
                label = "%d" % value
            else:
                label = "%.*f" % (self.decimals, value)
            return value, label
        return str(raw), str(raw)

    # -- grid assembly ----------------------------------------------------------

    def _ordered_categories(self, trackers: list["Tracker"]):
        """List of (category, [trackers]); honours ``category_order`` then first-seen.

        Uncategorised trackers fall into 'Other'.
        """
        order: list[str] = []
        groups: dict[str, list[Tracker]] = {}
        for tracker in trackers:
            cat = str(tracker.metadata(self.category) or "Other")
            if cat not in groups:
                groups[cat] = []
                order.append(cat)
            groups[cat].append(tracker)

        if self.category_order:
            preferred = [c.strip() for c in self.category_order.split(",") if c.strip()]
            order = [c for c in preferred if c in groups] + [c for c in order if c not in preferred]
        return [(cat, groups[cat]) for cat in order]

    async def perexperiment(self, experiment: "Experiment", trackers: list["Tracker"],
                            sequences: list["Sequence"]) -> list:
        from vot.analysis import failures as failures_module

        # Restrict and map sequences to column values.
        seq_pattern = re.compile(self.sequences) if self.sequences else None
        columns: dict[Any, str] = {}          # sortkey -> label
        seq_column: dict[str, Any] = {}        # sequence name -> sortkey
        kept_sequences: list[Sequence] = []
        for sequence in sequences:
            if seq_pattern is not None and not seq_pattern.search(sequence.name):
                continue
            # Skip sequences this experiment was not evaluated on (the dataset/list.txt
            # may be broader than the experiment, e.g. baseline sequences); otherwise the
            # per-sequence count analysis would raise on missing results.
            if not any(experiment.gather(tracker, sequence) for tracker in trackers):
                continue
            cv = self._column_value(sequence)
            if cv is None:
                continue
            sortkey, label = cv
            columns[sortkey] = label
            seq_column[sequence.name] = sortkey
            kept_sequences.append(sequence)

        if not kept_sequences or not columns:
            return []

        column_keys = sorted(columns.keys())
        column_labels = [columns[k] for k in column_keys]
        col_index = {k: i for i, k in enumerate(column_keys)}
        categories = self._ordered_categories(trackers)

        items: list = []
        for metric, analysis_name, metric_label in _METRICS:
            analysis = getattr(failures_module, analysis_name)()
            results = await self._single_result(analysis, experiment, trackers, kept_sequences)
            if results is None:
                continue

            # tracker reference -> [per-column summed count]
            grid: dict[str, list[float]] = {
                t.reference: [0.0] * len(column_keys) for t in trackers
            }
            for ti, tracker in enumerate(trackers):
                row = grid[tracker.reference]
                for si, sequence in enumerate(kept_sequences):
                    cell = results[ti, si]
                    if cell is None:
                        continue
                    row[col_index[seq_column[sequence.name]]] += float(cell[0])

            total = sum(sum(r) for r in grid.values())
            if metric == "crashes" and total == 0:
                # Don't clutter the report with an all-zero crash heatmap.
                continue

            identifier = "%s_heatmap_%s" % (metric, experiment.identifier)
            kind = "%s heatmap" % metric_label
            items.append(VegaSpec(identifier, self._vega_spec(
                identifier, metric_label, categories, column_labels, grid), kind=kind))
            items.append(self._mpl_plot(
                identifier, metric_label, categories, column_labels, grid, kind=kind))

        if items:
            items.append(Coverage(experiment.identifier, [s.name for s in kept_sequences]))
        return items

    def _axis_title(self) -> str:
        if self.axis_title:
            return self.axis_title
        return "Object size - diagonal (px)" if self.axis == "size" else "Sequence value"

    def _round(self, value: float) -> float:
        return round(value, self.decimals) if self.decimals > 0 else float(round(value))

    # -- Vega-Lite --------------------------------------------------------------

    def _vega_spec(self, identifier, metric_label, categories, column_labels, grid) -> dict:
        """Build a category-banded Vega-Lite heatmap matching the paper figures."""
        values = []
        vmax = 0.0
        for cat, trackers in categories:
            for tracker in trackers:
                row = grid[tracker.reference]
                for label, raw in zip(column_labels, row):
                    v = self._round(raw)
                    vmax = max(vmax, v)
                    values.append({"Category": cat, "Tracker": tracker.label,
                                   "Column": label, "Value": v})

        bands = []
        last = len(categories) - 1
        for i, (cat, _trackers) in enumerate(categories):
            x_axis = None
            if i == last:
                x_axis = {"labelAngle": 0, "title": self._axis_title(),
                          "titlePadding": 16, "titleFontSize": 15, "labelFontSize": 12}
            color_legend = None
            if i == last:
                color_legend = {"title": metric_label, "orient": "bottom",
                                "direction": "horizontal", "gradientLength": 200}
            text_threshold = vmax * 0.65
            bands.append({
                "title": {"text": cat.upper(), "fontSize": 12, "color": "#333",
                          "anchor": "start", "offset": 8, "fontWeight": 600},
                "transform": [{"filter": "datum.Category == '%s'" % cat.replace("'", "\\'")}],
                "width": {"step": 35}, "height": {"step": 30},
                "encoding": {
                    "x": {"field": "Column", "type": "ordinal", "axis": x_axis},
                    "y": {"field": "Tracker", "type": "nominal",
                          "axis": {"title": None, "minExtent": 100, "maxExtent": 100}},
                },
                "layer": [
                    {"mark": {"type": "rect", "cornerRadius": 2, "stroke": "white", "strokeWidth": 2},
                     "encoding": {"color": {"field": "Value", "type": "quantitative",
                                            "scale": {"scheme": self.scheme, "domain": [0, max(vmax, 1)]},
                                            "legend": color_legend}}},
                    {"mark": "text",
                     "encoding": {"text": {"field": "Value", "type": "quantitative",
                                           "format": ".%df" % self.decimals},
                                  "color": {"condition": {"test": "datum['Value'] > %g" % text_threshold,
                                                          "value": "black"}, "value": "white"}}},
                ],
            })

        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "background": "white",
            "title": {"text": "Tracker %s" % metric_label,
                      "subtitle": "Per-sequence count (darker is better / more stable)",
                      "fontSize": 18, "anchor": "start", "subtitleColor": "#666", "offset": 16},
            "config": {"axis": {"domain": False, "ticks": False, "labelFontSize": 11,
                                "labelColor": "#555", "titleColor": "#888"},
                       "view": {"stroke": "transparent"},
                       "text": {"fontWeight": "bold"}},
            "data": {"values": values},
            "vconcat": bands,
            "spacing": 10,
        }

    # -- matplotlib -------------------------------------------------------------

    def _mpl_plot(self, identifier, metric_label, categories, column_labels, grid, kind=None) -> "ReportPlot":
        """Static matplotlib equivalent of the Vega heatmap, for LaTeX/PDF output."""
        import numpy as np
        import matplotlib
        from matplotlib.figure import Figure

        ncols = len(column_labels)
        vmax = max(1.0, max(self._round(v) for r in grid.values() for v in r))
        cmap = matplotlib.colormaps[self.scheme]

        nrows_total = sum(len(ts) for _c, ts in categories)
        fig = Figure(figsize=(max(4.0, 0.55 * ncols + 2.2), 0.42 * nrows_total + 0.9 * len(categories) + 1.0))
        fig.patch.set_facecolor("white")
        height_ratios = [len(ts) for _c, ts in categories]
        gs = fig.add_gridspec(len(categories), 1, height_ratios=height_ratios, hspace=0.65)

        for gi, (cat, trackers) in enumerate(categories):
            ax = fig.add_subplot(gs[gi])
            data = np.array([[self._round(v) for v in grid[t.reference]] for t in trackers], dtype=float)
            ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(cat.upper(), loc="left", fontsize=9, fontweight="bold", color="#333", pad=4)
            ax.set_yticks(range(len(trackers)))
            ax.set_yticklabels([t.label for t in trackers], fontsize=8)
            ax.set_xticks(range(ncols))
            if gi == len(categories) - 1:
                ax.set_xticklabels(column_labels, fontsize=9)
                ax.set_xlabel(self._axis_title(), fontsize=11, labelpad=8)
            else:
                ax.set_xticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            thr = vmax * 0.65
            fmt = "%.{}f".format(self.decimals)
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    ax.text(c, r, fmt % data[r, c], ha="center", va="center", fontsize=8,
                            color="black" if data[r, c] > thr else "white")

        fig.suptitle("Tracker %s" % metric_label, x=0.02, ha="left", fontsize=13, fontweight="bold")
        return _FigurePlot(identifier, fig, kind=kind)


class _FigurePlot(ReportPlot):
    """A :class:`vot.report.Plot` wrapping a pre-built matplotlib figure.

    Bypasses the base figure machinery so we can lay out a multi-panel heatmap
    ourselves while still being rendered by every backend's ``Plot`` branch.
    """

    def __init__(self, identifier: str, figure, kind: str | None = None) -> None:
        self._identifier = identifier
        self._kind = kind
        self._figure = figure

    def save(self, output, fmt: str) -> None:
        """Saves the wrapped figure, keeping the white background (not transparent)."""
        self._figure.savefig(output, format=fmt, bbox_inches="tight", pad_inches=0.05,
                             facecolor="white")
