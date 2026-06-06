"""HTML report generation.

This module is used to generate HTML reports from the results of the experiments.
"""
import os
import io
import json
import datetime

import dominate
from dominate.tags import h1, h2, table, thead, tbody, tr, th, td, div, li, ol, ul, span, style, link, script, video, a, button, label, input_, meta
from dominate.util import raw, text

from vot import toolkit_version, check_debug, get_logger
from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.workspace import Storage
from vot.report.common import format_value, read_resource, merge_repeats
from vot.report import StyleManager, Table, Plot, Video, VegaSpec
from vot.utilities import Progress
from vot.utilities.data import Grid

ORDER_CLASSES = {1: "first", 2: "second", 3: "third"}

def insert_cell(value, order):
    """Inserts a cell into the data table."""
    attrs = dict(data_sort_value=order, data_value=value)
    if order in ORDER_CLASSES:
        attrs["cls"] = ORDER_CLASSES[order]
    td(format_value(value), **attrs)

def table_cell(value):
    """Returns a cell for the data table."""
    if isinstance(value, str):
        return value
    elif isinstance(value, Tracker):
        return value.label
    elif isinstance(value, Sequence):
        return value.name
    return format_value(value)

def grid_table(data: Grid, rows: list[str], columns: list[str]):
    """Generates a table from a grid object."""

    assert data.dimensions == 2
    assert data.size(0) == len(rows) and data.size(1) == len(columns)

    with table() as element:
        with thead():
            with tr():
                th()
                [th(table_cell(column)) for column in columns]
        with tbody():
            for i, row in enumerate(rows):
                with tr():
                    th(table_cell(row))
                    for value in data.row(i):
                        if isinstance(value, tuple):
                            if len(value) == 1:
                                value = value[0]
                        insert_cell(value, None)

    return element

def generate_html_document(trackers: list[Tracker], sequences: list[Sequence], reports, storage: Storage, metadata: dict | None = None):
    """Generates an HTML document from the results of the experiments.

    :param trackers: List of trackers.
    :type trackers: list
    :param sequences: List of sequences.
    :type sequences: list
    :param reports: Mapping from section name to a list of report items (Table / Plot / Video).
    :type reports: dict
    :param storage: Storage object.
    :type storage: Storage
    :param metadata: Metadata dictionary.
    :type metadata: dict
    """

    # Encoding the preview videos is the slow part of HTML report generation; show a
    # progress bar over them so a long export is not a silent wait.
    video_total = sum(1 for section in reports.values()
                      for item in section if isinstance(item, Video))
    video_progress = Progress("Exporting videos", video_total) if video_total else None

    def insert_video(data: Video):
        """Insert a video into the document."""
        name = data.identifier + ".mp4"

        with storage.write(name, binary=True) as handle:
            data.save(handle, "mp4")

        if video_progress is not None:
            video_progress.relative(1)

        with video(src=name, controls=True, preload="auto", autoplay=False, loop=False, width="100%", height="100%"):
            raw("Your browser does not support the video tag.")

    def insert_figure(figure):
        """Inserts a matplotlib figure into the document."""
        buffer = io.StringIO()
        figure.save(buffer, "SVG")
        raw(buffer.getvalue())

    def insert_vega(item: VegaSpec):
        """Embeds a Vega-Lite spec live and writes the raw spec next to the report."""
        spec_json = json.dumps(item.spec)
        spec_name = item.identifier + ".vl.json"
        with storage.write(spec_name) as handle:
            handle.write(spec_json)

        target = "vega_" + item.identifier
        div(id=target)
        with script(type="text/javascript"):
            raw("vegaEmbed('#%s', %s, {actions: true}).catch(console.error);" % (target, spec_json))

    def insert_mplfigure(figure):
        """Inserts a matplotlib figure into the document."""
        buffer = io.StringIO()
        figure.savefig(buffer, format="SVG", bbox_inches='tight', pad_inches=0.01, dpi=200)
        raw(buffer.getvalue())

    def add_style(name, linked=False):
        """Adds a style to the document."""
        if linked:
            link(rel='stylesheet', href='file://' + os.path.join(os.path.dirname(__file__), name))
        else:
            style(read_resource(name))

    def add_script(name, linked=False):
        """Adds a script to the document."""
        if linked:
            script(type='text/javascript', src='file://' + os.path.join(os.path.dirname(__file__), name))
        else:
            with script(type='text/javascript'):
                raw("//<![CDATA[\n" + read_resource(name) + "\n//]]>")

    logger = get_logger()
    
    legend = StyleManager.default().legend(Tracker)

    doc = dominate.document(title='VOT report')

    linked = check_debug()

    has_vega = any(isinstance(item, VegaSpec) for section in reports.values() for item in section)

    with doc.head:
        meta(charset="utf-8")
        add_style("pure.css", linked)
        add_style("report.css", linked)
        add_style("controls.css", linked)
        add_script("jquery.js", linked)
        add_script("table.js", linked)
        add_script("report.js", linked)
        add_script("controls.js", linked)
        if has_vega:
            # Vega-Lite rendering relies on these CDN bundles (needs network at view time);
            # the static matplotlib heatmap is the offline fallback for the same data.
            script(type="text/javascript", src="https://cdn.jsdelivr.net/npm/vega@5")
            script(type="text/javascript", src="https://cdn.jsdelivr.net/npm/vega-lite@5")
            script(type="text/javascript", src="https://cdn.jsdelivr.net/npm/vega-embed@6")

    # TODO: make table more general (now it assumes a tracker per row)
    def make_table(data: Table):
        """Generates a table from a Table object."""
        if len(data.header[2]) == 0:
            logger.debug("No measures found, skipping table")
        else:
            with table(cls="overview-table pure-table pure-table-horizontal pure-table-striped"):
                with thead():
                    with tr():
                        th()
                        [th(c[0].identifier, colspan=c[1]) for c in merge_repeats(data.header[0])]
                    with tr():
                        th()
                        [th(c[0].title, colspan=c[1]) for c in merge_repeats(data.header[1])]
                    with tr():
                        th("Trackers")
                        [th(c.abbreviation, data_sort="int" if order else "") for c, order in zip(data.header[2], data.order)]
                with tbody():
                    for tracker, row in data.data.items():
                        with tr(data_tracker=tracker.reference):
                            with td():
                                insert_mplfigure(legend.figure(tracker))
                                span(tracker.label)
                            for value, order in zip(row, data.order):
                                insert_cell(value, order[tracker] if not order is None else None)

    def item_kind(item):
        """Human-readable analysis subtype used to group/filter an item in the panel."""
        return item.kind or ("Preview" if isinstance(item, Video) else "Other")

    def vega_identifiers(section):
        """Identifiers in a section that are rendered live as Vega-Lite."""
        return {item.identifier for item in section if isinstance(item, VegaSpec)}

    # A matplotlib Plot sharing its identifier with a Vega spec is the static twin kept for
    # LaTeX/PDF; in HTML the live Vega version is shown instead and the twin is hidden by default.
    duplicate_ids = set()
    kinds = []
    for section in reports.values():
        vega_ids = vega_identifiers(section)
        for item in section:
            if isinstance(item, Table):
                continue
            if isinstance(item, Plot) and item.identifier in vega_ids:
                duplicate_ids.add(item.identifier)
            kind = item_kind(item)
            if kind not in kinds:
                kinds.append(kind)
    has_duplicates = len(duplicate_ids) > 0

    def make_control_panel():
        """A sticky panel that doubles as the tracker legend and the visibility filter."""
        with div(id="control-panel"):
            with div(cls="cp-bar"):
                span("Filters", cls="cp-title")
                button("»", id="cp-collapse", type="button", title="Collapse the panel")
            with div(cls="cp-content"):
                with div(cls="cp-section"):
                    with div(cls="cp-section-head"):
                        span("Trackers")
                        with span(cls="cp-actions"):
                            a("all", data_act="all", data_group="tracker")
                            text(" · ")
                            a("none", data_act="none", data_group="tracker")
                    with ul(cls="cp-list cp-trackers"):
                        for tracker in trackers:
                            with li(data_tracker=legend.number(tracker)):
                                with span(cls="cp-swatch"):
                                    insert_mplfigure(legend.figure(tracker))
                                span(tracker.label, cls="cp-label")
                with div(cls="cp-section"):
                    with div(cls="cp-section-head"):
                        span("Analyses")
                        with span(cls="cp-actions"):
                            a("all", data_act="all", data_group="kind")
                            text(" · ")
                            a("none", data_act="none", data_group="kind")
                    with ul(cls="cp-list cp-kinds"):
                        for kind in kinds:
                            with li(data_kind=kind):
                                span(kind, cls="cp-label")
                if has_duplicates:
                    with label(cls="cp-check"):
                        input_(type="checkbox", id="cp-mpl-dup")
                        text(" Static heatmap duplicates")

    def insert_item(item):
        """Wraps a plot/video/spec in a collapsible, filterable container."""
        duplicate = isinstance(item, Plot) and item.identifier in duplicate_ids
        classes = ["report-item", "video" if isinstance(item, Video) else "plot"]
        if duplicate:
            classes.append("mpl-duplicate")
        with div(cls=" ".join(classes), data_kind=item_kind(item), data_identifier=item.identifier):
            with div(cls="item-head"):
                button("×", cls="item-collapse", type="button", title="Hide or show this item")
                span(item.identifier, cls="item-title")
            with div(cls="item-body"):
                if isinstance(item, Video):
                    insert_video(item)
                elif isinstance(item, VegaSpec):
                    insert_vega(item)
                else:
                    insert_figure(item)

    metadata = metadata or dict()
    metadata["Version"] = toolkit_version()
    metadata["Created"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["Trackers"] = ", ".join([tracker.label for tracker in trackers])
    metadata.setdefault("Sequences", ", ".join([sequence.name for sequence in sequences]))

    with doc:

        make_control_panel()

        with div(id="wrapper"):

            h1("Analysis report document")

            with ul(id="metadata"):
                for key, value in metadata.items():
                    with li():
                        span(key)
                        text(": " + value)

            with div(id="index"):
                h2("Index")
                with ol():
                    for key, _ in reports.items():
                        li(a(key, href="#"+key))

            for key, section in reports.items():

                a(name=key)
                h2(key, cls="section")

                for item in section:
                    if isinstance(item, Table):
                        make_table(item)
                    elif isinstance(item, (Plot, Video, VegaSpec)):
                        insert_item(item)
                    else:
                        logger.warning("Unsupported report item type %s", item)

            with div(id="footer"):
                text("Generated by ")
                a("VOT toolkit", href="https://github.com/votchallenge/toolkit")
                text(" version %s" % toolkit_version())

    if video_progress is not None:
        video_progress.close()

    with storage.write("report.html") as filehandle:
        filehandle.write(doc.render())
