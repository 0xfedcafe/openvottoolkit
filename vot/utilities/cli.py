"""Command line interface for the toolkit.

This module provides a command line interface for the toolkit. It is used to run
experiments, manage trackers and datasets, and to perform other tasks.
"""

import os
import sys
import argparse
import logging
from typing import Any
import yaml
from datetime import datetime

from .. import check_updates, toolkit_version, get_logger, check_debug
from . import Progress, normalize_path

logger = get_logger()

class EnvDefault(argparse.Action):
    """Argparse action that resorts to a value in a specified envvar if no value is
    provided via program arguments."""
    def __init__(self, envvar: str, required: bool = True, default: Any | None = None, separator: str | None = None, **kwargs: Any) -> None:
        """Initialize the action."""
        # previously if default is 0, false, empty string it would be considered as not provided
        if default is None and envvar and envvar in os.environ:
            default = os.environ[envvar]
        if separator and isinstance(default, str):
            default = default.split(separator)
        if required and default is not None:
            required = False
        self.separator = separator
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        """Call the action."""
        if self.separator and isinstance(values, str):
            values = values.split(self.separator)
        setattr(namespace, self.dest, values)

def do_test(config: argparse.Namespace) -> None:
    """Run a test for a tracker.

    :param config: Configuration
    :type config: argparse.Namespace
    """

    from vot.tracker import Registry
    from vot.tracker.tests import run_tracker_test
    from vot import config as global_config

    trackers = Registry(global_config.registry)

    if not config.tracker:
        logger.error("Unable to continue without a tracker")
        logger.error("List of found trackers: ")
        for k in trackers.identifiers():
            logger.error(" * %s", k)
        return

    if not config.tracker in trackers:
        logger.error("Tracker does not exist")
        return

    tracker = trackers[config.tracker]

    runtime = tracker.runtime(log=True)

    run_tracker_test(runtime, config.visualize, config.sequence, config.ignore)


def do_initialize(config: argparse.Namespace) -> None:
    """Initialize a workspace. If a stack is provided, the workspace is initialized with
    the stack. If no stack is provided, but a dataset exists, then a dummy config can be
    created for this custom dataset. If neither is provided, the user is prompted to
    provide a stack.

    :param config: Configuration
    :type config: argparse.Namespace
    """

    from vot.workspace import WorkspaceException, Workspace
    from ..stack import resolve_stack, list_integrated_stacks

    if Workspace.exists(config.workspace):
        logger.error("Workspace already initialized")
        return

    if config.stack is None:
        if os.path.isfile(os.path.join(config.workspace, "configuration.m")):
            from vot.utilities.migration import migrate_matlab_workspace
            migrate_matlab_workspace(config.workspace)
            return
        elif os.path.isfile(os.path.join(config.workspace, "sequences")):
            sequences_directory = os.path.join(config.workspace, "sequences")
            # Attempt to load a dataset from the sequences directory
            from vot.dataset import load_dataset
            logger.info("Found sequences directory, attempting to load dataset")
            dataset = None
            try:
                dataset = load_dataset(sequences_directory)
                logger.info("Loaded dataset: %s", dataset)
            except Exception:
                pass
            if dataset is not None:
                default_config = dict(dataset=dataset)
                Workspace.initialize(config.workspace, default_config, download=False)
                logger.info("Initialized workspace in '%s'", config.workspace)
            else:
                logger.error("Unable to load dataset from sequences directory")
            return

        else:
            stacks = list_integrated_stacks()
            logger.error("Unable to continue without a stack")
            logger.error("List of available integrated stacks: ")
            for k, v in sorted(stacks.items(), key=lambda x: x[0]):
                logger.error(" * %s - %s", k, v)

            return

    assert config.stack is not None
    stack_file = resolve_stack(config.stack)

    if stack_file is None:
        logger.error("Experiment stack %s not found", config.stack)
        return

    default_config = dict(stack=config.stack, registry=["./trackers.ini"])

    try:
        Workspace.initialize(config.workspace, default_config, download=not config.nodownload)
        logger.info("Initialized workspace in '%s'", config.workspace)
    except WorkspaceException as we:
        logger.error("Error during workspace initialization: %s", we)

def do_evaluate(config: argparse.Namespace) -> None:
    """Run an evaluation for a tracker on an experiment stack and a set of sequences.

    Trackers are taken from ``config.trackers``; the literal reference ``all`` expands to every
    tracker in the registry, and ``#tag`` references expand to all trackers carrying that tag.

    :param config: Configuration
    :type config: argparse.Namespace
    """

    from vot.experiment import run_experiment
    from ..tracker import TrackerException
    from ..workspace import Workspace

    workspace = Workspace.load(config.workspace)

    logger.debug("Loaded workspace in '%s'", config.workspace)

    # The literal "all" expands to every tracker registered in the registry.
    references = config.trackers
    if "all" in references:
        references = workspace.registry.identifiers()
        logger.info("Evaluating all %d trackers in the registry", len(references))

    trackers = workspace.registry.resolve(*references, storage=workspace.storage.substorage("results"), skip_unknown=False)

    if len(trackers) == 0:
        logger.error("Unable to continue without at least on tracker")
        logger.error("List of available found trackers: ")
        for k in workspace.registry.identifiers():
            logger.error(" * %s", k)
        return

    # Filter experiments
    if config.experiments:
        experiments = [v for k, v in workspace.stack.experiments.items() if k in config.experiments.split(",")]
    else:
        experiments = workspace.stack

    if len(experiments) == 0:
        logger.error("No experiments found, stopping.")
        return

    try:
        for tracker in trackers:
            logger.debug("Evaluating tracker %s", tracker.identifier)
            for experiment in experiments:
                run_experiment(experiment, tracker, list(workspace.dataset), config.force, config.persist)

        logger.info("Evaluation concluded successfuly")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by the user")
    except TrackerException as te:
        logger.error("Evaluation interrupted by tracker error: {}".format(te))

def do_analysis(args: argparse.Namespace) -> None:
    """Run an analysis for a tracker on an experiment stack and a set of sequences.
    Analysis results are serialized to disk either as a JSON file or as a YAML file.

    :param args: Configuration
    :type args: argparse.Namespace
    """
    from vot import config

    from vot.analysis import AnalysisProcessor, process_stack_analyses
    from vot.report import generate_serialized
    from ..workspace import Workspace
    from ..workspace.storage import Cache

    workspace = Workspace.load(args.workspace)

    logger.debug("Loaded workspace in '%s'", args.workspace)

    if not args.trackers:
        trackers = workspace.list_results(workspace.registry)
    else:
        trackers = workspace.registry.resolve(*args.trackers, storage=workspace.storage.substorage("results"), skip_unknown=False)

    if not trackers:
        logger.warning("No trackers resolved, stopping.")
        return

    logger.debug("Running analysis for %d trackers", len(trackers))

    if config.worker_pool_size == 1:

        if args.debug:
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

    try:

        with AnalysisProcessor(executor, cache):

            results = process_stack_analyses(workspace, trackers, args.sequences.split(",") if args.sequences else None, args.experiments.split(",") if args.experiments else None)

            if results is None:
                return

            if args.name is None:
                name = "{:%Y-%m-%dT%H-%M-%S.%f%z}".format(datetime.now())
            else:
                name = args.name

            storage = workspace.storage.substorage("analysis")

            sequences = list(workspace.dataset) if args.sequences is None else [s for s in workspace.dataset if s.name in args.sequences.split(",")]

            if args.format == "json":
                generate_serialized(trackers, sequences, results, storage, "json", name)
            elif args.format == "yaml":
                generate_serialized(trackers, sequences, results, storage, "yaml", name)
            else:
                raise ValueError("Unknown format '{}'".format(args.format))

            logger.info("Analysis successful, report available as %s", name)

    finally:

        executor.shutdown(wait=True)

def do_report(config: argparse.Namespace) -> None:
    """Generate a report for a one or multiple trackers on an experiment stack and a set
    of sequences.

    :param config: Configuration
    :type config: argparse.Namespace
    """

    from vot.report import generate_document
    from ..workspace import Workspace


    if config.name is None:
        name = "{:%Y-%m-%dT%H-%M-%S.%f%z}".format(datetime.now())
    else:
        name = config.name

    workspace = Workspace.load(config.workspace)

    logger.debug("Loaded workspace in '%s'", config.workspace)

    if not config.trackers:
        trackers = workspace.list_results(workspace.registry)
    else:
        trackers = workspace.registry.resolve(*config.trackers, storage=workspace.storage.substorage("results"), skip_unknown=False)

    if not trackers:
        logger.warning("No trackers resolved, stopping.")
        return

    logger.debug("Running report generation for %d trackers", len(trackers))

    generate_document(workspace, trackers, config.format, name, config.sequences.split(",") if config.sequences else None, config.experiments.split(",") if config.experiments else None)
    
    logger.info("Report generation successful, document available as %s", name)
    
    
def do_pack(config: argparse.Namespace) -> None:
    """Package results to a ZIP file so that they can be submitted to a challenge.

    :param config: Configuration
    :type config: argparse.Namespace
    """

    import zipfile, io
    from shutil import copyfileobj
    from ..workspace import Workspace
    from vot.utilities.io import YAMLEncoder

    workspace = Workspace.load(config.workspace)

    logger.debug("Loaded workspace in '%s'", config.workspace)

    tracker = workspace.registry[config.tracker]

    logger.info("Packaging results for tracker %s", tracker.identifier)

    all_files = []
    can_finish = True

    with Progress("Scanning", len(workspace.dataset) * len(workspace.stack)) as progress:

        for experiment in workspace.stack:
            sequences = experiment.transform(list(workspace.dataset))
            for sequence in sequences:
                complete, files, results = experiment.scan(tracker, sequence)
                all_files.extend([(f, experiment.identifier, sequence.name, results) for f in files])
                if not complete:
                    logger.error("Results are not complete for experiment %s, sequence %s", experiment.identifier, sequence.name)
                    can_finish = False
                progress.relative(1)

    if not can_finish:
        logger.error("Unable to continue, experiments not complete")
        return

    logger.debug("Collected %d files, compressing to archive ...", len(all_files))

    timestamp = datetime.now()

    archive_name = "{}_{:%Y-%m-%dT%H-%M-%S.%f%z}.zip".format(tracker.identifier, timestamp)

    with Progress("Compressing", len(all_files)) as progress:

        manifest = dict(identifier=tracker.identifier, configuration=tracker.describe(),
            timestamp="{:%Y-%m-%dT%H-%M-%S.%f%z}".format(timestamp), platform=sys.platform,
            python=sys.version, toolkit=toolkit_version(), stack=workspace.dump()["stack"])

        zip_date_time = (timestamp.year, timestamp.month, timestamp.day,
                         timestamp.hour, timestamp.minute, timestamp.second)

        with zipfile.ZipFile(workspace.storage.write(archive_name, binary=True), mode="w") as archive:
            for f in all_files:
                info = zipfile.ZipInfo(filename=os.path.join(f[1], f[2], f[0]), date_time=zip_date_time)
                with archive.open(info, mode="w") as fout, f[3].read(f[0]) as fin:
                    if isinstance(fin, io.TextIOBase):
                        copyfileobj(fin, io.TextIOWrapper(fout))
                    else:
                        copyfileobj(fin, fout)
                progress.relative(1)

            info = zipfile.ZipInfo(filename="manifest.yml", date_time=zip_date_time)
            with io.TextIOWrapper(archive.open(info, mode="w")) as fout:
                yaml.dump(manifest, fout, Dumper=YAMLEncoder)

    logger.info("Result packaging successful, archive available in %s", archive_name)

def print_sequence_list(sequences_dir: str, infos: list) -> None:
    """Pretty-prints a sequence listing with channels, frame count and groundtruth status.

    Frame counts not declared in the sequence metadata are shown inferred from disk, prefixed
    with ``~`` and explained by a footnote.

    Output is colourised only when writing to a terminal so piped output stays plain text.

    :param sequences_dir: Directory the sequences were listed from (shown in the header).
    :param infos: List of :class:`vot.dataset.preparation.SequenceInfo` entries.
    """
    import colorama

    use_color = sys.stdout.isatty()

    def paint(text: str, *styles: str) -> str:
        """Wraps text in ANSI styles when writing to a terminal, otherwise returns it as-is."""
        if not use_color:
            return text
        return "".join(styles) + text + colorama.Style.RESET_ALL

    name_width = max(len(info.name) for info in infos)
    channel_width = max(len(", ".join(info.channels)) if info.channels else 1 for info in infos)

    print()
    print(paint("Sequences in {} ({})".format(sequences_dir, len(infos)), colorama.Style.BRIGHT))
    print()

    any_inferred = any(info.length is None and info.inferred_length is not None for info in infos)

    for info in infos:
        name = paint(info.name.ljust(name_width), colorama.Style.BRIGHT, colorama.Fore.CYAN)
        channels_text = ", ".join(info.channels) if info.channels else "-"
        channels = paint(channels_text.ljust(channel_width), colorama.Fore.YELLOW)

        if info.length is not None:
            length = "{:>6} frames".format(info.length)
        elif info.inferred_length is not None:
            # Metadata declares no length; show the count inferred from disk, marked with '~'.
            length = paint("~{:>5} frames".format(info.inferred_length), colorama.Style.DIM)
        else:
            length = paint("     ? frames", colorama.Style.DIM)

        if not info.present:
            status = paint("missing directory", colorama.Fore.RED)
        elif info.has_groundtruth:
            status = paint("groundtruth", colorama.Fore.GREEN)
        else:
            status = paint("no groundtruth", colorama.Fore.RED)

        print("  {}   {}   {}   {}".format(name, channels, length, status))
    print()

    if any_inferred:
        print(paint("  ~ frame count inferred from groundtruth.txt or channel images "
                     "(not declared in sequence metadata)", colorama.Style.DIM))
        print()


def do_sequences(config: argparse.Namespace) -> None:
    """Dispatcher for the ``vot sequences`` subcommand. Routes to one of the helpers in
    :mod:`vot.dataset.preparation` based on ``config.sequences_action``.

    :param config: Configuration
    :type config: argparse.Namespace
    """
    from vot.dataset import preparation as seq_utils
    from vot.dataset.layout import SequenceList

    action = config.sequences_action
    if action is None:
        config.sequences_parser.print_help()
        return

    if action == "extract-frames":
        seq_utils.extract_frames(config.video, config.output_dir, fps=config.fps,
                                 start_frame=config.start_frame, quality=config.quality,
                                 channel=config.channel)

    elif action == "import-video":
        # The sequences live in ``<workspace>/sequences`` (same convention as ``do_initialize``).
        sequences_dir = os.path.join(config.workspace, "sequences")
        seq_utils.import_video(config.video, sequences_dir, name=config.name,
                               fps=config.fps, quality=config.quality, channel=config.channel)

    elif action == "remove":
        sequences_dir = os.path.join(config.workspace, "sequences")
        if not config.force:
            answer = input("Remove sequence '{}' from {}? [y/N] ".format(config.name, sequences_dir))
            if answer.strip().lower() not in ("y", "yes"):
                logger.info("Aborted")
                return
        seq_utils.remove_sequence(sequences_dir, config.name)

    elif action == "list":
        sequences_dir = os.path.join(config.workspace, "sequences")
        infos = seq_utils.collect_sequence_info(sequences_dir)
        if not infos:
            logger.info("No sequences found in %s", sequences_dir)
        else:
            print_sequence_list(sequences_dir, infos)

    elif action == "yolo-to-vot":
        seq_utils.yolo_to_vot(config.source, config.destination, fps=config.fps)

    elif action == "anchors":
        for sequence_dir in config.sequences:
            seq_utils.generate_anchors_for_sequence(sequence_dir, force=config.force)

    elif action == "metadata":
        for sequence_dir in config.sequences:
            seq_utils.write_sequence_metadata(sequence_dir, fps=config.fps, width=config.width,
                                              height=config.height, length=config.length, force=config.force)

    elif action == "reverse":
        seq_utils.reverse_sequence(config.source, config.destination)

    elif action == "delayed-init":
        seq_utils.delayed_init_variants(config.source, count=config.count, repetitions=config.repetitions,
                                        output_base=config.output_base)

    elif action == "slice":
        # ``take_slice`` is a building block reused by speedup/size-slices/baseline with temp
        # dirs, so it does not auto-register; register the user-facing destination here.
        output_path = seq_utils.take_slice(config.source, config.begin_frame, config.end_frame, config.output_dir)
        SequenceList(output_path.parent).append(output_path.name)

    elif action == "subsample":
        output_path = seq_utils.subsample_sequence(config.source, config.step, config.output_dir)
        SequenceList(output_path.parent).append(output_path.name)

    elif action == "size-slices":
        size_ranges = []
        for spec in config.size_ranges:
            parts = spec.split(":")
            if len(parts) != 3:
                raise argparse.ArgumentTypeError("Invalid size range '{}'; expected min:max:label".format(spec))
            size_ranges.append((float(parts[0]), float(parts[1]), parts[2]))
        seq_utils.create_size_slices(config.source, size_ranges, config.output_dir,
                                     target_frames=config.target_frames,
                                     min_bbox_movements=config.min_bbox_movements,
                                     prefer_no_speedup=not config.allow_speedup,
                                     check_initial_size_only=config.initial_size_only)

    elif action == "speedup":
        seq_utils.create_speedup_experiments(config.source, config.output_dir,
                                             speedup_factors=config.factors,
                                             start_frame=config.start_frame,
                                             end_frame=config.end_frame,
                                             sequence_prefix=config.prefix)

    elif action == "baseline":
        seq_utils.create_baseline_slice(config.source, config.output_dir,
                                        start_frame=config.start_frame, end_frame=config.end_frame,
                                        concatenate_path=config.concatenate)

    else:
        logger.error("Unknown sequences subcommand: %s", action)


def main() -> None:
    """Entrypoint to the toolkit Command Line Interface utility, should be executed as a
    program and provided with arguments."""

    parser = argparse.ArgumentParser(description='VOT Toolkit Command Line Interface', prog="vot")
    parser.add_argument("--debug", "-d", default=False, help="Enable debug mode and verbose logging", required=False, action='store_true')
    parser.add_argument("--registry", default=".", help='Tracker registry paths', required=False)
    parser.add_argument("--version", "-V", action="version", version="%(prog)s " + toolkit_version())

    subparsers = parser.add_subparsers(help='commands', dest='action', title="Commands")

    test_parser = subparsers.add_parser('test', help='Test a tracker integration on a synthetic sequence')
    test_parser.add_argument("tracker", help='Tracker identifier', nargs="?")
    test_parser.add_argument("--visualize", "-g", default=False, required=False, help='Visualize results of the test session', action='store_true')
    test_parser.add_argument("--sequence", "-s", required=False, help='Path to sequence to use instead of dummy')
    test_parser.add_argument("--ignore", required=False, help='Object IDs to ignore', type=lambda x: x.split(","), default=[])

    workspace_parser = subparsers.add_parser('configure', aliases=["initialize"], help='Setup a new workspace and download data')
    workspace_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    workspace_parser.add_argument("--nodownload", default=False, required=False, help="Do not download dataset if specified in stack", action='store_true')
    workspace_parser.add_argument("stack", nargs="?", help='Experiment stack')

    evaluate_parser = subparsers.add_parser('evaluate', aliases=["run"], help='Evaluate one or more trackers in a given workspace')
    evaluate_parser.add_argument("trackers", nargs='+', default=None, help='Tracker identifiers, or "all" to evaluate every tracker in the registry')
    evaluate_parser.add_argument("--force", "-f", default=False, help="Force rerun of the entire evaluation", required=False, action='store_true')
    evaluate_parser.add_argument("--persist", "-p", default=False, help="Persist execution even in case of an error", required=False, action='store_true')
    evaluate_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    evaluate_parser.add_argument("--experiments", default=None, help='Filter specified experiments (comma separated names)', required=False)

    analysis_parser = subparsers.add_parser('analysis', aliases=["analyse", "analyze"], help='Run analysis of results')
    analysis_parser.add_argument("trackers", nargs='*', help='Tracker identifiers')
    analysis_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    analysis_parser.add_argument("--format", choices=("json", "yaml"), default="json", help='Analysis output format')
    analysis_parser.add_argument("--name", required=False, help='Analysis output name')
    analysis_parser.add_argument("--sequences", default=None, help='Filter specified sequences (comma separated names)', required=False)
    analysis_parser.add_argument("--experiments", default=None, help='Filter specified experiments (comma separated names)', required=False)

    report_parser = subparsers.add_parser('report', aliases=["document"], help='Generate report document')
    report_parser.add_argument("trackers", nargs='*', help='Tracker identifiers')
    report_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    report_parser.add_argument("--format", choices=("html", "latex", "plots"), default="html", help='Analysis output format')
    report_parser.add_argument("--name", required=False, help='Document output name')
    report_parser.add_argument("--sequences", default=None, help='Filter specified sequences (comma separated names)', required=False)
    report_parser.add_argument("--experiments", default=None, help='Filter specified experiments (comma separated names)', required=False)

    pack_parser = subparsers.add_parser('pack', help='Package results for submission')
    pack_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path')
    pack_parser.add_argument("tracker", help='Tracker identifier')

    sequences_parser = subparsers.add_parser('sequences', help='Dataset / sequence preparation utilities')
    sequences_subparsers = sequences_parser.add_subparsers(dest='sequences_action', title='Sequences actions')
    # Expose the parser on the namespace so do_sequences can print its usage when no action is given.
    sequences_parser.set_defaults(sequences_parser=sequences_parser)

    extract_parser = sequences_subparsers.add_parser('extract-frames', help='Extract frames from a video file using ffmpeg')
    extract_parser.add_argument("video", help='Input video file (MP4/MOV/AVI/...)')
    extract_parser.add_argument("output_dir", help='Output directory for extracted frames')
    extract_parser.add_argument("--fps", type=float, default=None, help='Target FPS (default: keep source rate)')
    extract_parser.add_argument("--start-frame", type=int, default=1, help='Starting frame number')
    extract_parser.add_argument("--quality", type=int, choices=[1, 2, 3, 4, 5], default=2, help='JPEG quality (1 = best, 5 = worst)')
    extract_parser.add_argument("--channel", default="color", help='Channel subdirectory to nest frames under (e.g. color -> color/%%08d.jpg). Empty string writes frames flat.')

    import_parser = sequences_subparsers.add_parser('import-video', help='Import a video file as a new sequence in the workspace')
    import_parser.add_argument("video", help='Input video file (MP4/MOV/AVI/...)')
    import_parser.add_argument("--name", default=None, help='Sequence/folder name (default: video filename without extension)')
    import_parser.add_argument("--fps", type=float, default=None, help='Target FPS (default: keep source rate)')
    import_parser.add_argument("--quality", type=int, choices=[1, 2, 3, 4, 5], default=2, help='JPEG quality (1 = best, 5 = worst)')
    import_parser.add_argument("--channel", default="color", help='Channel subdirectory to nest frames under (e.g. color -> channels.color=color/%%08d.jpg). Empty string writes a flat sequence.')
    import_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path (default: current directory)')

    remove_parser = sequences_subparsers.add_parser('remove', help='Remove a sequence from the workspace (its directory and list.txt entry)')
    remove_parser.add_argument("name", help='Sequence name to remove')
    remove_parser.add_argument("--force", "-f", action='store_true', help='Do not ask for confirmation')
    remove_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path (default: current directory)')

    list_parser = sequences_subparsers.add_parser('list', help='List sequences in the workspace')
    list_parser.add_argument("--workspace", default=os.getcwd(), help='Workspace path (default: current directory)')

    yolo_parser = sequences_subparsers.add_parser('yolo-to-vot', help='Convert a YOLO-format directory to a VOT sequence')
    yolo_parser.add_argument("source", help='Source directory with .png + .txt YOLO pairs')
    yolo_parser.add_argument("destination", help='Destination directory for the VOT sequence')
    yolo_parser.add_argument("--fps", type=int, default=30, help='Frame rate to record in metadata (default: 30)')

    anchors_parser = sequences_subparsers.add_parser('anchors', help='Generate anchor.value files for one or more sequences')
    anchors_parser.add_argument("sequences", nargs='+', help='Sequence directories to process')
    anchors_parser.add_argument("--force", action='store_true', help='Overwrite existing anchor.value files')

    metadata_parser = sequences_subparsers.add_parser('metadata', help='Generate sequence metadata files for one or more sequences')
    metadata_parser.add_argument("sequences", nargs='+', help='Sequence directories to process')
    metadata_parser.add_argument("--fps", type=float, default=30, help='Frame rate (default: 30)')
    metadata_parser.add_argument("--width", type=int, default=None, help='Frame width (auto-detected if omitted)')
    metadata_parser.add_argument("--height", type=int, default=None, help='Frame height (auto-detected if omitted)')
    metadata_parser.add_argument("--length", type=int, default=None, help='Explicit frame count')
    metadata_parser.add_argument("--force", action='store_true', help='Overwrite existing sequence files')

    reverse_parser = sequences_subparsers.add_parser('reverse', help='Reverse a VOT sequence (frame order + annotations)')
    reverse_parser.add_argument("source", help='Source VOT sequence directory')
    reverse_parser.add_argument("destination", help='Destination directory for the reversed sequence')

    delayed_parser = sequences_subparsers.add_parser('delayed-init', help='Generate delayed-init variants by stripping leading frames')
    delayed_parser.add_argument("source", help='Source VOT sequence directory')
    delayed_parser.add_argument("--count", "-c", type=int, default=50, help='Frames to strip per repetition (default: 50)')
    delayed_parser.add_argument("--repetitions", "-r", type=int, default=10, help='Number of variants to produce (default: 10)')
    delayed_parser.add_argument("--output-base", default=None, help='Parent directory for the variants (default: source parent)')

    slice_parser = sequences_subparsers.add_parser('slice', help='Copy a frame slice into a new sequence directory')
    slice_parser.add_argument("source", help='Source VOT sequence directory')
    slice_parser.add_argument("output_dir", help='Destination directory for the slice')
    slice_parser.add_argument("begin_frame", type=int, help='First frame to include (1-based)')
    slice_parser.add_argument("end_frame", type=int, help='Last frame to include (1-based, inclusive)')

    subsample_parser = sequences_subparsers.add_parser('subsample', help='Keep every step-th frame of a sequence')
    subsample_parser.add_argument("source", help='Source VOT sequence directory')
    subsample_parser.add_argument("output_dir", help='Destination directory for the subsampled sequence')
    subsample_parser.add_argument("step", type=int, help='Subsampling step (2 = every 2nd frame)')

    size_parser = sequences_subparsers.add_parser('size-slices', help='Create test slices by object size range')
    size_parser.add_argument("source", help='Source VOT sequence directory')
    size_parser.add_argument("output_dir", help='Directory where slices are written')
    size_parser.add_argument("--size-range", dest='size_ranges', action='append', required=True,
                             help='Size range as min:max:label (repeatable)')
    size_parser.add_argument("--target-frames", type=int, default=100, help='Target slice length (default: 100)')
    size_parser.add_argument("--min-bbox-movements", type=float, default=1.5, help='Movement threshold (default: 1.5)')
    size_parser.add_argument("--allow-speedup", action='store_true', help='Pick sped-up windows if they have a higher quality score')
    size_parser.add_argument("--initial-size-only", action='store_true', help='Only require the first frame to be in the size range')

    speedup_parser = sequences_subparsers.add_parser('speedup', help='Generate temporally-subsampled speedup variants')
    speedup_parser.add_argument("source", help='Source VOT sequence directory')
    speedup_parser.add_argument("output_dir", help='Parent directory for the generated variants')
    speedup_parser.add_argument("--factors", nargs='+', type=int, default=[2, 3, 4, 5], help='Speedup factors (default: 2 3 4 5)')
    speedup_parser.add_argument("--start-frame", type=int, default=0, help='First frame (0-based, default: 0)')
    speedup_parser.add_argument("--end-frame", type=int, default=None, help='Last frame (0-based, default: last)')
    speedup_parser.add_argument("--prefix", default=None, help='Prefix for variant directory names (default: source folder name)')

    baseline_parser = sequences_subparsers.add_parser('baseline', help='Create a 1x baseline slice, optionally concatenating a second sequence')
    baseline_parser.add_argument("source", help='Source VOT sequence directory')
    baseline_parser.add_argument("output_dir", help='Destination directory')
    baseline_parser.add_argument("--start-frame", type=int, default=0, help='First frame (0-based, default: 0)')
    baseline_parser.add_argument("--end-frame", type=int, default=None, help='Last frame (0-based, default: last)')
    baseline_parser.add_argument("--concatenate", default=None, help='Optional sequence to append after the slice')

    from vot import print_config

    try:

        args = parser.parse_args()

        if args.registry:
            os.environ["VOT_REGISTRY"] = os.pathsep.join(os.environ.get("VOT_REGISTRY", "").split(os.pathsep) + [args.registry])

        if args.debug:
            os.environ["VOT_DEBUG_MODE"] = "1"
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        print_config()
        
        def check_version() -> None:
            """Check if a newer version of the toolkit is available."""
            update, version = check_updates()
            if update:
                logger.warning("A newer version of the VOT toolkit is available (%s), please update.", version)

        if args.action == "test":
            check_version()
            do_test(args)
        elif args.action in ["configure", "initialize"]:
            check_version()
            do_initialize(args)
        elif args.action in ["evaluate", "run"]:
            check_version()
            do_evaluate(args)
        elif args.action in ["analysis", "analyse", "analyze"]:
            check_version()
            do_analysis(args)
        elif args.action in ["report", "document"]:
            check_version()
            do_report(args)
        elif args.action == "pack":
            check_version()
            do_pack(args)
        elif args.action == "sequences":
            do_sequences(args)
        else:
            parser.print_help()

    except argparse.ArgumentError as e:
        logger.error(e)
        exit(-1)
    except Exception as e:
        logger.exception(e, exc_info=check_debug())
        exit(1)

    exit(0)
