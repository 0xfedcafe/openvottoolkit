"""Video report helpers and report element implementations."""

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from attributee import Boolean

from vot.dataset import Sequence
from vot.dataset.proxy import FrameMapSequence
from vot.region import SpecialCode, is_special
from vot.region.shapes import Mask
from vot.tracker import Tracker
from vot.tracker.results import Trajectory
from vot.experiment.multirun import MultiRunExperiment, Experiment
from vot.experiment.multistart import MultiStartExperiment, find_anchors
from vot.report import ObjectVideo, SeparableReport

# Experiment types that expose a ``gather`` method returning full-length trajectories in
# sequence coordinates. ``MultiStartExperiment.gather`` remaps its per-anchor runs so they
# can be drawn the same way as multi-run repetitions.
_VIDEO_EXPERIMENTS = (MultiRunExperiment, MultiStartExperiment)

# Lower bound for the shared decoded-frame cache. The cache is actually sized to the
# longest sequence being previewed (see ``perexperiment``): a multistart sequence yields
# many overlapping anchor-run videos that read the same source frames, and a fixed bound
# smaller than the sequence makes every one of those videos re-decode every frame.
_IMAGE_CACHE_FRAMES = 512


@dataclass(frozen=True, slots=True)
class _PreviewRun:
    """One preview video to generate for a sequence.

    A multistart experiment yields one run per anchor (each trimmed to the frames that
    anchor run covers, in the order the tracker actually ran them); other experiments
    yield a single full-length run.
    """

    suffix: str
    """Identifier/filename suffix that distinguishes this run's video ("" for a single run)."""

    label: str | None
    """Extra text describing the run, appended to the video's overlay label, or ``None``."""

    frames: tuple[int, ...] | None
    """Absolute sequence frame indices the video covers, in play order; ``None`` for the
    whole sequence in natural order. A backward multistart run lists them descending
    (anchor first) so the preview plays in the tracker's run direction."""

    index: int | None
    """Position of the run in ``gather()``'s trajectory list; ``None`` draws every trajectory."""


def _region_bounds(region) -> tuple[int, int, int, int] | None:
    """Returns the ``(x1, y1, x2, y2)`` bounding box of a shape region, or ``None`` for
    special / empty regions that have no spatial extent."""
    from vot.region.shapes import Shape
    if not isinstance(region, Shape):
        return None
    try:
        x1, y1, x2, y2 = region.bounds()
    except Exception:
        return None
    return int(x1), int(y1), int(x2), int(y2)


class _LabeledObjectVideo(ObjectVideo):
    """An :class:`ObjectVideo` that draws a fixed text title and a ``frame / total`` counter onto
    every frame. Used by :class:`PreviewVideos` (especially in ``separate=True`` mode) so each output
    video is visually identifiable.

    It can additionally draw a per-region text label next to a bounding box (see
    :meth:`set_box_label`), so trackers stacked into a single video can be told apart.
    """

    def __init__(self, identifier: str, frames, fps: int = 10, trait: str | None = None,
                 label: str | None = None, abs_frames: list[int] | None = None,
                 image_cache: MutableMapping | None = None, cache_key: str | None = None) -> None:
        super().__init__(identifier, frames, fps=fps, trait=trait)
        self._label = label
        # Maps a region key (as passed to ``draw``) to a text label drawn next to that
        # region's bounding box. Used to identify trackers when several share one video.
        self._box_labels: dict[str, str] = {}
        # ``abs_frames[f]`` is the absolute source-sequence index shown at video frame
        # ``f`` (a multistart anchor run covers only a sub-range, and a backward run
        # lists them descending). ``image_cache`` is shared across the preview videos of
        # one report run so overlapping anchor runs decode each source frame only once;
        # it is keyed by that absolute index.
        self._abs_frames = abs_frames
        self._image_cache = image_cache
        self._cache_key = cache_key
        # Video-frame indices at which the previewed tracker emitted a FAILURE region
        # (tracking-quality loss) or a CRASH region (process exception / timeout). Only
        # populated for single-tracker (``separate``) videos; ``render`` turns each into
        # a running tally. ``None`` leaves the corresponding counter off.
        self._failure_frames: list[int] | None = None
        self._crash_frames: list[int] | None = None
        self._lost_frames: list[int] | None = None
        self._cached_frames: frozenset[int] | None = None

    def _load_image(self, frame: int) -> npt.NDArray[np.uint8]:
        if self._image_cache is None:
            return super()._load_image(frame)
        abs_frame = self._abs_frames[frame] if self._abs_frames is not None else frame
        key = (self._cache_key, abs_frame)
        cached = self._image_cache.get(key)
        if cached is None:
            cached = super()._load_image(frame)
            self._image_cache[key] = cached
        # ImageDrawHandle draws onto the array in place, so hand out a private copy
        # and keep the cached one pristine.
        return cached.copy()

    def set_box_label(self, key: str, text: str) -> None:
        """Register a text label to be drawn next to the bounding box stored under ``key``."""
        self._box_labels[key] = text

    def set_failures(self, frames: list[int]) -> None:
        """Record the video-frame indices at which the tracker failed; :meth:`render` then
        draws a running count of failures up to the current frame."""
        self._failure_frames = sorted(frames)

    def set_crashes(self, frames: list[int]) -> None:
        """Record the video-frame indices at which the tracker process crashed; :meth:`render`
        then draws a running count of crashes up to the current frame, separately from
        tracking failures."""
        self._crash_frames = sorted(frames)

    def set_losses(self, frames: list[int]) -> None:
        """Record video-frame indices where the tracker emitted a zero-pixel Mask;
        :meth:`render` draws a running tally beside failures and crashes."""
        self._lost_frames = sorted(frames)

    def set_cached_frames(self, frames: list[int]) -> None:
        """Record video-frame indices where the realtime runtime replayed a cached
        status. :meth:`render` draws a "cached" badge on each such frame."""
        self._cached_frames = frozenset(frames)

    def _box_label_color(self, key: str) -> tuple[int, int, int]:
        """Returns the RGB colour of region ``key`` so its label matches the drawn box."""
        try:
            from vot.utilities.draw import Color
            color = self._manager.plot_style(key).region_style().get("color")
            if not isinstance(color, Color):
                return (255, 255, 255)
            r, g, b, _ = color.to_int()
            return (r, g, b)
        except Exception:
            return (255, 255, 255)

    def render(self, frame: int) -> npt.NDArray[np.uint8]:
        image = super().render(frame)
        try:
            import cv2
        except ImportError:
            return image

        # ``ImageDrawHandle.array`` may return a read-only view of the underlying buffer; OpenCV
        # refuses to write into a read-only ndarray, so take a writable copy here. ``np.copy`` is
        # used because ``np.ascontiguousarray`` may return the same (still read-only) buffer when
        # the array is already C-contiguous.
        if not image.flags.writeable:
            image = np.copy(image)
        height, width = image.shape[:2]
        scale = max(0.5, min(width, height) / 720.0)
        thickness = max(1, int(round(2 * scale)))
        font = getattr(cv2, "FONT_HERSHEY_SIMPLEX")

        def _draw(text: str, origin: tuple[int, int], text_scale: float,
                  color: tuple[int, int, int] = (255, 255, 255)) -> None:
            (text_w, text_h), baseline = cv2.getTextSize(text, font, text_scale, thickness)
            x, y = origin
            # Semi-transparent dark plate behind the text so it stays legible on bright frames.
            plate = image.copy()
            cv2.rectangle(plate, (x - 4, y - text_h - 6), (x + text_w + 4, y + baseline + 2), (0, 0, 0), -1)
            cv2.addWeighted(plate, 0.55, image, 0.45, 0, image)
            cv2.putText(image, text, (x, y), font, text_scale, color, thickness, getattr(cv2, "LINE_AA"))

        # Per-box labels: identify each region (tracker / groundtruth) right at its bounding box,
        # in the same colour as the box, so stacked trackers are distinguishable.
        box_scale = max(0.4, scale * 0.7)
        for key, text in self._box_labels.items():
            regions = self._regions.get(key)
            if regions is None:
                continue
            bounds = _region_bounds(regions[frame])
            if bounds is None:
                continue
            x1, y1, _, _ = bounds
            (text_w, text_h), _ = cv2.getTextSize(text, font, box_scale, thickness)
            # Clamp the label into the frame; drop it below the box if it would overflow the top.
            label_x = min(max(0, x1), max(0, width - text_w - 8))
            label_y = y1 - 6
            if label_y - text_h < 0:
                label_y = min(y1 + text_h + 8, height - 4)
            _draw(text, (label_x, label_y), box_scale, self._box_label_color(key))

        if self._label:
            _draw(self._label, (12, 12 + int(28 * scale)), scale)

        row_y = 12 + int(28 * scale)
        counter = "{}/{}".format(frame + 1, len(self))
        (counter_w, counter_h), counter_base = cv2.getTextSize(counter, font, scale, thickness)
        _draw(counter, (width - counter_w - 12, row_y), scale)

        # Running failure / crash tallies, drawn under the frame counter (single-tracker
        # videos only). Each tally turns red once its count is non-zero. The y offsets
        # stack each row under the previous one with a constant gap so the text plates
        # don't overlap.
        tally_y = row_y
        if self._failure_frames is not None:
            failures = sum(1 for f in self._failure_frames if f <= frame)
            tally = "failures: {}".format(failures)
            (tally_w, _), _ = cv2.getTextSize(tally, font, scale, thickness)
            tally_y = tally_y + counter_base + counter_h + 14
            _draw(tally, (width - tally_w - 12, tally_y), scale,
                  (255, 90, 90) if failures else (255, 255, 255))
        if self._crash_frames is not None:
            crashes = sum(1 for f in self._crash_frames if f <= frame)
            tally = "crashes: {}".format(crashes)
            (tally_w, tally_h), tally_base = cv2.getTextSize(tally, font, scale, thickness)
            tally_y = tally_y + tally_base + tally_h + 14
            _draw(tally, (width - tally_w - 12, tally_y), scale,
                  (255, 90, 90) if crashes else (255, 255, 255))
        if self._lost_frames is not None:
            lost = sum(1 for f in self._lost_frames if f <= frame)
            tally = "lost: {}".format(lost)
            (tally_w, tally_h), tally_base = cv2.getTextSize(tally, font, scale, thickness)
            tally_y = tally_y + tally_base + tally_h + 14
            _draw(tally, (width - tally_w - 12, tally_y), scale,
                  (255, 90, 90) if lost else (255, 255, 255))

        # Bottom-left "cached" badge for realtime frames where no live tracker
        # invocation occurred (the runtime replayed the previous status). Drawn in a
        # muted amber so it's clearly an informational marker, not a fault.
        if self._cached_frames is not None and frame in self._cached_frames:
            badge_scale = max(0.4, scale * 0.7)
            _draw("cached", (12, height - 12), badge_scale, (220, 180, 60))
        return image

# Note: ``skvideo`` is imported lazily inside ``VideoWriterScikitH264`` to avoid
# the hard dependency at import time. Pyright doesn't have stubs for it; the
# runtime branch uses ``# type: ignore[import-untyped]``.

# Video writer classes that raised at runtime in this process. Once a backend fails and a
# fallback succeeds (e.g. the unmaintained scikit-video 1.1.x against numpy >= 2.0), it is
# not retried for the remaining videos: re-probing it would spawn a doomed ffmpeg
# subprocess and re-render a frame for every single video in the report.
_RUNTIME_BROKEN_WRITERS: "set[type]" = set()


class VideoWriter:
    """Abstract interface for writing a stream of rendered frames."""

    def __init__(self, filename: str, fps: int = 30) -> None:
        """Initialize writer target and frame rate."""
        self._filename = filename
        self._fps = fps

    def __call__(self, frame: npt.NDArray[np.uint8]) -> None:
        """Append a frame to the output stream."""
        raise NotImplementedError()

    def close(self) -> None:
        """Finalize and close underlying resources."""
        raise NotImplementedError()


class VideoWriterScikitH264(VideoWriter):
    """FFmpeg-backed H.264 writer implemented via scikit-video."""

    def __init__(self, filename: str, fps: int = 30) -> None:
        super().__init__(filename, fps)
        # ``Any`` because ``skvideo`` has no type stubs; we annotate explicitly so
        # the ``self._writer.close()`` / ``writeFrame(...)`` calls are not flagged
        # against ``None``.
        self._writer: Any = None

    def _handle(self) -> Any:
        """Create or return the underlying FFmpeg writer handle."""
        try:
            import skvideo.io  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("The scikit-video package is required for video export.")
        if self._writer is None:
            self._writer = skvideo.io.FFmpegWriter(
                self._filename,
                inputdict={'-r': str(self._fps), '-vcodec': 'libx264'},
            )
        return self._writer

    def __call__(self, frame: npt.NDArray[np.uint8]) -> None:
        """Write a single RGB frame to the H.264 stream."""
        self._handle().writeFrame(frame)

    def close(self) -> None:
        """Close the FFmpeg writer if it was initialized."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class VideoWriterOpenCV(VideoWriter):
    """Video writer that uses OpenCV codecs.

    For ``.mp4`` output the writer prefers the ``avc1`` (H.264) fourcc because the legacy ``mp4v``
    (MPEG-4 Part 2) codec is no longer accepted by mainstream browsers in ``<video>`` tags. If the
    OpenCV build does not ship libx264 the writer falls back to ``mp4v``; the resulting file is still
    valid but may not play in browsers.
    """

    DEFAULT_CODECS = {
        ".mp4": ("avc1", "mp4v"),
        ".m4v": ("avc1", "mp4v"),
        ".avi": ("xvid", "mjpg"),
        ".mov": ("avc1", "mp4v"),
    }

    def __init__(self, filename: str, fps: int = 30, codec: str | None = None) -> None:
        """Initialize OpenCV writer.

        :param filename: Path to the output file.
        :param fps: Frames per second of the output stream.
        :param codec: Explicit fourcc codec to use. When omitted, codec preference is inferred from the file extension via :data:`DEFAULT_CODECS` and a fallback list is tried in order until ``VideoWriter`` reports the file was opened.
        """
        super().__init__(filename, fps)
        if codec is not None:
            self._codecs: tuple[str, ...] = (codec,)
        else:
            ext = os.path.splitext(filename)[1].lower()
            self._codecs = self.DEFAULT_CODECS.get(ext, ("mp4v",))
        # ``Any`` because the bound writer comes from cv2 and its return type is
        # untyped in the cv2 stubs we have available.
        self._writer: Any = None
        self._width: int = 0
        self._height: int = 0

    def __call__(self, frame: npt.NDArray[np.uint8]) -> None:
        """Append one RGB frame to the OpenCV stream."""
        try:
            import cv2
        except ImportError:
            raise ImportError("The OpenCV package is required for video export.")
        if self._writer is None:
            self._height, self._width = frame.shape[:2]
            # ``VideoWriter_fourcc`` lives on the ``cv2`` C-extension and isn't
            # visible in the cv2 type stubs; ``getattr`` keeps pyright quiet.
            fourcc_factory = getattr(cv2, "VideoWriter_fourcc")
            last_codec = None
            for candidate in self._codecs:
                fourcc = fourcc_factory(*candidate.lower())
                writer = cv2.VideoWriter(self._filename, fourcc, self._fps, (self._width, self._height))
                last_codec = candidate
                if writer.isOpened():
                    self._writer = writer
                    break
                writer.release()
            if self._writer is None:
                raise RuntimeError(
                    "OpenCV could not open a VideoWriter for {} with any of {} "
                    "(last tried: {})".format(self._filename, ", ".join(self._codecs), last_codec)
                )
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        """Release the OpenCV writer if it was initialized."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class PreviewVideos(SeparableReport):
    """A report that generates video previews for the tracker results."""

    groundtruth = Boolean(default=False, description="If set, the groundtruth is shown with the tracker output.")
    separate = Boolean(default=False, description="If set, each tracker is shown in a separate video.")

    def _runs(self, experiment: Experiment, sequence: Sequence) -> list[_PreviewRun]:
        """Return one :class:`_PreviewRun` per preview video for the sequence.

        Multistart anchor runs overlap heavily — a forward run from frame 0 and a
        backward run from the last frame both span the whole sequence — so they cannot
        share a video without clobbering each other. Multistart therefore yields one
        run per anchor, each trimmed to the frames that run actually covers; every
        other experiment yields a single full-length run.
        """
        if not isinstance(experiment, MultiStartExperiment):
            return [_PreviewRun(suffix="", label=None, frames=None, index=None)]

        forward, backward = find_anchors(sequence, experiment.anchor)
        last = len(sequence) - 1
        runs: list[_PreviewRun] = []
        # Order must match MultiStartExperiment.gather (forward anchors, then backward)
        # so ``index`` selects the matching trajectory; ``len(runs)`` is the position
        # the run about to be appended will occupy.
        for anchor in forward:
            # A forward run is initialized at ``anchor`` and tracks to the end; the video
            # plays anchor -> end, the same order the tracker ran.
            runs.append(_PreviewRun(
                suffix="_anchor{:08d}_forward".format(anchor),
                label="forward from anchor {}".format(anchor),
                frames=tuple(range(anchor, last + 1)),
                index=len(runs),
            ))
        for anchor in backward:
            # A backward run is initialized at ``anchor`` and tracks back to frame 0; the
            # video plays anchor -> 0 (descending) so it is watched in the tracker's run
            # direction, as it actually unfolded, instead of time-reversed.
            runs.append(_PreviewRun(
                suffix="_anchor{:08d}_backward".format(anchor),
                label="backward from anchor {}".format(anchor),
                frames=tuple(range(anchor, -1, -1)),
                index=len(runs),
            ))
        return runs

    def _populate_video(
        self,
        video: "_LabeledObjectVideo",
        experiment: Experiment,
        trackers: list[Tracker],
        sequence: Sequence,
        run_index: int | None = None,
        abs_frames: list[int] | None = None,
        gather_cache: dict | None = None,
    ) -> None:
        """Draw trajectories for all requested trackers and objects into a video.

        When several trackers share one video (``separate=False``), each region is also
        labelled with its tracker name next to the bounding box so the boxes can be told
        apart; in ``separate=True`` mode the video title already identifies the tracker
        and a running count of its FAILURE frames is overlaid as a failure counter.

        ``run_index`` selects a single gathered trajectory (one multistart anchor run);
        ``None`` draws every gathered trajectory together (multirun repetitions).
        ``abs_frames`` maps each video frame to its absolute sequence index (a multistart
        anchor run covers only a sub-range, a backward run in descending order), while
        ``sequence`` and the gathered trajectories are always in absolute coordinates.

        ``gather_cache`` memoizes ``experiment.gather`` results. A multistart sequence
        yields one preview video per anchor and each would otherwise re-gather every
        anchor's trajectory (O(anchors^2) trajectory reads); the cache makes it O(anchors).
        """

        label_boxes = not self.separate
        frame_count = len(video)
        if abs_frames is None:
            abs_frames = list(range(frame_count))

        if self.groundtruth:
            # ``Sequence.groundtruth`` raises on multi-object sequences; iterate the objects
            # explicitly so multi-object stacks (e.g. ``tests/multiobject``) are supported.
            # Each object gets its own key so their groundtruth regions do not overwrite each
            # other in a shared frame.
            for obj in sequence.objects():
                drawn = False
                for frame in range(frame_count):
                    region = sequence.object(obj, abs_frames[frame])
                    if region is None or is_special(region):
                        continue
                    video(frame, "_groundtruth_" + obj, region)
                    drawn = True
                if drawn and label_boxes:
                    video.set_box_label("_groundtruth_" + obj, "groundtruth")

        # ``experiment.gather`` is defined on ``MultiRunExperiment`` and
        # ``MultiStartExperiment`` (the ``compatible`` check restricts this report to
        # those); narrow here so the ``.gather`` call is type-safe.
        if not isinstance(experiment, _VIDEO_EXPERIMENTS):
            raise TypeError(
                "PreviewVideos requires a MultiRunExperiment or MultiStartExperiment "
                f"but received {type(experiment).__name__}"
            )

        # In separate (single-tracker) mode, record the video frames where the tracker
        # emitted a FAILURE region (tracking-quality loss) or a CRASH region (process
        # exception / timeout) so the video can draw running tallies for each. With
        # several trackers stacked into one video the tallies would be ambiguous, so
        # they are left off there.
        failure_frames: set[int] = set()
        crash_frames: set[int] = set()
        lost_frames: set[int] = set()
        cached_frames: set[int] = set()
        is_realtime = getattr(experiment, "realtime", None) is not None

        for tracker in trackers:

            for obj in sequence.objects():
                # ``pad=True`` keeps the list aligned with the anchor order from
                # ``_runs`` so ``run_index`` selects the matching anchor run. The result
                # is identical for every anchor run of this (sequence, tracker, object),
                # so it is gathered once and reused via ``gather_cache``.
                cache_key = (sequence.name, tracker.identifier, obj)
                if gather_cache is not None and cache_key in gather_cache:
                    trajectories = gather_cache[cache_key]
                else:
                    trajectories = experiment.gather(tracker, sequence, objects=[obj], pad=True)
                    if gather_cache is not None:
                        gather_cache[cache_key] = trajectories

                if len(trajectories) == 0:
                    continue

                if run_index is None:
                    selected = [t for t in trajectories if t is not None]
                elif run_index < len(trajectories) and trajectories[run_index] is not None:
                    selected = [trajectories[run_index]]
                else:
                    selected = []

                key = obj + "_" + tracker.identifier
                drawn = False
                for trajectory in selected:
                    assert isinstance(trajectory, Trajectory)
                    for frame in range(frame_count):
                        abs_idx = abs_frames[frame]
                        region = trajectory.region(abs_idx)
                        is_cached = False
                        if is_realtime:
                            properties = trajectory.properties(abs_idx)
                            time_value = properties.get("time") if isinstance(properties, dict) else None
                            try:
                                elapsed = float(time_value) if time_value is not None else 0.0
                            except (TypeError, ValueError):
                                elapsed = 0.0
                            if elapsed <= 0:
                                is_cached = True
                                cached_frames.add(frame)
                        if is_special(region):
                            if is_special(region, SpecialCode.FAILURE):
                                failure_frames.add(frame)
                            elif is_special(region, SpecialCode.CRASH):
                                crash_frames.add(frame)
                            continue
                        # Skip cached frames: the corner badge already flags them, and
                        # any empty mask upstream was already counted as lost on the
                        # live frame that produced it.
                        if is_cached:
                            continue
                        if isinstance(region, Mask) and (region.mask.size == 0 or int(region.mask.sum()) == 0):
                            lost_frames.add(frame)
                            continue
                        video(frame, key, region)
                        drawn = True

                if drawn and label_boxes:
                    video.set_box_label(key, tracker.identifier)

        if self.separate and failure_frames:
            video.set_failures(sorted(failure_frames))
        if self.separate and crash_frames:
            video.set_crashes(sorted(crash_frames))
        if self.separate and lost_frames:
            video.set_losses(sorted(lost_frames))
        if self.separate and cached_frames:
            video.set_cached_frames(sorted(cached_frames))

    async def generate_video(
        self,
        experiment: Experiment,
        trackers: list[Tracker],
        sequence: Sequence,
        identifier: str | None = None,
        label: str | None = None,
        run_index: int | None = None,
        frames: tuple[int, ...] | None = None,
        image_cache: MutableMapping | None = None,
        gather_cache: dict | None = None,
    ) -> ObjectVideo:
        """Build one preview video for a sequence and a set of trackers.

        :param label: Text overlaid on every frame (top-left). Defaults to the tracker identifier
            when only one tracker is rendered, or the sequence name when multiple trackers are
            stacked into the same video.
        :param run_index: Selects a single multistart anchor run; ``None`` draws all gathered
            trajectories together.
        :param frames: Absolute sequence frame indices to render, in play order; ``None``
            renders the whole sequence in natural order.
        :param image_cache: Shared decoded-frame cache passed to the video.
        :param gather_cache: Shared cache of ``experiment.gather`` results.
        """
        if frames is None:
            abs_frames = list(range(len(sequence)))
        else:
            abs_frames = list(frames)
        # A FrameMapSequence over ``abs_frames`` plays exactly those source frames in the
        # given order -- ascending for a forward run, descending for a backward one.
        video_frames = FrameMapSequence(sequence, abs_frames)

        if label is None:
            label = trackers[0].identifier if len(trackers) == 1 else sequence.name
        # Play the preview at the sequence's real frame rate; otherwise the video runs at
        # the ``ObjectVideo`` default of 10 fps and a 30 fps sequence looks 3x slow.
        fps_meta = sequence.metadata("fps", 30) or 30
        fps = int(fps_meta) if isinstance(fps_meta, (int, float, str)) else 30
        video = _LabeledObjectVideo(identifier or sequence.identifier, video_frames, fps=fps,
                                    label=label, abs_frames=abs_frames, image_cache=image_cache,
                                    cache_key=sequence.name)
        self._populate_video(video, experiment, trackers, sequence,
                             run_index=run_index, abs_frames=abs_frames,
                             gather_cache=gather_cache)

        return video

    async def perexperiment(self, experiment: Experiment, trackers: list[Tracker], sequences: list[Sequence]):
        """Generate per-sequence videos, optionally split per tracker.

        For a multistart experiment each anchor run becomes its own video, trimmed to the
        frames that run covers, so the overlapping forward/backward runs are previewed as
        clean single-box clips.
        """
        from cachetools import LRUCache

        videos = []
        # Shared across every preview video of this run so the overlapping anchor-run
        # videos decode each source frame only once. Sized to the longest sequence so a
        # whole sequence's frames fit at once (a smaller bound makes each of a sequence's
        # many anchor-run videos re-decode every frame).
        cache_frames = max((len(sequence) for sequence in sequences), default=0)
        image_cache: MutableMapping = LRUCache(maxsize=max(cache_frames, _IMAGE_CACHE_FRAMES))
        # Memoizes experiment.gather results so a sequence's anchor-run videos load each
        # tracker's trajectories once instead of once per anchor (O(anchors) not O(anchors^2)).
        gather_cache: dict = {}

        for sequence in sequences:

            for run in self._runs(experiment, sequence):

                # Namespace each preview under the experiment so a stack with
                # multiple compatible experiments (e.g. ``baseline`` and
                # ``unsupervised`` in vot2016) does not have the second pass
                # overwrite the first: both produced ``<sequence>.mp4`` before
                # and the HTML referenced the shared filename from each
                # experiment's section. ``LocalStorage.write`` makedirs the
                # parent, so a ``/`` in the identifier lands the file in a
                # per-experiment subfolder.
                exp_prefix = experiment.identifier + "/"
                if self.separate:
                    for tracker in trackers:
                        video = await self.generate_video(
                            experiment,
                            [tracker],
                            sequence,
                            identifier="{}{}{}_{}".format(exp_prefix, sequence.identifier, run.suffix, tracker.identifier),
                            label=tracker.identifier if run.label is None
                                else "{} - {}".format(tracker.identifier, run.label),
                            run_index=run.index,
                            frames=run.frames,
                            image_cache=image_cache,
                            gather_cache=gather_cache,
                        )
                        videos.append(video)
                else:
                    video = await self.generate_video(
                        experiment,
                        trackers,
                        sequence,
                        identifier=exp_prefix + sequence.identifier + run.suffix,
                        label=None if run.label is None
                            else "{} - {}".format(sequence.name, run.label),
                        run_index=run.index,
                        frames=run.frames,
                        image_cache=image_cache,
                        gather_cache=gather_cache,
                    )
                    videos.append(video)

        return videos

    def compatible(self, experiment):
        """Restrict this report element to experiments that expose gathered trajectories."""
        return isinstance(experiment, _VIDEO_EXPERIMENTS)