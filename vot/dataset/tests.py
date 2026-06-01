"""Unit tests for the dataset layout, preparation, statistics and common modules.

These cover the consolidation of the former ``vot.utilities.sequences`` module into the
``vot.dataset`` package: the on-disk layout primitives, the ``list.txt`` index, the
generalized (mask-agnostic, multi-channel) sequence transforms and sequence I/O.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2


def setUpModule() -> None:
    """Silences expected library log noise so the test output stays readable.

    Several tests deliberately exercise error/warning paths (a missing image, a missing
    sequence directory, an annotation-less import); the resulting ``vot`` logger and OpenCV
    messages are expected and would otherwise clutter the test report.
    """
    logging.getLogger("vot").setLevel(logging.CRITICAL)
    try:
        cv2.setLogLevel(0)  # type: ignore[attr-defined]  # OpenCV LOG_LEVEL_SILENT, absent from cv2 stub
    except Exception:
        pass

from vot.dataset import load_sequence
from vot.dataset import layout, common
from vot.dataset.layout import (DEFAULT_FRAME_MASK, GROUNDTRUTH_FILE, LIST_FILE, METADATA_FILE,
                                SequenceList, channel_keys, frame_filename, image_size,
                                list_image_files, read_metadata, write_metadata)
from vot.dataset import statistics as stats
from vot.dataset import preparation as prep
from vot.region.shapes import Rectangle
from vot.utilities import write_properties

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _write_frame(path: Path, width: int, height: int, seed: int) -> None:
    """Writes a deterministic-but-distinct image at ``path`` (extension picks the codec)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _make_sequence(root: Path, length: int = 8, channels: dict | None = None,
                   objects: int = 1, declare_length: bool = True,
                   width: int = 16, height: int = 12, name: str = "synthetic") -> Path:
    """Builds a minimal on-disk VOT sequence for tests.

    :param root: Sequence directory to create.
    :param length: Number of frames.
    :param channels: Mapping channel name -> relative frame pattern. Defaults to a single
        ``color`` channel with the canonical mask.
    :param objects: Number of groundtruth objects (1 -> ``groundtruth.txt``, else per-object files).
    :param declare_length: Whether to record ``length``/``width``/``height`` in the metadata.
    :param name: Value for the metadata ``name`` field.

    :returns: The created sequence directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if channels is None:
        channels = {"color": "color/" + DEFAULT_FRAME_MASK}

    for index in range(1, length + 1):
        for pattern in channels.values():
            _write_frame(root / (pattern % index), width, height, seed=index)

    # Groundtruth: a 4x4 box moving diagonally so frame i is uniquely identifiable.
    gt_lines = ["{},{},4,4".format(i, i) for i in range(length)]
    if objects == 1:
        (root / GROUNDTRUTH_FILE).write_text("\n".join(gt_lines) + "\n")
    else:
        for obj in range(objects):
            (root / "groundtruth_{:03d}.txt".format(obj)).write_text("\n".join(gt_lines) + "\n")

    metadata: dict = {"name": name, "fps": 30, "format": "default", "channel.default": "color"}
    for channel_name, pattern in channels.items():
        metadata["channels." + channel_name] = pattern
    if declare_length:
        metadata["length"] = length
        metadata["width"] = width
        metadata["height"] = height
    write_properties(str(root / METADATA_FILE), metadata)
    return root


def _gt_origins(sequence_dir: Path) -> list[tuple[int, int]]:
    """Returns the ``(x, y)`` origin of every groundtruth rectangle of a sequence."""
    rectangles = stats._read_rectangles(sequence_dir)
    return [(int(r.x), int(r.y)) for r in rectangles]


class _TempCase(unittest.TestCase):
    """Base test case providing a fresh temporary directory per test."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vot_dataset_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestLayoutConstants(_TempCase):
    """Tests for the layout constants and stateless helpers."""

    def test_constants(self) -> None:
        self.assertEqual(DEFAULT_FRAME_MASK, "%08d.jpg")
        self.assertEqual(METADATA_FILE, "sequence")
        self.assertEqual(GROUNDTRUTH_FILE, "groundtruth.txt")
        self.assertEqual(LIST_FILE, "list.txt")

    def test_frame_filename(self) -> None:
        self.assertEqual(frame_filename(1), "00000001.jpg")
        self.assertEqual(frame_filename(123), "00000123.jpg")

    def test_list_image_files_sorted(self) -> None:
        for index in (3, 1, 2):
            _write_frame(self.tmp / "{:08d}.jpg".format(index), 8, 8, seed=index)
        _write_frame(self.tmp / "00000004.png", 8, 8, seed=4)
        files = list_image_files(self.tmp)
        self.assertEqual([p.name for p in files],
                         ["00000001.jpg", "00000002.jpg", "00000003.jpg", "00000004.png"])

    def test_list_image_files_empty(self) -> None:
        self.assertEqual(list_image_files(self.tmp), [])

    def test_image_size(self) -> None:
        _write_frame(self.tmp / "frame.jpg", 24, 18, seed=1)
        self.assertEqual(image_size(self.tmp / "frame.jpg"), (24, 18))

    def test_image_size_missing(self) -> None:
        self.assertIsNone(image_size(self.tmp / "missing.jpg"))

    def test_channel_keys(self) -> None:
        metadata = {"channels.color": "color/%08d.jpg", "channels.ir": "ir/%08d.png",
                    "fps": 30, "name": "x"}
        self.assertEqual(channel_keys(metadata), ["color", "ir"])

    def test_channel_keys_empty(self) -> None:
        self.assertEqual(channel_keys({"fps": 30}), [])


class TestReadWriteMetadata(_TempCase):
    """Tests for layout.read_metadata / write_metadata."""

    def test_read_metadata_coerce_and_defaults(self) -> None:
        write_properties(str(self.tmp / METADATA_FILE),
                         {"fps": "25", "length": "10", "width": "640", "height": "480"})
        metadata = read_metadata(self.tmp)
        self.assertEqual(metadata["fps"], 25)
        self.assertEqual(metadata["length"], 10)
        self.assertIsInstance(metadata["length"], int)
        self.assertEqual(metadata["format"], "default")
        self.assertEqual(metadata["channel.default"], "color")
        self.assertEqual(metadata["root"], str(self.tmp))

    def test_read_metadata_raw(self) -> None:
        write_properties(str(self.tmp / METADATA_FILE), {"length": "7"})
        metadata = read_metadata(self.tmp, coerce=False, defaults=False)
        self.assertEqual(metadata["length"], "7")
        self.assertNotIn("root", metadata)
        self.assertNotIn("format", metadata)

    def test_read_metadata_fps_default(self) -> None:
        write_properties(str(self.tmp / METADATA_FILE), {"name": "x"})
        self.assertEqual(read_metadata(self.tmp)["fps"], 30)

    def test_read_metadata_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_metadata(self.tmp)

    def test_write_metadata_roundtrip(self) -> None:
        write_metadata(self.tmp, {"name": "demo", "fps": 30, "length": 5})
        self.assertTrue((self.tmp / METADATA_FILE).is_file())
        metadata = read_metadata(self.tmp, coerce=False, defaults=False)
        self.assertEqual(metadata["name"], "demo")
        self.assertEqual(metadata["length"], "5")


class TestSequenceList(_TempCase):
    """Tests for the layout.SequenceList list.txt index manager."""

    def test_read_absent(self) -> None:
        self.assertIsNone(SequenceList(self.tmp).read())
        self.assertFalse(SequenceList(self.tmp).exists())

    def test_write_and_read(self) -> None:
        SequenceList(self.tmp).write(["alpha", "beta", "gamma"])
        self.assertTrue(SequenceList(self.tmp).exists())
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha", "beta", "gamma"])

    def test_write_skips_blank_lines_on_read(self) -> None:
        (self.tmp / LIST_FILE).write_text("alpha\n\n  \nbeta\n")
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha", "beta"])

    def test_append_creates_file(self) -> None:
        self.assertTrue(SequenceList(self.tmp).append("alpha"))
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha"])

    def test_append_deduplicates(self) -> None:
        SequenceList(self.tmp).append("alpha")
        self.assertFalse(SequenceList(self.tmp).append("alpha"))
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha"])

    def test_append_fixes_missing_trailing_newline(self) -> None:
        (self.tmp / LIST_FILE).write_text("alpha")  # no trailing newline
        SequenceList(self.tmp).append("beta")
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha", "beta"])

    def test_remove_existing(self) -> None:
        SequenceList(self.tmp).write(["alpha", "beta", "gamma"])
        self.assertTrue(SequenceList(self.tmp).remove("beta"))
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha", "gamma"])

    def test_remove_absent_entry(self) -> None:
        SequenceList(self.tmp).write(["alpha"])
        self.assertFalse(SequenceList(self.tmp).remove("missing"))
        self.assertEqual(SequenceList(self.tmp).read(), ["alpha"])

    def test_remove_no_file(self) -> None:
        self.assertFalse(SequenceList(self.tmp).remove("alpha"))

    def test_remove_last_entry_leaves_empty_file(self) -> None:
        SequenceList(self.tmp).write(["alpha"])
        self.assertTrue(SequenceList(self.tmp).remove("alpha"))
        self.assertEqual(SequenceList(self.tmp).read(), [])


class TestCommonReadWrite(_TempCase):
    """Tests for vot.dataset.common: read_sequence, read_sequence_legacy, list_sequences, write_sequence."""

    def test_read_sequence(self) -> None:
        _make_sequence(self.tmp / "seq", length=6)
        sequence = common.read_sequence(self.tmp / "seq")
        self.assertIsNotNone(sequence)
        assert sequence is not None
        self.assertEqual(len(sequence), 6)
        self.assertEqual(list(sequence.channels()), ["color"])

    def test_read_sequence_missing_metadata(self) -> None:
        self.assertIsNone(common.read_sequence(self.tmp))

    def test_read_sequence_legacy(self) -> None:
        seq_dir = _make_sequence(self.tmp / "seq", length=5)
        (seq_dir / METADATA_FILE).unlink()
        # Move frames to the root so the legacy default mask resolves them.
        for index in range(1, 6):
            (seq_dir / "color" / frame_filename(index)).rename(seq_dir / frame_filename(index))
        sequence = common.read_sequence_legacy(seq_dir)
        self.assertIsNotNone(sequence)
        assert sequence is not None
        self.assertEqual(len(sequence), 5)

    def test_read_sequence_legacy_missing_groundtruth(self) -> None:
        self.assertIsNone(common.read_sequence_legacy(self.tmp))

    def test_list_sequences_from_listfile_directory(self) -> None:
        SequenceList(self.tmp).write(["one", "two"])
        self.assertEqual(common.list_sequences(self.tmp), ["one", "two"])

    def test_list_sequences_path_is_file(self) -> None:
        list_path = self.tmp / "mylist.txt"
        list_path.write_text("one\ntwo\nthree\n")
        self.assertEqual(common.list_sequences(list_path), ["one", "two", "three"])

    def test_list_sequences_none(self) -> None:
        self.assertIsNone(common.list_sequences(self.tmp / "does_not_exist"))

    def test_write_sequence_roundtrip(self) -> None:
        _make_sequence(self.tmp / "src", length=5)
        sequence = load_sequence(str(self.tmp / "src"))
        common.write_sequence(self.tmp / "dst", sequence)
        reloaded = load_sequence(str(self.tmp / "dst"))
        self.assertEqual(len(reloaded), 5)
        self.assertEqual(list(reloaded.channels()), ["color"])
        self.assertTrue((self.tmp / "dst" / METADATA_FILE).is_file())

    def test_write_sequence_copies_jpeg_bytes(self) -> None:
        src = _make_sequence(self.tmp / "src", length=4)
        sequence = load_sequence(str(src))
        common.write_sequence(self.tmp / "dst", sequence)
        original = (src / "color" / frame_filename(1)).read_bytes()
        written = (self.tmp / "dst" / "color" / frame_filename(1)).read_bytes()
        self.assertEqual(original, written, "file-backed JPEG frames must be copied byte-for-byte")

    def test_write_sequence_reencodes_non_jpeg(self) -> None:
        _make_sequence(self.tmp / "src", length=3, channels={"color": "color/%08d.png"})
        sequence = load_sequence(str(self.tmp / "src"))
        common.write_sequence(self.tmp / "dst", sequence)
        written = self.tmp / "dst" / "color" / frame_filename(1)
        self.assertTrue(written.is_file())
        self.assertIsNotNone(cv2.imread(str(written)))

    def test_write_sequence_preserves_metadata(self) -> None:
        _make_sequence(self.tmp / "src", length=4, name="keepme")
        sequence = load_sequence(str(self.tmp / "src"))
        common.write_sequence(self.tmp / "dst", sequence)
        metadata = read_metadata(self.tmp / "dst", coerce=False, defaults=False)
        self.assertEqual(metadata.get("name"), "keepme")
        self.assertEqual(metadata["length"], "4")

    def test_write_sequence_multiobject(self) -> None:
        _make_sequence(self.tmp / "src", length=4, objects=2)
        sequence = load_sequence(str(self.tmp / "src"))
        common.write_sequence(self.tmp / "dst", sequence)
        gt_files = sorted(p.name for p in (self.tmp / "dst").glob("groundtruth_*.txt"))
        self.assertEqual(len(gt_files), 2)


class TestDiscovery(_TempCase):
    """Tests for preparation.discover_sequences / collect_sequence_info."""

    def test_discover_from_listfile(self) -> None:
        _make_sequence(self.tmp / "a")
        _make_sequence(self.tmp / "b")
        SequenceList(self.tmp).write(["b", "a"])
        self.assertEqual(prep.discover_sequences(self.tmp), ["b", "a"])

    def test_discover_by_directory_scan(self) -> None:
        _make_sequence(self.tmp / "a")
        _make_sequence(self.tmp / "b")
        (self.tmp / "not_a_sequence").mkdir()
        self.assertEqual(prep.discover_sequences(self.tmp), ["a", "b"])

    def test_discover_empty(self) -> None:
        self.assertEqual(prep.discover_sequences(self.tmp / "missing"), [])

    def test_collect_sequence_info(self) -> None:
        _make_sequence(self.tmp / "a", length=7,
                       channels={"color": "color/%08d.jpg", "ir": "ir/%08d.jpg"})
        _make_sequence(self.tmp / "b", length=3)
        (self.tmp / "b" / GROUNDTRUTH_FILE).unlink()
        SequenceList(self.tmp).write(["a", "b"])

        infos = {info.name: info for info in prep.collect_sequence_info(self.tmp)}
        self.assertEqual(set(infos), {"a", "b"})
        self.assertEqual(infos["a"].channels, ["color", "ir"])
        self.assertEqual(infos["a"].length, 7)
        self.assertIsNone(infos["a"].inferred_length)
        self.assertTrue(infos["a"].has_groundtruth)
        self.assertTrue(infos["a"].present)
        self.assertFalse(infos["b"].has_groundtruth)

    def test_collect_sequence_info_inferred_length(self) -> None:
        # Metadata omits 'length'; the count is inferred from groundtruth.txt.
        _make_sequence(self.tmp / "a", length=5, declare_length=False)
        # No groundtruth either: the count falls back to channel image files.
        _make_sequence(self.tmp / "b", length=4, declare_length=False,
                       channels={"color": "color/%08d.jpg", "ir": "ir/%08d.jpg"})
        (self.tmp / "b" / GROUNDTRUTH_FILE).unlink()
        SequenceList(self.tmp).write(["a", "b"])

        infos = {info.name: info for info in prep.collect_sequence_info(self.tmp)}
        self.assertIsNone(infos["a"].length)
        self.assertEqual(infos["a"].inferred_length, 5)
        self.assertIsNone(infos["b"].length)
        self.assertEqual(infos["b"].inferred_length, 4)

    def test_collect_sequence_info_missing_directory(self) -> None:
        SequenceList(self.tmp).write(["ghost"])
        info = prep.collect_sequence_info(self.tmp)[0]
        self.assertEqual(info.name, "ghost")
        self.assertFalse(info.present)
        self.assertEqual(info.channels, [])


class TestMetadataAndAnchors(_TempCase):
    """Tests for write_sequence_metadata, write_anchor_file and generate_anchors_for_sequence."""

    def test_write_sequence_metadata_infers(self) -> None:
        # write_sequence_metadata defaults to the canonical color/ channel layout.
        for index in range(1, 6):
            _write_frame(self.tmp / "color" / frame_filename(index), 20, 14, seed=index)
        self.assertTrue(prep.write_sequence_metadata(self.tmp))
        metadata = read_metadata(self.tmp)
        self.assertEqual(metadata["length"], 5)
        self.assertEqual(metadata["width"], 20)
        self.assertEqual(metadata["height"], 14)

    def test_write_sequence_metadata_skips_existing(self) -> None:
        for index in range(1, 4):
            _write_frame(self.tmp / "color" / frame_filename(index), 10, 10, seed=index)
        self.assertTrue(prep.write_sequence_metadata(self.tmp))
        self.assertFalse(prep.write_sequence_metadata(self.tmp))
        self.assertTrue(prep.write_sequence_metadata(self.tmp, force=True))

    def test_write_anchor_file(self) -> None:
        path = self.tmp / "anchor.value"
        prep.write_anchor_file(path, 5)
        self.assertEqual(path.read_text().split(), ["1.0", "0.0", "0.0", "0.0", "-1.0"])

    def test_write_anchor_file_single_frame(self) -> None:
        path = self.tmp / "anchor.value"
        prep.write_anchor_file(path, 1)
        self.assertEqual(path.read_text().split(), ["1.0"])

    def test_write_anchor_file_invalid(self) -> None:
        with self.assertRaises(ValueError):
            prep.write_anchor_file(self.tmp / "anchor.value", 0)

    def test_generate_anchors_for_sequence(self) -> None:
        seq_dir = _make_sequence(self.tmp / "seq", length=6)
        self.assertTrue(prep.generate_anchors_for_sequence(seq_dir))
        anchor = (seq_dir / "anchor.value").read_text().split()
        self.assertEqual(len(anchor), 6)
        self.assertEqual(anchor[0], "1.0")
        self.assertEqual(anchor[-1], "-1.0")
        self.assertFalse(prep.generate_anchors_for_sequence(seq_dir))


class TestTransforms(_TempCase):
    """Tests for the generalized frame transforms (slice / subsample / reverse / delayed-init)."""

    def test_take_slice(self) -> None:
        _make_sequence(self.tmp / "src", length=10)
        prep.take_slice(self.tmp / "src", 3, 7, self.tmp / "dst")
        self.assertEqual(len(load_sequence(str(self.tmp / "dst"))), 5)
        # Output frame 0 corresponds to source frame index 2 (1-based frame 3).
        self.assertEqual(_gt_origins(self.tmp / "dst")[0], (2, 2))

    def test_take_slice_invalid_bounds(self) -> None:
        _make_sequence(self.tmp / "src", length=10)
        with self.assertRaises(ValueError):
            prep.take_slice(self.tmp / "src", 0, 5, self.tmp / "dst")
        with self.assertRaises(ValueError):
            prep.take_slice(self.tmp / "src", 1, 99, self.tmp / "dst")
        with self.assertRaises(ValueError):
            prep.take_slice(self.tmp / "src", 8, 3, self.tmp / "dst")

    def test_take_slice_custom_mask_multichannel(self) -> None:
        # Mimics real datasets like ``afterrain``: per-channel masks, channel subdirs, no length.
        _make_sequence(self.tmp / "src", length=8, declare_length=False,
                       channels={"color": "color/%05dv.jpg", "ir": "ir/%05di.jpg"})
        prep.take_slice(self.tmp / "src", 2, 5, self.tmp / "dst")
        result = load_sequence(str(self.tmp / "dst"))
        self.assertEqual(len(result), 4)
        self.assertEqual(sorted(result.channels()), ["color", "ir"])
        # Output is written with the canonical mask in channel subdirectories.
        self.assertTrue((self.tmp / "dst" / "color" / frame_filename(1)).is_file())
        self.assertTrue((self.tmp / "dst" / "ir" / frame_filename(1)).is_file())
        self.assertEqual(_gt_origins(self.tmp / "dst")[0], (1, 1))

    def test_subsample_sequence(self) -> None:
        _make_sequence(self.tmp / "src", length=10)
        prep.subsample_sequence(self.tmp / "src", 3, self.tmp / "dst")
        origins = _gt_origins(self.tmp / "dst")
        self.assertEqual(origins, [(0, 0), (3, 3), (6, 6), (9, 9)])

    def test_subsample_invalid_step(self) -> None:
        _make_sequence(self.tmp / "src", length=5)
        with self.assertRaises(ValueError):
            prep.subsample_sequence(self.tmp / "src", 0, self.tmp / "dst")

    def test_reverse_sequence(self) -> None:
        _make_sequence(self.tmp / "src", length=6)
        prep.reverse_sequence(self.tmp / "src", self.tmp / "dst")
        origins = _gt_origins(self.tmp / "dst")
        self.assertEqual(origins, [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1), (0, 0)])

    def test_reverse_sequence_multichannel(self) -> None:
        _make_sequence(self.tmp / "src", length=5,
                       channels={"color": "color/%08d.jpg", "ir": "ir/%08d.jpg"})
        prep.reverse_sequence(self.tmp / "src", self.tmp / "dst")
        result = load_sequence(str(self.tmp / "dst"))
        self.assertEqual(len(result), 5)
        self.assertEqual(sorted(result.channels()), ["color", "ir"])

    def test_delayed_init_variants(self) -> None:
        _make_sequence(self.tmp / "src", length=12)
        created = prep.delayed_init_variants(self.tmp / "src", count=3, repetitions=2,
                                             output_base=self.tmp)
        self.assertEqual(len(created), 2)
        # Variant k strips k*count leading frames.
        self.assertEqual(len(load_sequence(str(created[0]))), 9)
        self.assertEqual(len(load_sequence(str(created[1]))), 6)
        self.assertEqual(_gt_origins(created[0])[0], (3, 3))
        self.assertEqual(_gt_origins(created[1])[0], (6, 6))

    def test_delayed_init_existing_directory(self) -> None:
        _make_sequence(self.tmp / "src", length=8)
        (self.tmp / "src_1_2").mkdir()
        with self.assertRaises(FileExistsError):
            prep.delayed_init_variants(self.tmp / "src", count=2, repetitions=1,
                                       output_base=self.tmp)

    def test_slice_preserves_tags_and_values(self) -> None:
        seq_dir = _make_sequence(self.tmp / "src", length=10)
        # A per-frame tag file and a per-frame value file.
        (seq_dir / "occlusion.tag").write_text("\n".join("1" if i % 2 else "0" for i in range(10)))
        (seq_dir / "anchor.value").write_text("\n".join(str(float(i)) for i in range(10)))
        prep.take_slice(seq_dir, 2, 6, self.tmp / "dst")
        result = load_sequence(str(self.tmp / "dst"))
        self.assertIn("occlusion", result.tags())
        self.assertIn("anchor", result.values())


class TestListRegistration(_TempCase):
    """Top-level sequence-producing helpers append their outputs to ``list.txt``.

    The two building blocks ``take_slice`` and ``subsample_sequence`` are reused
    internally by higher-level helpers with temp-dir paths, so they intentionally
    stay side-effect-free; the higher-level helpers register on their behalf.
    """

    def test_reverse_sequence_registers_output(self) -> None:
        _make_sequence(self.tmp / "src", length=4)
        prep.reverse_sequence(self.tmp / "src", self.tmp / "dst")
        self.assertEqual(SequenceList(self.tmp).read(), ["dst"])

    def test_delayed_init_variants_register_each_variant(self) -> None:
        _make_sequence(self.tmp / "src", length=12)
        created = prep.delayed_init_variants(self.tmp / "src", count=3, repetitions=2,
                                             output_base=self.tmp)
        self.assertEqual(SequenceList(self.tmp).read(),
                         [p.name for p in created])

    def test_create_baseline_slice_registers_output(self) -> None:
        _make_sequence(self.tmp / "src", length=10)
        prep.create_baseline_slice(self.tmp / "src", self.tmp / "dst",
                                   start_frame=2, end_frame=6)
        self.assertEqual(SequenceList(self.tmp).read(), ["dst"])

    def test_create_speedup_sequence_registers_output(self) -> None:
        _make_sequence(self.tmp / "src", length=12)
        # speedup_factor != 1 exercises the take_slice -> subsample_sequence path
        # with an internal ``.tmp_*`` directory that must NOT be registered.
        prep.create_speedup_sequence(self.tmp / "src", speedup_factor=3,
                                     output_dir=self.tmp / "fast")
        entries = SequenceList(self.tmp).read()
        self.assertEqual(entries, ["fast"])

    def test_create_speedup_experiments_registers_each_variant(self) -> None:
        _make_sequence(self.tmp / "src", length=20)
        prep.create_speedup_experiments(self.tmp / "src", self.tmp / "out",
                                        speedup_factors=(2, 4))
        # Variants are written under ``out/``, so ``list.txt`` lives next to them.
        entries = SequenceList(self.tmp / "out").read()
        assert entries is not None
        self.assertEqual(sorted(entries),
                         sorted(["src_speedup_2x_10f", "src_speedup_4x_5f"]))

    def test_take_slice_does_not_register(self) -> None:
        """``take_slice`` is a building block; registration happens at the call site."""
        _make_sequence(self.tmp / "src", length=6)
        prep.take_slice(self.tmp / "src", 2, 5, self.tmp / "dst")
        self.assertIsNone(SequenceList(self.tmp).read())

    def test_subsample_sequence_does_not_register(self) -> None:
        """``subsample_sequence`` is a building block; registration happens at the call site."""
        _make_sequence(self.tmp / "src", length=6)
        prep.subsample_sequence(self.tmp / "src", 2, self.tmp / "dst")
        self.assertIsNone(SequenceList(self.tmp).read())

    def test_baseline_slice_is_idempotent_on_relist(self) -> None:
        """Re-running with the same destination must not duplicate the list entry."""
        _make_sequence(self.tmp / "src", length=8)
        prep.create_baseline_slice(self.tmp / "src", self.tmp / "dst",
                                   start_frame=0, end_frame=3)
        prep.create_baseline_slice(self.tmp / "src", self.tmp / "dst",
                                   start_frame=0, end_frame=3)
        self.assertEqual(SequenceList(self.tmp).read(), ["dst"])


class TestConcatenate(_TempCase):
    """Appending one sequence after another via ``_concatenate_sequence``.

    Appended frames must follow the *target* channel's own directory and filename mask, not
    the canonical ``<channel>/%08d.jpg``; otherwise a custom-mask sequence (e.g. ``%05dv.jpg``)
    gains frames the loader cannot find.
    """

    def test_concatenate_end_to_end(self) -> None:
        _make_sequence(self.tmp / "src", length=5)
        _make_sequence(self.tmp / "tail", length=4)
        prep.create_baseline_slice(self.tmp / "src", self.tmp / "dst",
                                   start_frame=0, end_frame=2,
                                   concatenate_path=self.tmp / "tail")
        result = load_sequence(str(self.tmp / "dst"))
        self.assertEqual(len(result), 7)  # 3-frame slice + 4-frame tail
        self.assertTrue(all(frame.image("color") is not None for frame in result))

    def test_concatenate_preserves_custom_mask(self) -> None:
        _make_sequence(self.tmp / "tgt", length=3, channels={"color": "color/%05dv.jpg"})
        _make_sequence(self.tmp / "tail", length=2)
        prep._concatenate_sequence(self.tmp / "tgt", self.tmp / "tail")

        # Appended frames keep the target's mask (``00004v.jpg``), not the canonical default.
        self.assertTrue((self.tmp / "tgt" / "color" / "00004v.jpg").is_file())
        self.assertTrue((self.tmp / "tgt" / "color" / "00005v.jpg").is_file())
        self.assertFalse((self.tmp / "tgt" / "color" / frame_filename(4)).exists())

        result = load_sequence(str(self.tmp / "tgt"))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(frame.image("color") is not None for frame in result))
        # Groundtruth is appended too: the tail's origins follow the target's.
        self.assertEqual(_gt_origins(self.tmp / "tgt"),
                         [(0, 0), (1, 1), (2, 2), (0, 0), (1, 1)])

    def test_concatenate_preserves_flat_layout(self) -> None:
        # Frames stored directly in the sequence root (no channel subdirectory).
        _make_sequence(self.tmp / "tgt", length=3, channels={"color": "%03d.jpg"})
        _make_sequence(self.tmp / "tail", length=2)
        prep._concatenate_sequence(self.tmp / "tgt", self.tmp / "tail")

        self.assertTrue((self.tmp / "tgt" / "004.jpg").is_file())
        self.assertFalse((self.tmp / "tgt" / "color").exists())

        result = load_sequence(str(self.tmp / "tgt"))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(frame.image("color") is not None for frame in result))


class TestRemoveSequence(_TempCase):
    """Tests for preparation.remove_sequence."""

    def test_remove_sequence(self) -> None:
        _make_sequence(self.tmp / "victim")
        SequenceList(self.tmp).write(["victim", "survivor"])
        self.assertTrue(prep.remove_sequence(self.tmp, "victim"))
        self.assertFalse((self.tmp / "victim").exists())
        self.assertEqual(SequenceList(self.tmp).read(), ["survivor"])

    def test_remove_sequence_listentry_only(self) -> None:
        SequenceList(self.tmp).write(["ghost"])
        self.assertTrue(prep.remove_sequence(self.tmp, "ghost"))
        self.assertEqual(SequenceList(self.tmp).read(), [])

    def test_remove_sequence_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            prep.remove_sequence(self.tmp, "nonexistent")


class TestStatistics(_TempCase):
    """Tests for the statistics module."""

    def test_read_rectangles(self) -> None:
        _make_sequence(self.tmp / "seq", length=5)
        rectangles = stats._read_rectangles(self.tmp / "seq")
        self.assertEqual(len(rectangles), 5)
        self.assertIsInstance(rectangles[0], Rectangle)

    def test_read_rectangles_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            stats._read_rectangles(self.tmp)

    def test_compute_movement_stats(self) -> None:
        _make_sequence(self.tmp / "seq", length=8)
        movement = stats.compute_movement_stats(self.tmp / "seq")
        self.assertEqual(len(movement.time_seconds), 8)
        self.assertEqual(len(movement.velocity_mag), 8)
        # Diagonal motion of 1px/frame at 30fps -> positive velocity.
        self.assertGreater(movement.summary()["avg_velocity"], 0.0)

    def test_compute_movement_stats_frame_range(self) -> None:
        _make_sequence(self.tmp / "seq", length=10)
        movement = stats.compute_movement_stats(self.tmp / "seq", frame_range=(2, 5))
        self.assertEqual(len(movement.time_seconds), 4)

    def test_movement_stats_summary_keys(self) -> None:
        _make_sequence(self.tmp / "seq", length=6)
        summary = stats.compute_movement_stats(self.tmp / "seq").summary()
        for key in ("duration_sec", "frames", "avg_velocity", "max_velocity",
                    "avg_area", "area_variance"):
            self.assertIn(key, summary)
        self.assertEqual(summary["frames"], 6)

    def test_find_size_range_windows(self) -> None:
        _make_sequence(self.tmp / "seq", length=20)
        windows = stats.find_size_range_windows(self.tmp / "seq", 0.0, 1000.0,
                                                target_frames=5, stride=5)
        self.assertTrue(windows)
        for window in windows:
            self.assertEqual(window["end_frame"] - window["start_frame"] + 1, 5)

    def test_verify_slice(self) -> None:
        _make_sequence(self.tmp / "seq", length=8)
        report = stats.verify_slice(self.tmp / "seq", 0.0, 1000.0, check_initial_size_only=False)
        self.assertEqual(report["length"], 8)
        self.assertTrue(report["size_in_range"])


class TestImporters(_TempCase):
    """Tests for YOLO / video / XML importers."""

    def test_yolo_line_to_xywh(self) -> None:
        box = prep._yolo_line_to_xywh("0 0.5 0.5 0.2 0.4", 100, 100)
        self.assertIsNotNone(box)
        assert box is not None
        x, y, w, h = box
        self.assertAlmostEqual(x, 40.0)
        self.assertAlmostEqual(y, 30.0)
        self.assertAlmostEqual(w, 20.0)
        self.assertAlmostEqual(h, 40.0)

    def test_yolo_line_to_xywh_malformed(self) -> None:
        self.assertIsNone(prep._yolo_line_to_xywh("0 0.5", 100, 100))

    def _make_yolo_source(self, source: Path, count: int = 3,
                          width: int = 32, height: int = 24) -> Path:
        """Builds a YOLO-format source directory of ``count`` ``.png`` + ``.txt`` pairs.

        Every label is a centered box spanning a quarter of the frame in each dimension."""
        source.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            _write_frame(source / "frame_{}.png".format(index), width, height, seed=index)
            (source / "frame_{}.txt".format(index)).write_text("0 0.5 0.5 0.25 0.25\n")
        return source

    def test_yolo_to_vot(self) -> None:
        source = self._make_yolo_source(self.tmp / "yolo", count=3)
        count = prep.yolo_to_vot(source, self.tmp / "out")
        self.assertEqual(count, 3)
        sequence = load_sequence(str(self.tmp / "out"))
        self.assertEqual(len(sequence), 3)
        self.assertEqual(list(sequence.channels()), ["color"])

    def test_yolo_to_vot_frames_under_color_channel(self) -> None:
        """Regression: frames must be nested under ``color/`` to match the metadata.

        A prior version wrote frames flat in the destination root while the metadata
        declared ``channels.color=color/%08d.jpg``, so the loader looked for
        ``color/00000001.jpg`` and failed to find any frame."""
        self._make_yolo_source(self.tmp / "yolo", count=3)
        prep.yolo_to_vot(self.tmp / "yolo", self.tmp / "out")
        self.assertTrue((self.tmp / "out" / "color" / frame_filename(1)).is_file())
        self.assertFalse((self.tmp / "out" / frame_filename(1)).exists(),
                         "frames must not be written flat in the destination root")

    def test_yolo_to_vot_frames_are_loadable(self) -> None:
        """The converted frames must be readable through the sequence loader."""
        self._make_yolo_source(self.tmp / "yolo", count=3, width=32, height=24)
        prep.yolo_to_vot(self.tmp / "yolo", self.tmp / "out")
        sequence = load_sequence(str(self.tmp / "out"))
        self.assertEqual(sequence.size, (32, 24))  # (width, height)
        channel = sequence.channel("color")
        assert channel is not None
        image = channel.frame(0)
        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual(image.shape, (24, 32, 3))  # (height, width, channels)

    def test_yolo_to_vot_metadata_matches_frames(self) -> None:
        """The declared channel pattern must resolve to frames that exist on disk."""
        self._make_yolo_source(self.tmp / "yolo", count=4, width=20, height=16)
        prep.yolo_to_vot(self.tmp / "yolo", self.tmp / "out", fps=25)
        metadata = read_metadata(self.tmp / "out", coerce=False, defaults=False)
        self.assertEqual(metadata["channels.color"], "color/" + DEFAULT_FRAME_MASK)
        self.assertEqual(metadata["length"], "4")
        self.assertEqual(metadata["width"], "20")
        self.assertEqual(metadata["height"], "16")
        self.assertEqual(metadata["fps"], "25")
        resolved = self.tmp / "out" / (metadata["channels.color"] % 1)
        self.assertTrue(resolved.is_file())

    def test_yolo_to_vot_groundtruth_conversion(self) -> None:
        """YOLO boxes become pixel ``x,y,w,h``; missing/empty labels become the UNKNOWN special code ``0``."""
        source = self.tmp / "yolo"
        source.mkdir()
        _write_frame(source / "frame_0.png", 32, 24, seed=0)
        (source / "frame_0.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        _write_frame(source / "frame_1.png", 32, 24, seed=1)  # no label file
        _write_frame(source / "frame_2.png", 32, 24, seed=2)
        (source / "frame_2.txt").write_text("")               # empty label file
        prep.yolo_to_vot(source, self.tmp / "out")
        lines = (self.tmp / "out" / GROUNDTRUTH_FILE).read_text().splitlines()
        # cx=16, cy=12, w=8, h=6 -> x=12, y=9
        self.assertEqual(lines, ["12.00,9.00,8.00,6.00", "0", "0"])

    def test_yolo_to_vot_rgba_input_flattened_to_rgb(self) -> None:
        """RGBA PNGs are flattened onto white and saved as 3-channel RGB frames."""
        from PIL import Image
        source = self.tmp / "yolo"
        source.mkdir()
        for index in range(2):
            Image.new("RGBA", (16, 12), (10, 20, 30, 128)).save(source / "frame_{}.png".format(index))
            (source / "frame_{}.txt".format(index)).write_text("0 0.5 0.5 0.5 0.5\n")
        prep.yolo_to_vot(source, self.tmp / "out")
        channel = load_sequence(str(self.tmp / "out")).channel("color")
        assert channel is not None
        image = channel.frame(0)
        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual(image.shape[2], 3)

    def test_yolo_to_vot_no_png_raises(self) -> None:
        source = self.tmp / "empty"
        source.mkdir()
        with self.assertRaises(FileNotFoundError):
            prep.yolo_to_vot(source, self.tmp / "out")

    def test_yolo_to_vot_then_reverse_roundtrip(self) -> None:
        """The raw-dataset pipeline (convert -> reverse) keeps frames loadable.

        Guards against a layout mismatch between the importer and the frame transforms:
        the reversed sequence must still resolve its frames under ``color/``."""
        self._make_yolo_source(self.tmp / "yolo", count=5)
        prep.yolo_to_vot(self.tmp / "yolo", self.tmp / "fwd")
        prep.reverse_sequence(self.tmp / "fwd", self.tmp / "rev")
        reversed_seq = load_sequence(str(self.tmp / "rev"))
        self.assertEqual(len(reversed_seq), 5)
        channel = reversed_seq.channel("color")
        assert channel is not None
        self.assertIsNotNone(channel.frame(0))
        self.assertTrue((self.tmp / "rev" / "color" / frame_filename(1)).is_file())

    def test_reverse_xml_box_frames(self) -> None:
        xml = ('<annotations><track><box frame="0"/><box frame="1"/>'
               '<box frame="2"/></track></annotations>')
        source = self.tmp / "ann.xml"
        source.write_text(xml)
        output = prep.reverse_xml_annotations(source)
        frames = sorted(int(b.attrib["frame"]) for b in ET.parse(output).getroot().iter("box"))
        self.assertEqual(frames, [0, 1, 2])
        original = [0, 1, 2]
        reversed_frames = [int(b.attrib["frame"]) for b in ET.parse(output).getroot().iter("box")]
        self.assertEqual(reversed_frames, list(reversed(original)))

    def test_reverse_xml_image_fallback(self) -> None:
        xml = ('<annotations><image id="0" name="img00000000.png"/>'
               '<image id="1" name="img00000001.png"/></annotations>')
        source = self.tmp / "ann.xml"
        source.write_text(xml)
        output = prep.reverse_xml_annotations(source)
        ids = [int(i.attrib["id"]) for i in ET.parse(output).getroot().iter("image")]
        self.assertEqual(ids, [1, 0])

    def test_reverse_xml_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            prep.reverse_xml_annotations(self.tmp / "missing.xml")

    def test_import_video_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            prep.import_video(self.tmp / "missing.mp4", self.tmp)

    def test_import_video_existing_target(self) -> None:
        video = self.tmp / "clip.mp4"
        video.write_bytes(b"not really a video")
        target = self.tmp / "clip"
        target.mkdir()
        (target / "stale.txt").write_text("x")
        with self.assertRaises(FileExistsError):
            prep.import_video(video, self.tmp)

    @unittest.skipUnless(_HAS_FFMPEG, "ffmpeg not available")
    def test_extract_frames_and_import_video(self) -> None:
        video = self.tmp / "clip.mp4"
        subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=32x24:rate=10",
                        str(video)], check=True, capture_output=True)
        sequence_dir = prep.import_video(video, self.tmp / "sequences")
        self.assertTrue(sequence_dir.is_dir())
        self.assertTrue((sequence_dir / METADATA_FILE).is_file())
        entries = SequenceList(self.tmp / "sequences").read()
        assert entries is not None
        self.assertIn("clip", entries)
        # extract_frames nests frames under the color/ channel subdirectory.
        self.assertGreater(len(list_image_files(sequence_dir / "color")), 0)


if __name__ == "__main__":
    unittest.main()
