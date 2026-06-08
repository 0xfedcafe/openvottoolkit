"""Tests for workspace related methods and classes."""

import logging
import os
import tempfile
import unittest
from vot import get_logger
from vot.workspace.storage import Cache, LocalStorage

from vot.workspace import Workspace, NullStorage

class TestStacks(unittest.TestCase):
    """Tests for workspace related methods."""

    def test_void_storage(self) -> None:
        """Test if void storage works."""

        storage = NullStorage()

        with storage.write("test.data") as handle:
            handle.write("test")

        self.assertIsNone(storage.read("test.data"))

    def test_local_storage(self) -> None:
        """Test if local storage works."""

        with tempfile.TemporaryDirectory() as testdir:
            storage = LocalStorage(testdir)

            with storage.write("test.txt") as handle:
                handle.write("Test")

            self.assertTrue(storage.isdocument("test.txt"))

        # TODO: more tests

    def test_local_storage_copy(self) -> None:
        """Test if local storage copies files into the storage root."""

        with tempfile.TemporaryDirectory() as testdir:
            source = os.path.join(testdir, "source.txt")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("Test")

            storage = LocalStorage(os.path.join(testdir, "storage"))
            storage.copy(source, os.path.join("nested", "copied.txt"))

            self.assertTrue(storage.isdocument(os.path.join("nested", "copied.txt")))
            self.assertTrue(os.path.isfile(source))

    def test_workspace_create(self) -> None:
        """Test if workspace creation works."""

        get_logger().setLevel(logging.WARN) # Disable progress bar

        default_config = dict(stack="tests/basic", registry=["./trackers.ini"])

        with tempfile.TemporaryDirectory() as testdir:
            Workspace.initialize(testdir, default_config, download=True)
            Workspace.load(testdir)

    def test_cache(self) -> None:
        """Test if local storage cache works."""

        with tempfile.TemporaryDirectory() as testdir:

            cache = Cache(LocalStorage(testdir))

            self.assertFalse("test" in cache)

            cache["test"] = 1

            self.assertTrue("test" in cache)

            self.assertTrue(cache["test"] == 1)

            del cache["test"]

            self.assertRaises(KeyError, lambda: cache["test"])

    def test_cache_nested_key(self) -> None:
        """Test if persistent cache tuple keys are stored as relative paths."""

        with tempfile.TemporaryDirectory() as testdir:

            storage = LocalStorage(testdir)
            cache = Cache(storage)
            key = ("analysis", "experiment", "value")

            cache[key] = {"result": 1}

            self.assertTrue(storage.isdocument(os.path.join(*key)))
            self.assertEqual(Cache(storage)[key], {"result": 1})

    def test_cache_with_non_filesystem_storage(self) -> None:
        """Test if cache tuple keys do not require filesystem storage."""

        cache = Cache(NullStorage())
        key = ("analysis", "value")

        cache[key] = 1

        self.assertTrue(key in cache)
        self.assertEqual(cache[key], 1)

        del cache[key]

        self.assertFalse(key in cache)


class TestLocalStorageListing(unittest.TestCase):
    """Tests for ``LocalStorage.folders`` / ``LocalStorage.documents``.

    These distinguish two cases that both yield an empty list but for different reasons: a
    storage root that does not exist on disk, and one that exists but holds nothing of the
    requested kind. ``folders`` lists only subdirectories, ``documents`` only files.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def test_missing_root_lists_as_empty(self) -> None:
        """A storage whose root does not exist lists as empty instead of raising."""
        storage = LocalStorage(os.path.join(self.root, "does_not_exist"))
        self.assertFalse(os.path.exists(storage.base))
        self.assertEqual(storage.folders(), [])
        self.assertEqual(storage.documents(), [])

    def test_empty_root_lists_as_empty(self) -> None:
        """A storage whose root exists but is empty lists as empty."""
        storage = LocalStorage(self.root)
        self.assertTrue(os.path.isdir(storage.base))
        self.assertEqual(storage.folders(), [])
        self.assertEqual(storage.documents(), [])

    def test_folders_lists_only_subdirectories(self) -> None:
        """``folders`` returns subdirectory names and ignores files."""
        os.mkdir(os.path.join(self.root, "alpha"))
        os.mkdir(os.path.join(self.root, "beta"))
        with open(os.path.join(self.root, "note.txt"), "w") as fp:
            fp.write("x")

        storage = LocalStorage(self.root)
        self.assertEqual(sorted(storage.folders()), ["alpha", "beta"])

    def test_documents_lists_only_files(self) -> None:
        """``documents`` returns file names and ignores subdirectories."""
        os.mkdir(os.path.join(self.root, "alpha"))
        with open(os.path.join(self.root, "note.txt"), "w") as fp:
            fp.write("x")

        storage = LocalStorage(self.root)
        self.assertEqual(storage.documents(), ["note.txt"])

    def test_substorage_of_missing_subfolder_lists_as_empty(self) -> None:
        """A substorage pointing at a not-yet-created subfolder lists as empty.

        This is the fresh-workspace case: ``storage.substorage("results")`` before any
        evaluation has written results.
        """
        storage = LocalStorage(self.root).substorage("results")
        self.assertFalse(os.path.exists(storage.base))
        self.assertEqual(storage.folders(), [])
        self.assertEqual(storage.documents(), [])
