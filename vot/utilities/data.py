"""Data structures for storing data in a grid."""

import functools
import unittest
from typing import Any, Callable, Iterator

Index = int | tuple[int, ...]

class Grid(object):
    """A grid is a multidimensional array with named dimensions."""

    @staticmethod
    def scalar(obj: Any) -> "Grid":
        """Creates a grid with a single cell containing the given object.

        :param obj: The object to store in the grid.
        :type obj: object
        """
        grid = Grid(1,1)
        grid[0, 0] = obj
        return grid

    def __init__(self, *size: int) -> None:
        """Creates a grid with the given dimensions.

        :param size: The size of each dimension.
        :type size: int
        """
        assert len(size) > 0
        self._size: tuple[int, ...] = size
        self._data: list[Any] = [None] * functools.reduce(lambda x, y: x * y, size)

    def _ravel(self, pos: Index) -> int:
        """Converts a multidimensional index to a single index.

        :param pos: The multidimensional index.
        :type pos: tuple

        :returns: The single index.
        :rtype: int"""
        if not isinstance(pos, tuple):
            pos = (pos, )
        assert len(pos) == len(self._size)

        raveled = 0
        row = 1
        for n, i in zip(reversed(self._size), reversed(pos)):
            if i < 0 or i >= n:
                raise IndexError("Index out of bounds")
            raveled = i * row + raveled
            row = row * n
        return raveled

    def _unravel(self, index: int) -> tuple[int, ...]:
        """Converts a single index to a multidimensional index.

        :param index: The single index.
        :type index: int

        :returns: The multidimensional index.
        :rtype: tuple"""
        unraveled = []
        for n in reversed(self._size):
            unraveled.append(index % n)
            index = index // n
        return tuple(reversed(unraveled))

    def __str__(self) -> str:
        """Returns a string representation of the grid."""
        return str(self._data)

    @property
    def dimensions(self) -> int:
        """Returns the number of dimensions of the grid."""
        return len(self._size)

    def size(self, i: int | None = None) -> int | tuple[int, ...]:
        """Returns the size of the grid or the size of a specific dimension.

        :param i: The dimension to query. If None, the size of the grid is returned.
        :type i: int

        :returns: The size of the grid or the size of the given dimension.
        :rtype: int"""
        if i is None:
            return tuple(self._size)
        assert i >= 0 and i < len(self._size)
        return self._size[i]

    def __len__(self) -> int:
        """Returns the number of elements in the grid."""
        return len(self._data)

    def __getitem__(self, i: Index) -> Any:
        """Returns the element at the given index.

        :param i: The index of the element. If the grid is one-dimensional, the index can be an integer.
        :type i: tuple

        :returns: The element at the given index.
        :rtype: object"""
        return self._data[self._ravel(i)]

    def __setitem__(self, i: Index, data: Any) -> None:
        """Sets the element at the given index.

        :param i: The index of the element. If the grid is one-dimensional, the index can be an integer.
        :type i: tuple
        :param data: The data to store at the given index.
        :type data: object
        """
        self._data[self._ravel(i)] = data

    def __iter__(self) -> Iterator[Any]:
        """Returns an iterator over the elements of the grid."""
        return iter(self._data)

    def cell(self, *i: int) -> "Grid":
        """Returns the element at the given index packed in a scalar grid.

        :param i: The index of the element. If the grid is one-dimensional, the index can be an integer.
        :type i: int

        :returns: The element at the given index packed in a scalar grid.
        :rtype: object"""
        return Grid.scalar(self[i])

    def column(self, i: int) -> "Grid":
        """Returns the column at the given index.

        :param i: The index of the column.
        :type i: int

        :returns: The column at the given index.
        :rtype: Grid"""
        assert self.dimensions == 2
        column = Grid(1, self._size[0])
        for j in range(self._size[0]):
            column[0, j] = self[j, i]
        return column

    def row(self, i: int) -> "Grid":
        """Returns the row at the given index.

        :param i: The index of the row.
        :type i: int

        :returns: The row at the given index.
        :rtype: Grid"""
        assert self.dimensions == 2
        row = Grid(self._size[1], 1)
        for j in range(self._size[1]):
            row[j, 0] = self[i, j]
        return row

    def foreach(self, cb: Callable[..., Any]) -> "Grid":
        """Applies a function to each element of the grid.

        :param cb: The function to apply to each element. The first argument is the element, the following arguments are the indices of the element.
        :type cb: function

        :returns: A grid containing the results of the function.
        :rtype: Grid"""
        result = Grid(*self._size)

        for i, x in enumerate(self._data):
            a = self._unravel(i)
            result[a] = cb(x, *a)

        return result

class TestGrid(unittest.TestCase):
    """Unit tests for the Grid class."""

    def test_foreach1(self) -> None:
        """Tests the foreach method."""

        a = Grid(5, 3)

        b = a.foreach(lambda x, i, j: 5)

        self.assertTrue(all([x == 5 for x in b]), "Output incorrect")

    def test_foreach2(self) -> None:
        """Tests the foreach method."""

        a = Grid(5, 6, 3)

        b = a.foreach(lambda x, i, j, k: k)

        reference = [x % 3 for x in range(len(a))]

        self.assertListEqual(list(b), reference)
