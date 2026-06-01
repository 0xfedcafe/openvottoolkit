"""Tests for the region module."""

import unittest

import numpy as np

from vot.region.raster import rasterize_polygon, rasterize_rectangle, copy_mask, calculate_overlap

class TestShapesMethods(unittest.TestCase):
    """Tests for the shapes module changes."""

    def test_rectangle_to_2points(self):
        from vot.region import Rectangle
        r = Rectangle(10, 20, 30, 40)
        self.assertEqual(r.to_2points(), (10, 20, 40, 60))

    def test_rectangle_populate_points(self):
        from vot.region import Rectangle
        pts = Rectangle.populate_points(10, 20, 40, 60)
        self.assertEqual(pts, [(10, 20), (40, 20), (40, 60), (10, 60)])

    def test_polygon_resize(self):
        from vot.region import Polygon
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p_resized = p.resize(2.0)
        self.assertEqual(p_resized.points(), [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)])

    def test_polygon_move(self):
        from vot.region import Polygon
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p_moved = p.move(5.0, 5.0)
        self.assertEqual(p_moved.points(), [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])

    def test_polygon_is_empty(self):
        from vot.region import Polygon
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertFalse(p.is_empty())
        
        # Degenerate points (width=0, height=0)
        p_empty = Polygon([(5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0)])
        self.assertTrue(p_empty.is_empty())

        # Float polygons should not be empty even if their bounds round to the same integer pixel.
        # Expected original behavior
        p_subpixel = Polygon([(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)])
        self.assertFalse(p_subpixel.is_empty())
        
        # Zero-height geometric polygon should be empty
        p_line = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (0.0, 0.0)])
        self.assertTrue(p_line.is_empty())

    def test_rectangle_is_empty(self):
        from vot.region import Rectangle
        r = Rectangle(0, 0, 10, 10)
        self.assertFalse(r.is_empty())
        
        # Sub-pixel rectangles should not be empty
        r_sub = Rectangle(0.0, 0.0, 0.4, 0.4)
        self.assertFalse(r_sub.is_empty())
        
        # Zero width or height should be empty
        r_empty = Rectangle(5, 5, 0, 0)
        self.assertTrue(r_empty.is_empty())
        r_line = Rectangle(0, 0, 10, 0)
        self.assertTrue(r_line.is_empty())

    def test_mask_move(self):
        from vot.region import Mask
        m = Mask(np.ones((10, 10), dtype=np.uint8), offset=(5, 5))
        m_moved = m.move(dx=2, dy=-1)
        self.assertEqual(m_moved.offset, (7, 4))
        self.assertTrue(not m_moved.is_empty())

class TestRasterMethods(unittest.TestCase):
    """Tests for the raster module."""

    def test_rasterize_polygon(self):
        """Tests if the polygon rasterization works correctly."""
        points = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.float32)
        np.testing.assert_array_equal(rasterize_polygon(points, (0, 0, 99, 99)), np.ones((100, 100), dtype=np.uint8))

    def test_rasterize_rectangle(self):
        """Tests if the rectangle rasterization works correctly."""
        np.testing.assert_array_equal(rasterize_rectangle(np.array([[0], [0], [100], [100]], dtype=np.float32), (0, 0, 99, 99)), np.ones((100, 100), dtype=np.uint8))

    def test_copy_mask(self):
        """Tests if the mask copy works correctly."""
        mask = np.ones((100, 100), dtype=np.uint8)
        np.testing.assert_array_equal(copy_mask(mask, (0, 0), (0, 0, 99, 99)), np.ones((100, 100), dtype=np.uint8))

    def test_calculate_overlap(self):
        """Tests if the overlap calculation works correctly."""
        from vot.region import Rectangle

        r1 = Rectangle(0, 0, 100, 100)
        self.assertEqual(calculate_overlap(r1, r1), 1)

        r1 = Rectangle(0, 0, 0, 0)        
        self.assertEqual(calculate_overlap(r1, r1), 1)

    def test_ignore_mask(self):
        """Tests if the mask ignore works correctly."""
        from vot.region import Mask

        r1 = Mask(np.ones((100, 100), dtype=np.uint8))
        r2 = Mask(np.ones((100, 100), dtype=np.uint8))
        ignore = Mask(np.zeros((100, 100), dtype=np.uint8))
        self.assertEqual(calculate_overlap(r1, r2, ignore=ignore), 1)

        ignore = Mask(np.ones((100, 100), dtype=np.uint8))
        self.assertEqual(calculate_overlap(r1, r2, ignore=ignore), 0)
        
    def test_empty_mask(self):
        """Tests if the empty mask is correctly detected."""
        from vot.region import Mask

        mask = Mask(np.zeros((100, 100), dtype=np.uint8))
        self.assertTrue(mask.is_empty())

        mask = Mask(np.ones((100, 100), dtype=np.uint8))
        self.assertFalse(mask.is_empty())

    def test_binary_format(self):
        """Tests if the binary format of a region matched the plain-text one."""
        import io

        from vot.region import Rectangle, Polygon, Mask, Point, Special, SpecialCode
        from vot.region.io import read_trajectory, write_trajectory
        from vot.region.raster import calculate_overlaps

        trajectory = [
            Rectangle(0, 0, 100, 100),
            Rectangle(0, 10, 100, 100),
            Rectangle(0, 0, 200, 100),
            Polygon([(0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)]),
            Mask(np.ones((100, 100), dtype=np.uint8)),
            Mask(np.zeros((100, 100), dtype=np.uint8)),
            Point(50, 50),
            Special(SpecialCode.INITIALIZATION)
        ]

        binf = io.BytesIO()
        txtf = io.StringIO()

        write_trajectory(binf, trajectory)
        write_trajectory(txtf, trajectory)

        binf.seek(0)
        txtf.seek(0)

        bint = read_trajectory(binf)
        txtt = read_trajectory(txtf)

        o1 = calculate_overlaps(bint, txtt, None)
        o2 = calculate_overlaps(bint, trajectory, None)

        self.assertTrue(np.all(np.array(o1) == 1))
        self.assertTrue(np.all(np.array(o2) == 1))

    def test_rle(self):
        """Test if RLE encoding works for limited stride representation."""
        from vot.region.io import rle_to_mask, mask_to_rle
        rle = [0, 2, 122103, 9, 260, 19, 256, 21, 256, 22, 254, 24, 252, 26, 251, 27, 250, 28, 249, 28, 250, 28, 249, 28, 249, 29, 249, 30, 247, 33, 245, 33, 244, 34, 244, 37, 241, 39, 239, 41, 237, 41, 236, 43, 235, 45, 234, 47, 233, 47, 231, 48, 230, 48, 230, 11, 7, 29, 231, 9, 9, 29, 230, 8, 11, 28, 230, 7, 12, 28, 230, 7, 13, 27, 231, 5, 14, 27, 233, 2, 16, 26, 253, 23, 255, 22, 256, 20, 258, 19, 259, 17, 3]
        rle = np.array(rle)
        m1 = rle_to_mask(np.array(rle, dtype=np.int32), 277, 478)

        r2 = mask_to_rle(m1, maxstride=255)
        m2 = rle_to_mask(np.array(r2, dtype=np.int32), 277, 478)

        np.testing.assert_array_equal(m1, m2)

class TestParseRegion(unittest.TestCase):
    """Tests for non-finite handling in :func:`vot.region.io.parse_region`."""

    def test_valid_shapes_parse(self):
        """Lines of finite values parse into the matching shape type."""
        from vot.region.io import parse_region
        from vot.region import Rectangle, Point, Polygon, Special

        self.assertIsInstance(parse_region("5,10,30,40"), Rectangle)
        self.assertIsInstance(parse_region("5,10"), Point)
        self.assertIsInstance(parse_region("0,0,0,100,100,100,100,0"), Polygon)
        # A single token is a special-region code, not a shape.
        self.assertIsInstance(parse_region("0"), Special)

    def test_special_code_roundtrip(self):
        """A lone integer token decodes to the matching :class:`SpecialCode`."""
        from vot.region.io import parse_region
        from vot.region import Special, SpecialCode

        zero = parse_region("0")
        assert isinstance(zero, Special)
        self.assertEqual(zero.code, SpecialCode.UNKNOWN)

        two = parse_region("2")
        assert isinstance(two, Special)
        self.assertEqual(two.code, SpecialCode.FAILURE)

    def test_all_nonfinite_is_unknown(self):
        """A line that is entirely NaN/Inf is the "object absent" sentinel."""
        from vot.region.io import parse_region
        from vot.region import Special, SpecialCode

        for line in ("nan", "nan,nan", "nan,nan,nan,nan", "inf,inf,inf,inf",
                     "nan,inf,-inf,nan"):
            region = parse_region(line)
            assert isinstance(region, Special), line
            self.assertEqual(region.code, SpecialCode.UNKNOWN, msg=line)

    def test_partial_nonfinite_raises(self):
        """A line mixing finite and non-finite values is corrupt and must raise."""
        from vot.region.io import parse_region
        from vot.region import RegionException

        for line in ("nan,5,10,20", "5,10,nan,20", "inf,5,10,20",
                     "5,nan", "0,0,nan,100,100,100,100,0"):
            with self.assertRaises(RegionException, msg=line):
                parse_region(line)

    def test_partial_nonfinite_error_names_line(self):
        """The raised error includes the offending line for diagnosis."""
        from vot.region.io import parse_region
        from vot.region import RegionException

        with self.assertRaisesRegex(RegionException, r"nan,5,10,20"):
            parse_region("nan,5,10,20")


class TestRegionModule(unittest.TestCase):
    """Tests for module-level concerns and assorted fixes."""

    def test_module_has_docstring(self):
        """The module docstring must survive the ``from __future__`` import ordering."""
        import vot.region
        self.assertIsNotNone(vot.region.__doc__)
        assert vot.region.__doc__ is not None
        self.assertIn("region", vot.region.__doc__.lower())

    def test_mask_convert_rectangle_overlaps_fully(self):
        """Converting a Rectangle to a Mask must keep its location (no double offset)."""
        from vot.region import Rectangle, Mask, calculate_overlap

        rect = Rectangle(10, 10, 5, 5)
        self.assertAlmostEqual(calculate_overlap(rect, Mask.convert(rect)), 1.0)

    def test_read_readonly_binary_trajectory(self):
        """A read-only binary trajectory must still be read as binary (peeked with 'rb')."""
        import os
        import stat
        import tempfile

        from vot.region import Rectangle, Region
        from vot.region.io import read_trajectory, write_trajectory
        from vot.region.raster import calculate_overlaps

        trajectory: list[Region] = [Rectangle(0, 0, 10, 10), Rectangle(5, 5, 10, 10)]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trajectory.bin")
            write_trajectory(path, trajectory)
            os.chmod(path, stat.S_IRUSR)
            try:
                read_back = read_trajectory(path)
                overlaps = calculate_overlaps(read_back, trajectory, None)
                self.assertTrue(all(o == 1 for o in overlaps))
            finally:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


class TestEncodeMask(unittest.TestCase):
    """Tests for :func:`vot.region.io.encode_mask`."""

    def test_empty_mask(self):
        """An all-zero mask encodes to the empty sentinel, not a spurious 1x1 region."""
        from vot.region.io import encode_mask
        bbox, rle = encode_mask(np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(bbox, (0, 0, 0, 0))
        self.assertEqual(rle, [0])

    def test_nonempty_mask_bbox(self):
        """A non-empty mask reports its minimal bounding box as (tl_x, tl_y, w, h)."""
        from vot.region.io import encode_mask
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1  # rows 2-4, cols 3-6
        bbox, _ = encode_mask(mask)
        self.assertEqual(bbox, (3, 2, 4, 3))