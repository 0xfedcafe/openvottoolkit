"""Unit tests for analysis module."""


import unittest

class Tests(unittest.TestCase):
    """Unit tests for analysis module."""

    @staticmethod
    def _make_trajectory(regions):
        """Builds a :class:`Trajectory` from a list of regions for marker tests."""
        from vot.tracker.results import Trajectory
        trajectory = Trajectory(len(regions))
        for frame, region in enumerate(regions):
            trajectory.set(frame, region)
        return trajectory

    def test_perfect_accuracy(self):
        import numpy as np

        from vot.region import Rectangle, Region, Special, SpecialCode
        from vot.analysis.accuracy import gather_overlaps

        trajectory: list[Region] = [Rectangle(0, 0, 100, 100) for _ in range(30)]
        groundtruth: list[Region] = [Rectangle(0, 0, 100, 100) for _ in range(30)]

        trajectory[0] = Special(SpecialCode.INITIALIZATION)

        overlaps, _ = gather_overlaps(trajectory, groundtruth)

        self.assertEqual(np.mean(overlaps), 1)

    def _run_eao_subcompute(self, trajectory_regions):
        """Run :meth:`EAOCurve.subcompute` against a stubbed experiment that
        returns the supplied trajectory, intercepting ``compute_eao_curve`` so
        the test can inspect how the trajectory was split into runs.

        Returns the ``(overlaps_all, weights_all, success_all)`` lists that
        ``subcompute`` would have passed on to ``compute_eao_curve``.
        """
        from unittest.mock import MagicMock, patch

        from vot.region import Rectangle
        from vot.experiment.multirun import SupervisedExperiment
        from vot.analysis.supervised import EAOCurve

        trajectory = self._make_trajectory(trajectory_regions)

        groundtruth = [Rectangle(0, 0, 10, 10) for _ in range(len(trajectory_regions))]

        experiment = MagicMock(spec=SupervisedExperiment)
        experiment.gather.return_value = [trajectory]

        sequence = MagicMock()
        sequence.groundtruth.return_value = groundtruth
        sequence.name = "seq"
        sequence.size = (100, 100)

        captured: dict = {}

        def fake_curve(overlaps, weights, success, min_length=0):
            captured["overlaps"] = overlaps
            captured["weights"] = weights
            captured["success"] = success
            return [0.0]

        with patch("vot.analysis.supervised.compute_eao_curve", side_effect=fake_curve):
            EAOCurve().subcompute(experiment, MagicMock(), [sequence], [])

        return captured["overlaps"], captured["weights"], captured["success"]

    def test_eao_curve_pairs_well_formed_trajectory(self):
        """Normal trajectory (init-fail-init): one failed run per init/fail pair,
        one successful run for the final init that survives to the end."""
        from vot.region import Rectangle, Special, SpecialCode

        # frames 0..3 tracked, FAILURE at 4, gap, INIT at 6, frames 7..11 tracked
        regions = [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(3)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.UNKNOWN)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(5)]

        overlaps_all, weights_all, success_all = self._run_eao_subcompute(regions)

        self.assertEqual(success_all, [False, True])
        self.assertEqual(weights_all, [1.0, 1.0])
        self.assertEqual([len(o) for o in overlaps_all], [4, 6])

    def test_eao_curve_treats_crash_as_run_terminator(self):
        """A CRASH after a successful INIT terminates the tracking run for EAO
        purposes — equivalent to a FAILURE, since either way the tracker
        stopped producing output."""
        from vot.region import Rectangle, Special, SpecialCode

        # Tracker runs from 0..3, crashes at frame 4, recovers at frame 6.
        regions = [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(3)]
        regions += [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.UNKNOWN)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(5)]

        overlaps_all, weights_all, success_all = self._run_eao_subcompute(regions)

        # Run 0..4 ends at CRASH (failed); run 6..end survives (success).
        self.assertEqual(success_all, [False, True])
        self.assertEqual(weights_all, [1.0, 1.0])
        self.assertEqual([len(o) for o in overlaps_all], [4, 6])

    def test_eao_curve_orphan_crash_contributes_length_one_failed_run(self):
        """A CRASH at frame 0 (the very first init crashed) is recorded as a
        length-1 failed run so the incident still drags down the EAO curve,
        equivalent to an ``INIT@0 + FAILURE@1`` pair that fails immediately."""
        from vot.region import Rectangle, Special, SpecialCode

        # First init crashed at frame 0, recovered at frame 1.
        regions = [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(6)]

        overlaps_all, weights_all, success_all = self._run_eao_subcompute(regions)

        # Length-1 failed run for the orphan CRASH + length-7 success run for
        # the recovery init.
        self.assertEqual(success_all, [False, True])
        self.assertEqual(weights_all, [1.0, 1.0])
        self.assertEqual([len(o) for o in overlaps_all], [1, len(regions) - 1])

    def test_eao_curve_multiple_orphan_crashes_each_contribute_failed_run(self):
        """Three consecutive init crashes before a successful init contribute
        three length-1 failed runs; the run beginning at the successful init
        adds another failed run ending at the eventual FAILURE."""
        from vot.region import Rectangle, Special, SpecialCode

        regions = [Special(SpecialCode.CRASH) for _ in range(3)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(4)]
        regions += [Special(SpecialCode.FAILURE)]

        overlaps_all, weights_all, success_all = self._run_eao_subcompute(regions)

        # Three length-1 orphan failed runs + one length-5 tracking-failure run
        # (init at frame 3, failure at frame 8 — half-open slice covers 3..7).
        self.assertEqual(success_all, [False, False, False, False])
        self.assertEqual(weights_all, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual([len(o) for o in overlaps_all], [1, 1, 1, 5])

    def test_eao_curve_orphan_crash_between_runs_is_failed_run(self):
        """A CRASH that happens during recovery (between a FAILURE and the
        next successful INIT) is recorded as its own length-1 failed run, so
        the wasted recovery attempt is visible to the curve."""
        from vot.region import Rectangle, Special, SpecialCode

        # Run 0..3 (failed at 4); crash during recovery at 6; recovered at 7.
        regions = [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(3)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.UNKNOWN)]
        regions += [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(3)]

        overlaps_all, weights_all, success_all = self._run_eao_subcompute(regions)

        # Three runs: 0..3 failed at FAILURE@4 (length 4), orphan CRASH@6
        # (length 1), 7..end survived (length 4).
        self.assertEqual(success_all, [False, False, True])
        self.assertEqual(weights_all, [1.0, 1.0, 1.0])
        self.assertEqual([len(o) for o in overlaps_all], [4, 1, 4])

    def test_eao_curve_failure_and_crash_at_first_frame_are_equivalent(self):
        """A tracker that initializes at frame 0 then fails immediately
        produces the same EAO contribution as a tracker that crashes on the
        first frame: both spend one frame on a failed attempt and then
        successfully track the same five frames after a 5-frame grace
        period. The redesign requires this equivalence because a crash and a
        first-frame failure both mean "the tracker produced no usable output
        for one frame, then was reset"."""
        from vot.region import Rectangle, Region, Special, SpecialCode

        rect = Rectangle(0, 0, 10, 10)
        skip = 5
        tracked = 5

        # Scenario A — "considered initialized at frame 0, fails":
        #   INIT@0, FAILURE@1, UNKNOWN@2..5, INIT@6, Rect@7..11.
        regions_a: list[Region] = [Special(SpecialCode.INITIALIZATION)]
        regions_a += [Special(SpecialCode.FAILURE)]
        regions_a += [Special(SpecialCode.UNKNOWN) for _ in range(skip - 1)]
        regions_a += [Special(SpecialCode.INITIALIZATION)]
        regions_a += [rect for _ in range(tracked)]

        # Scenario B — "given first frame, crashes":
        #   CRASH@0, UNKNOWN@1..4, INIT@5, Rect@6..10.
        regions_b: list[Region] = [Special(SpecialCode.CRASH)]
        regions_b += [Special(SpecialCode.UNKNOWN) for _ in range(skip - 1)]
        regions_b += [Special(SpecialCode.INITIALIZATION)]
        regions_b += [rect for _ in range(tracked)]

        # Combined incident count is identical (1 in each case): the FAILURE
        # in A and the CRASH in B both represent the single wasted frame.
        traj_a = self._make_trajectory(regions_a)
        traj_b = self._make_trajectory(regions_b)
        fa, ca = len(traj_a.failures()), len(traj_a.crashes())
        fb, cb = len(traj_b.failures()), len(traj_b.crashes())
        self.assertEqual(fa + ca, fb + cb)
        self.assertEqual(fa + ca, 1)

        # EAO contribution is identical: one length-1 failed run with the
        # incident's zero overlap, then one length-``tracked + 1`` success
        # run (the +1 is the INIT frame itself, which scores 0 against the
        # groundtruth before the Rects all score 1).
        overlaps_a, weights_a, success_a = self._run_eao_subcompute(regions_a)
        overlaps_b, weights_b, success_b = self._run_eao_subcompute(regions_b)

        self.assertEqual(success_a, success_b)
        self.assertEqual(weights_a, weights_b)
        self.assertEqual([len(o) for o in overlaps_a],
                         [len(o) for o in overlaps_b])
        for oa, ob in zip(overlaps_a, overlaps_b):
            self.assertEqual(list(oa), list(ob))
        # And concretely: a length-1 failed run + a length-``tracked + 1``
        # success run, both starting from a zero-overlap special frame.
        self.assertEqual(success_a, [False, True])
        self.assertEqual([len(o) for o in overlaps_a], [1, tracked + 1])

    def test_count_failures_only_counts_paired_failures(self):
        """``Trajectory.failures`` counts only FAILURE markers that terminate a
        real tracking run; CRASH markers and unpaired FAILUREs (defensive
        handling for malformed input) are ignored."""
        from vot.region import Rectangle, Region, Special, SpecialCode

        # Two orphan crashes during init, then a successful init, then a real
        # tracking failure, then a crash during recovery, then padding.
        regions: list[Region] = [Special(SpecialCode.CRASH) for _ in range(2)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(3)]
        regions += [Special(SpecialCode.FAILURE)]            # real tracking failure
        regions += [Special(SpecialCode.UNKNOWN) for _ in range(2)]
        regions += [Special(SpecialCode.CRASH)]              # orphan: init crashed during recovery
        regions += [Special(SpecialCode.UNKNOWN) for _ in range(2)]

        trajectory = self._make_trajectory(regions)
        self.assertEqual(len(trajectory.failures()), 1)
        self.assertEqual(len(trajectory), len(regions))

    def test_count_crashes_counts_all_crash_markers(self):
        """``Trajectory.crashes`` reports every CRASH marker — orphan crashes
        during recovery count the same as crashes that terminate a run, since
        both mean the tracker process failed to produce output."""
        from vot.region import Rectangle, Region, Special, SpecialCode

        regions: list[Region] = [Special(SpecialCode.CRASH) for _ in range(2)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(3)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.UNKNOWN) for _ in range(2)]
        regions += [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.UNKNOWN) for _ in range(2)]

        trajectory = self._make_trajectory(regions)
        self.assertEqual(len(trajectory.crashes()), 3)
        self.assertEqual(len(trajectory), len(regions))

    def test_count_failures_pairs_alternating_runs(self):
        """A well-formed init/fail/init/fail trajectory counts one failure per
        terminated run; a trailing init that survives is not counted."""
        from vot.region import Rectangle, Region, Special, SpecialCode

        regions: list[Region] = [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]

        trajectory = self._make_trajectory(regions)
        self.assertEqual(len(trajectory.failures()), 2)
        self.assertEqual(len(trajectory), len(regions))

    def test_marker_metrics_return_frame_indices(self):
        """``Trajectory.failures`` / ``crashes`` / ``markers`` report the frame
        positions of each incident, which the cumulative curves depend on."""
        from vot.region import Rectangle, Region, Special, SpecialCode

        # INIT@0, track, FAILURE@3, CRASH@4 (orphan), INIT@5, track, FAILURE@8.
        regions: list[Region] = [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]
        regions += [Special(SpecialCode.FAILURE)]

        trajectory = self._make_trajectory(regions)

        self.assertEqual(trajectory.failures(), [3, 8])
        self.assertEqual(trajectory.crashes(), [4])
        init_idxs, failure_idxs, crash_idxs = trajectory.markers()
        self.assertEqual(init_idxs, [0, 5])
        self.assertEqual(failure_idxs, [3, 8])
        self.assertEqual(crash_idxs, [4])

    def test_eao_curve_handles_trajectory_with_no_inits(self):
        """A trajectory where every recovery attempt crashed (no
        INITIALIZATION markers at all) should still be contributed as a
        single failed run so ``compute_eao_curve`` does not receive an empty
        list."""
        from vot.region import Special, SpecialCode

        regions = [Special(SpecialCode.CRASH) for _ in range(5)]

        overlaps_all, weights_all, success_all = self._run_eao_subcompute(regions)

        self.assertEqual(success_all, [False])
        self.assertEqual(weights_all, [1.0])
        self.assertEqual(len(overlaps_all), 1)
        self.assertEqual(len(overlaps_all[0]), 5)

    def test_per_frame_times_keeps_frame_indices_and_drops_zeros(self):
        """``_per_frame_times`` returns ``(frame_index, elapsed)`` for every frame
        with a positive ``time`` property. Frames with zero or missing time —
        which is how realtime records the skipped frames it replayed cached
        regions for — are dropped, but the surviving entries keep their real
        absolute frame indices."""
        from vot.region import Rectangle, Special, SpecialCode
        from vot.tracker.results import Trajectory
        from vot.analysis.speed import _per_frame_times

        # Realtime-shaped trajectory: init + warmup + sparse live invocations.
        trajectory = Trajectory(10)
        trajectory.set(0, Special(SpecialCode.INITIALIZATION), {"time": 0.001})
        for frame, time in [(1, 2.0), (2, 2.1), (3, 2.05), (4, 2.0)]:
            trajectory.set(frame, Rectangle(0, 0, 10, 10), {"time": time})
        # Skipped frames: cached status replay with time=0.
        for frame in (5, 6, 7, 8):
            trajectory.set(frame, Rectangle(0, 0, 10, 10), {"time": 0.0})
        # One late live invocation at the catch-up point.
        trajectory.set(9, Rectangle(0, 0, 10, 10), {"time": 2.2})

        pairs = _per_frame_times(trajectory)

        # Six live samples (init + 4 warmup + 1 late) with their true indices.
        self.assertEqual([f for f, _ in pairs], [0, 1, 2, 3, 4, 9])
        self.assertAlmostEqual(pairs[0][1], 0.001)
        self.assertAlmostEqual(pairs[-1][1], 2.2)

    def test_compute_speed_curve_preserves_frame_indices(self):
        """``compute_speed`` returns the FPS curve as ``(frame_index, fps)``
        pairs so a sparse realtime run plots its points at their real frame
        positions instead of collapsing to a packed array index. ``skip_initial``
        applies to the average only — the curve preserves every live sample."""
        from vot.region import Rectangle, Special, SpecialCode
        from vot.tracker.results import Trajectory
        from vot.analysis.speed import compute_speed

        trajectory = Trajectory(70)
        # init + 4 warmup live frames, ~60 skipped, then one late live frame.
        trajectory.set(0, Special(SpecialCode.INITIALIZATION), {"time": 0.001})
        for frame, time in [(1, 5.0), (2, 2.0), (3, 2.0), (4, 2.0)]:
            trajectory.set(frame, Rectangle(0, 0, 10, 10), {"time": time})
        for frame in range(5, 64):
            trajectory.set(frame, Rectangle(0, 0, 10, 10), {"time": 0.0})
        trajectory.set(64, Rectangle(0, 0, 10, 10), {"time": 2.0})

        metrics, per_frame_fps = compute_speed(trajectory, skip_initial=5)

        # The curve includes all six live frames at their real indices.
        self.assertEqual([f for f, _ in per_frame_fps], [0, 1, 2, 3, 4, 64])
        # ``skip_initial=5`` leaves only the late frame in the average; FPS = 1/2.
        self.assertAlmostEqual(metrics.fps, 0.5)

    def test_sequence_speed_merges_disjoint_trajectories_by_absolute_frame(self):
        """``SequenceSpeed.subcompute`` merges per-frame FPS samples across
        trajectories by absolute frame index — so two multistart anchor runs
        covering disjoint slices of the sequence stay at their real positions
        instead of being averaged element-wise on a packed array."""
        from unittest.mock import MagicMock

        from vot.region import Rectangle
        from vot.tracker.results import Trajectory
        from vot.experiment.multistart import MultiStartExperiment
        from vot.analysis.speed import SequenceSpeed

        # Trajectory A covers frames 0..2 with 0.5s/frame (= 2 FPS).
        trajectory_a = Trajectory(10)
        for frame in range(3):
            trajectory_a.set(frame, Rectangle(0, 0, 10, 10), {"time": 0.5})

        # Trajectory B covers frames 5..7 with 0.25s/frame (= 4 FPS).
        trajectory_b = Trajectory(10)
        for frame in range(5, 8):
            trajectory_b.set(frame, Rectangle(0, 0, 10, 10), {"time": 0.25})

        experiment = MagicMock(spec=MultiStartExperiment)
        experiment.gather.return_value = [trajectory_a, trajectory_b]

        sequence = MagicMock()
        sequence.__len__ = lambda self: 10
        sequence.name = "seq"

        analysis = SequenceSpeed(skip_initial=0)
        result = analysis.subcompute(experiment, MagicMock(), sequence, [])

        _, _, per_frame_fps, frames = result
        self.assertEqual(frames, 10)
        # Disjoint slices → six independent samples at their absolute frames.
        self.assertEqual([f for f, _ in per_frame_fps], [0, 1, 2, 5, 6, 7])
        for frame, fps in per_frame_fps[:3]:
            self.assertAlmostEqual(fps, 2.0)
        for frame, fps in per_frame_fps[3:]:
            self.assertAlmostEqual(fps, 4.0)

    def test_accuracy_robustness_surfaces_crash_count(self):
        """``AccuracyRobustness`` exposes a separate Crashes measure and
        folds the crash count into the AR exp-decay so crashing trackers
        don't hide behind unchanged Robustness."""
        import math
        from unittest.mock import MagicMock

        from vot.region import Rectangle, Special, SpecialCode
        from vot.tracker.results import Trajectory
        from vot.experiment.multirun import SupervisedExperiment
        from vot.analysis.supervised import AccuracyRobustness

        # 1 tracking failure (FAILURE@4), 2 crashes (CRASH@0 orphan, CRASH@6
        # between runs), one successful tail run.
        regions = [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.UNKNOWN)]
        regions += [Special(SpecialCode.CRASH)]
        regions += [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(2)]

        trajectory = Trajectory(len(regions))
        for frame, region in enumerate(regions):
            trajectory.set(frame, region)

        groundtruth = [Rectangle(0, 0, 10, 10) for _ in range(len(regions))]

        experiment = MagicMock(spec=SupervisedExperiment)
        experiment.gather.return_value = [trajectory]

        sequence = MagicMock()
        sequence.groundtruth.return_value = groundtruth
        sequence.name = "seq"
        sequence.size = (100, 100)
        sequence.__len__ = lambda self: len(regions)

        analysis = AccuracyRobustness()
        result = analysis.subcompute(experiment, MagicMock(), sequence, [])

        accuracy, failures, crashes, ar, frames = result
        self.assertEqual(failures, 1)
        self.assertEqual(crashes, 2)
        self.assertEqual(frames, len(regions))
        # AR exp-decay uses (failures + crashes) — sanity check the value.
        expected_ar_x = math.exp(-((failures + crashes) / len(regions)) * analysis.sensitivity)
        self.assertAlmostEqual(ar[0], expected_ar_x, places=6)

    def test_eao_curve_min_length_pads_failed_runs(self):
        """``compute_eao_curve`` extends failed runs as ``sum/N_s`` (the VOT2015
        zero-padding definition) out to ``min_length``. Without the extension the
        curve ends at the longest run and a score window outliving every run is
        silently truncated, inflating EAO of frequently-failing trackers."""
        import numpy as np
        from vot.analysis.supervised import compute_eao_curve

        # one failed run: init frame + 32 tracked frames at overlap 0.8
        curve = compute_eao_curve([[1.0] + [0.8] * 32], [1.0], [False], 51)

        self.assertEqual(len(curve), 51)
        self.assertAlmostEqual(float(curve[32]), 0.8, places=5)
        self.assertAlmostEqual(float(curve[50]), 0.8 * 32 / 50, places=5)
        # score over [20, 50]: 0.699 once the decay tail counts, not 0.8
        self.assertAlmostEqual(float(np.nanmean(curve[20:51])), 0.6994, places=3)

    def test_eao_curve_min_length_marks_undefined_columns_nan(self):
        """Columns where no run is active (a never-failing tracker past its
        longest success) are NaN — the definition removes all segments there —
        and must not be silently zeroed."""
        import numpy as np
        from vot.analysis.supervised import compute_eao_curve

        curve = compute_eao_curve([[1.0, 0.9, 0.9]], [1.0], [True], 10)

        self.assertEqual(len(curve), 10)
        self.assertTrue(np.all(np.isnan(curve[3:])))
        self.assertAlmostEqual(float(np.nanmean(curve[0:10])), (1.0 + 0.9 + 0.9) / 3, places=5)

    def test_eao_curve_default_min_length_preserves_behavior(self):
        """Without ``min_length`` the curve is unchanged from the original
        formulation (hand-checked values)."""
        from vot.analysis.supervised import compute_eao_curve

        curve = compute_eao_curve([[1, .5, .5, .5], [1, 0]], [1.0, 1.0], [True, False])

        self.assertEqual(curve.tolist(), [1.0, 0.25, 0.25, 0.25])

    def test_eao_curve_subcompute_extends_to_sequence_length(self):
        """``EAOCurve`` spans the longest sequence, not the longest run, so a
        score interval is averaged over the same columns for every tracker."""
        from unittest.mock import MagicMock

        from vot.region import Rectangle, Special, SpecialCode
        from vot.experiment.multirun import SupervisedExperiment
        from vot.analysis.supervised import EAOCurve

        regions = [Special(SpecialCode.INITIALIZATION)]
        regions += [Rectangle(0, 0, 10, 10) for _ in range(9)]
        regions += [Special(SpecialCode.FAILURE)]
        regions += [Special(SpecialCode.UNKNOWN) for _ in range(39)]
        trajectory = self._make_trajectory(regions)
        groundtruth = [Rectangle(0, 0, 10, 10) for _ in range(50)]

        experiment = MagicMock(spec=SupervisedExperiment)
        experiment.gather.return_value = [trajectory]

        sequence = MagicMock()
        sequence.groundtruth.return_value = groundtruth
        sequence.name = "seq"
        sequence.size = (100, 100)
        sequence.__len__ = lambda self: 50

        curve = EAOCurve().subcompute(experiment, MagicMock(), [sequence], [])[0]

        self.assertEqual(len(curve), 50)
        self.assertAlmostEqual(float(curve[9]), 1.0, places=5)
        # failed run carries 9 perfect frames, zero-padded by definition
        self.assertAlmostEqual(float(curve[49]), 9 / 49, places=5)

    def test_eao_score_skips_undefined_columns(self):
        """``EAOScore`` excludes NaN (undefined-by-definition) curve columns from
        the window mean instead of poisoning it."""
        from unittest.mock import MagicMock

        import numpy as np

        from vot.analysis.supervised import EAOScore
        from vot.utilities.data import Grid

        grid = Grid.scalar((np.array([0.5, 0.5, np.nan, np.nan]),))
        score = EAOScore(low=0, high=3)

        result = score.compute(MagicMock(), [MagicMock()], [], [grid])

        self.assertAlmostEqual(result[0, 0][0], 0.5)
