"""Invariants of the board-calibration emitter.

The calibration is the outer loop's whole input: `--board-calibration` scales every
dispatch cost by it, so a mistake here does not crash anything -- it quietly changes
which schedule the solver believes is feasible. These pin the parts where a plausible
mistake is invisible in the output:

1. the tick rate is the project's one tick rate, not a second copy of 24e6;
2. queue delay never reaches the multiplier, because charging a wait to an op makes the
   scheduler pay twice for its own placement;
3. instances of the same dispatch pool into ONE key, because a multiplier is a property
   of the code and the core rather than of which instance ran;
4. the pooled op tier is floored and the per-dispatch tier is not -- a few-hundred-tick
   dispatch shows 18x from timer granularity alone, which must not be allowed to inflate
   a multiplier that other networks then inherit;
5. `--validate-against` reports disagreement rather than smoothing it.

They need no board and no solver.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SCRIPT = os.path.join(_REPO, "scripts", "emit_board_calibration.py")


def _load():
    """Import the script by path; `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("emit_board_calibration", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["emit_board_calibration"] = mod
    spec.loader.exec_module(mod)
    return mod


cal = _load()

#: The columns a real runner trace carries. Written out in full so a change to the
#: runner's schema shows up here as a failure rather than as a silently empty table.
COLS = ["entry_id", "network", "instance", "dispatch_id", "op", "name", "core_kind",
        "hart", "predicted_start_ms", "predicted_duration_ms", "worker_kind_idx",
        "worker_hart", "actual_start_cycles", "actual_end_cycles"]

TICKS_PER_MS = cal.K1_RDTIME_HZ / 1e3


def write_trace(path, rows):
    """`rows` = [(net, instance, dispatch_id, op, pred_ms, actual_ms, queue_ms)]."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for i, (net, inst, did, op, pred, actual, queue) in enumerate(rows):
            start = int((10.0 + queue) * TICKS_PER_MS)
            end = start + int(actual * TICKS_PER_MS)
            w.writerow([i, net, inst, did, op, f"{op}.{did}", "rvv_c1", 4,
                        10.0, f"{pred:.6f}", 2, 4, start, end])


class TheTickRate(unittest.TestCase):

    def test_it_is_the_projects_tick_rate_not_a_second_copy(self):
        sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))
        import k1_trace
        self.assertEqual(cal.K1_RDTIME_HZ, k1_trace.K1_RDTIME_HZ)

    def test_a_dispatch_that_ran_exactly_as_predicted_gets_1x(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t_trace.csv")
            write_trace(p, [("dronet", 0, 0, "conv2d_s8", 2.0, 2.0, 0.0)])
            rows, why = cal.read_trace(p)
            self.assertIsNone(why)
            self.assertAlmostEqual(rows[0][4] / rows[0][3], 1.0, places=3)


class QueueDelayIsNotSlowness(unittest.TestCase):

    def test_a_long_wait_does_not_change_the_multiplier(self):
        """Two dispatches that RAN identically but waited differently must calibrate
        identically -- otherwise the multiplier encodes the scheduler's own placement
        and the solver is charged twice for it."""
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a_trace.csv")
            b = os.path.join(d, "b_trace.csv")
            write_trace(a, [("dronet", 0, 0, "conv2d_s8", 2.0, 3.0, 0.0)])
            write_trace(b, [("dronet", 0, 0, "conv2d_s8", 2.0, 3.0, 40.0)])
            ra, _ = cal.read_trace(a)
            rb, _ = cal.read_trace(b)
            self.assertAlmostEqual(ra[0][4], rb[0][4], places=6)


class OneKeyPerDispatchNotPerInstance(unittest.TestCase):

    def test_instances_pool_into_one_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t_trace.csv")
            write_trace(p, [("ffn_block", i, 0, "linear_s8", 2.0, 2.0 * (1 + i), 0.0)
                            for i in range(4)])
            out = os.path.join(d, "cal.json")
            r = subprocess.run([sys.executable, _SCRIPT, "--trace", p, "--out", out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            got = json.load(open(out))
            self.assertEqual(list(got["per_dispatch_multiplier"]), ["ffn_block/0"])
            # mean of 1x, 2x, 3x, 4x
            self.assertAlmostEqual(got["per_dispatch_multiplier"]["ffn_block/0"],
                                   2.5, places=3)
            self.assertEqual(got["coverage"]["nets_exact"], ["ffn_block"])


class TheFloorProtectsThePooledTierOnly(unittest.TestCase):

    def test_a_tiny_dispatch_cannot_inflate_the_op_multiplier(self):
        """A 20x ratio on a dispatch far under the floor is timer granularity, and the
        op tier is what OTHER networks inherit -- so it must not carry it."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t_trace.csv")
            write_trace(p, [("mlp_control", 0, 0, "cast_i8_to_f16", 0.001, 0.020, 0.0),
                            ("ffn_block", 0, 1, "linear_s8", 4.0, 4.8, 0.0)])
            out = os.path.join(d, "cal.json")
            r = subprocess.run([sys.executable, _SCRIPT, "--trace", p, "--out", out],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            got = json.load(open(out))
            self.assertNotIn("cast_i8_to_f16", got["per_op_multiplier"])
            self.assertAlmostEqual(got["per_op_multiplier"]["linear_s8"], 1.2, places=2)
            # ... but the per-dispatch tier keeps it, where it is compared only with
            # other measurements of the same dispatch.
            self.assertIn("mlp_control/0", got["per_dispatch_multiplier"])
            self.assertAlmostEqual(got["aggregate_multiplier"], 1.2, places=2)


class ValidationReportsDisagreement(unittest.TestCase):

    def test_an_off_key_is_named_not_smoothed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t_trace.csv")
            write_trace(p, [("dronet", 0, 0, "conv2d_s8", 2.0, 4.0, 0.0)])
            ref = os.path.join(d, "ref.json")
            json.dump({"per_dispatch_multiplier": {"dronet/0": 1.0},
                       "per_op_multiplier": {}, "aggregate_multiplier": 1.0},
                      open(ref, "w"))
            out = os.path.join(d, "cal.json")
            r = subprocess.run([sys.executable, _SCRIPT, "--trace", p, "--out", out,
                                "--validate-against", ref],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            v = json.load(open(out))["validation"]
            self.assertEqual(v["per_dispatch_multiplier"]["n_off"], 1)
            self.assertEqual(v["per_dispatch_multiplier"]["off"][0]["key"], "dronet/0")

    def test_a_trace_missing_columns_is_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad_trace.csv")
            with open(p, "w", newline="") as fh:
                csv.writer(fh).writerow(["network", "dispatch_id"])
            rows, why = cal.read_trace(p)
            self.assertIsNone(rows)
            self.assertIn("missing columns", why)


class TheRealTableReproduces(unittest.TestCase):
    """The reconstruction is checked against the committed table, not merely run.

    Skipped when the traces are not in the checkout, so this stays useful on a thin
    clone without pretending to have verified anything.
    """

    def test_forty_of_the_forty_eight_committed_keys_come_back(self):
        import glob
        traces = (glob.glob(os.path.join(
                      _REPO, "results/k1_feedback_exact/board_runs*/original_*_trace.csv"))
                  + glob.glob(os.path.join(
                      _REPO, "results/k1_feedback_exact/board_runs*/feedback_*_trace.csv")))
        ref = os.path.join(_REPO, "results/codesign_feedback/k1_board_calibration.json")
        if len(traces) < 60 or not os.path.exists(ref):
            self.skipTest("board traces or the committed calibration are not in this "
                          "checkout")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "cal.json")
            cmd = [sys.executable, _SCRIPT, "--out", out, "--validate-against", ref]
            for t in traces:
                cmd += ["--trace", t]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            v = json.load(open(out))["validation"]["per_dispatch_multiplier"]
            self.assertEqual(v["n_reference"], 48)
            self.assertEqual(v["reference_keys_absent_here"], [],
                             "the reconstruction must produce the same KEY SET")
            self.assertGreaterEqual(v["n_within_tol"], 40)


if __name__ == "__main__":
    unittest.main()
