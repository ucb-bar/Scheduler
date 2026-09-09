"""Invariants of the inner/outer ablation that a reviewer's objection would land on.

Three things make the difference between an ablation and a table of numbers, and none of
them are visible in the output once they go wrong:

1. the population is FAMILIES, not spec files, because the corpus contains
   byte-identical duplicates and variants that return bit-identical results;
2. the stratum is PRE-REGISTERED (cell A misses on board costs), because a stratum
   chosen after seeing results is indistinguishable from cherry-picking;
3. cells that scheduled different amounts of work are REFUSED rather than ranked.

These pin all three. They are cheap and need no solver.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _load():
    """Import the script by path; `scripts/` is not a package."""
    path = os.path.join(_REPO, "scripts", "ablate_feedback_loops.py")
    spec = importlib.util.spec_from_file_location("ablate_feedback_loops", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ablate_feedback_loops"] = mod
    spec.loader.exec_module(mod)
    return mod


abl = _load()


class FamilyIsTheUnit(unittest.TestCase):

    def test_the_byte_identical_pair_shares_a_family(self):
        """_evo_og.json and networks_k1_mb_3model_4hz_yolo_ctrl.json have the same
        md5. Counting them as two workloads inflates n for nothing."""
        self.assertEqual(abl.family_of("_evo_og"),
                         abl.family_of("networks_k1_mb_3model_4hz_yolo_ctrl"))

    def test_variants_that_returned_identical_results_share_a_family(self):
        """flight_deployed / _noshard / _singletons differ only in
        machine_combination_mode and returned 122.600 -> 110.083 with board arc
        [5,0,7,5] in all three cases."""
        fams = {abl.family_of(s) for s in ("networks_k1_flight_deployed",
                                           "networks_k1_flight_deployed_noshard",
                                           "networks_k1_flight_deployed_singletons")}
        self.assertEqual(len(fams), 1, f"expected one family, got {fams}")

    def test_every_family_member_is_classified(self):
        for fam, members in abl.FAMILIES.items():
            for m in members:
                self.assertEqual(abl.family_of(m), fam)

    def test_unknown_spec_is_not_silently_absorbed(self):
        self.assertEqual(abl.family_of("something_new"), "unclassified")

    def test_the_broken_spec_is_named(self):
        """networks_k1_mlp_dronet points at a dead gen/vmfb/.../RVV/ tree. It must be
        skipped by name, not counted as a workload the loop failed to improve."""
        self.assertIn("networks_k1_mlp_dronet", abl.UNRUNNABLE)


class ComparabilityIsRefusedNotRanked(unittest.TestCase):

    def test_equal_instance_counts_are_comparable(self):
        cells = {"A": {"instances": {"mlp": 5, "yolo": 1}},
                 "B": {"instances": {"mlp": 5, "yolo": 1}}}
        self.assertEqual(abl.comparability_of(cells)["status"], "ok")

    def test_different_instance_counts_are_refused(self):
        """The 4 Hz baseline was once re-solved without --max-periodic-iters 1, the
        loop grew mlp_control from 32 instances to 91, and the result sat on disk under
        the baseline's name. Every term still computed."""
        cells = {"A": {"instances": {"mlp": 32}},
                 "B": {"instances": {"mlp": 91}}}
        v = abl.comparability_of(cells)
        self.assertEqual(v["status"], "REFUSED")
        self.assertIn("amounts of work", v["why"])

    def test_a_cell_that_failed_to_solve_does_not_make_the_rest_incomparable(self):
        cells = {"A": {"instances": {"mlp": 5}},
                 "B": {"status": "solve_failed"},
                 "C": {"instances": {"mlp": 5}}}
        self.assertEqual(abl.comparability_of(cells)["status"], "ok")


class FairnessSwitchesAreForced(unittest.TestCase):

    def test_compaction_is_forced_off(self):
        """Compaction is idempotent on tight solvers and slack-eliminating on list
        schedulers, and the greedy family bypasses the registry so it never gets the
        post-pass at all. Leaving it on changes which arm wins."""
        self.assertEqual(abl.SOLVE_ENV.get("XPURT_NO_COMPACT"), "1")

    def test_cpsat_workers_are_pinned_for_the_run(self):
        self.assertIn("XPURT_CPSAT_WORKERS", abl.SOLVE_ENV)

    def test_missing_solver_is_distinguished_from_a_broken_one(self):
        self.assertEqual(abl.classify_failure(
            "ModuleNotFoundError: No module named 'ortools'"), "unavailable")
        self.assertEqual(abl.classify_failure("MSK_RES_ERR_LICENSE_EXPIRED"),
                         "unavailable")
        self.assertEqual(abl.classify_failure("ValueError: infeasible"), "error")
        self.assertEqual(abl.classify_failure("subprocess timed out"), "timeout")


class CellsMeanWhatTheySay(unittest.TestCase):

    def test_the_four_cells_are_the_two_by_two(self):
        got = {name: (inner, outer) for name, inner, outer, _label in abl.CELLS}
        self.assertEqual(got, {"A": (False, False), "B": (True, False),
                               "C": (False, True), "D": (True, True)})


if __name__ == "__main__":
    unittest.main()
