#!/usr/bin/env python3
"""Build a 2->5 network ladder where scheduling actually decides the outcome.

WHY THESE AND NOT THE EXISTING SPECS. The 24 runnable K1 specs are bimodal. Twelve have
a baseline that meets every deadline (density ~0.25 -- nothing is at stake, and the loop
can only win on terms nobody reads), and most of the rest ask for something no schedule
can deliver: `yolov8_nano_64x96` needs 23.95 ms at 8 cores against a 22 ms window, and
its core scaling has saturated (4->8 cores buys 1.6%), so 44 of the 50 residual misses in
the full ablation were infeasible by construction. Almost nothing sits in the band where
a scheduler decides whether the deadline is met, which is the band the loop exists for.

THE TENSION THIS LADDER BUILDS ON. Sharding buys wall time and costs efficiency: on the
board `ffn_block` goes 26.61 -> 7.72 ms across 1 -> 8 cores, a 3.45x speedup that consumes
61.8 core-ms instead of 26.61. So a solver cannot shard everything -- past a point the
core budget will not carry it -- and as networks accumulate it has to choose WHICH nets
to widen. That is a real scheduling decision with a right and a wrong answer, and it is
where a list heuristic should start to lose to an exact solver.

Every workload here is built so that:
  * the baseline (one core per net, RVV singletons) MISSES at least one deadline, and
  * a schedule that meets every deadline EXISTS using implementations already measured
    on the board -- so a failure is the loop's, not physics.

Measured whole-net board times (gen/profile_mb, rvv_x60, source=k1) used to size the
windows:

    net                  1 core    best        speedup   core-ms at best
    mlp_control            0.08     0.08 (1c)    1.00x      0.08
    attn_block             0.13     0.13 (1c)    1.00x      0.13
    fused_full             3.62     3.62 (1c)    1.00x      3.62
    dronet                 8.33     5.25 (4c)    1.59x     21.0
    ffn_block             26.61     7.72 (8c)    3.45x     61.8
    yolov8_nano_64x96     47.73    23.95 (8c)    1.99x    191.6

Usage:
  scripts/make_scaling_workloads.py --out-dir data/toplevel/scaling [--check]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPH = ("gen_mb/vmfb/{net}/spacemit_x60/rvv_x60/{net}.int8/"
         "{net}.int8_dispatch_graph.json")

HARDWARE = {
    "machines": {"cpu_p": 4, "cpu_e": 4},
    "profile_hw": {"cpu_p": "rvv_x60", "cpu_e": "rvv_x60"},
    # topo_tag_override MUST be false for sharding: with it true every combination is
    # costed from topo_0, so a 4-hart block is charged the single-hart time while
    # occupying four harts, and the failure is silent.
    "profile": {"target": "spacemit_x60", "topo_tag_override": False,
                "gen_root": "gen_mb"},
    "p_core_speedup": 1.0,
}
SCHEDULER = {"random_seed": 42, "solver_verbosity": 1, "time_limit": 300,
             "use_profiled": True, "prune_periodic": True,
             "restrict_makespan_to_nonperiodic": False,
             # The loop's levers set machine_combination_mode / enable_impls per
             # candidate; the baseline must start with neither, or cell A is not a
             # baseline.
             "machine_combination_mode": "singletons", "enable_impls": False}

#: net -> (period_ms, window_ms, num_instances). Windows chosen so the SINGLETON cost
#: overruns and a measured wider implementation fits.
LADDER = {
    # 2 nets: one net must widen. ffn 26.61 singleton > 10 ms window; 4c gives 9.71.
    "w2_ffn_tight": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
    },
    # 3 nets: TWO nets must widen and they compete for the same 8 cores.
    # dronet 8.33 > 7 ms window; 2c gives 6.05.
    "w3_ffn_dronet": {
        "mlp_control": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
        "dronet": (12.0, 7.0, 5),
    },
    # 4 nets: add steady periodic load that fragments availability (fused_full is
    # 0.72 cores of continuous work at a 5 ms period) without needing to widen.
    "w4_ffn_dronet_sensor": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
        "dronet": (12.0, 7.0, 5),
    },
    # 5 nets: THREE nets must widen, and the widest choice everywhere does NOT fit --
    # ffn at 8c is 61.8 core-ms against a 12 ms period, 5.15 of 8 cores on its own. The
    # solver has to pick widths per net rather than take the fastest implementation,
    # which is the decision a list heuristic makes locally and an exact solver makes
    # globally.
    "w5_ffn_dronet_yolo": {
        "mlp_control": (5.0, 5.0, 12),
        "fused_full": (5.0, 5.0, 12),
        "ffn_block": (12.0, 10.0, 5),
        "dronet": (12.0, 7.0, 5),
        "yolov8_nano_64x96": (40.0, 26.0, 1),
    },
}


def net_times(net):
    """`{n_cores: whole-net ms}` measured on the board, from the committed profiles."""
    out = {}
    for c in glob.glob(os.path.join(
            REPO, f"gen/profile_mb/rvv_x60/spacemit_x60/{net}/*/*/topo_*/results.csv")):
        width = len(os.path.basename(os.path.dirname(c)).split("_")) - 1
        t = sum(float(r["mean_time"] or 0) for r in csv.DictReader(open(c)))
        if t > 0:
            out[width] = min(out.get(width, 9e9), t)
    return out


def check(name, nets, log):
    """Is this workload in the band? Baseline must miss; a fix must exist; and the
    core budget at the widths a fix needs must still carry it."""
    ok = True
    singleton_util = 0.0
    widened_util = 0.0
    n_miss_baseline = 0
    for net, (period, window, _inst) in nets.items():
        t = net_times(net)
        if not t:
            log(f"    {net}: NO measured profile — cannot use this net")
            return False
        one = t.get(1)
        best_w = min(t, key=lambda w: t[w])
        best = t[best_w]
        singleton_util += one / period
        if one > window:
            n_miss_baseline += 1
            # the fix, and what it costs in cores
            fits = [w for w in sorted(t) if t[w] <= window]
            if not fits:
                log(f"    {net}: INFEASIBLE — needs {best:.2f} ms at {best_w}c "
                    f"against a {window:.1f} ms window ({best / window:.2f}x)")
                ok = False
                continue
            w = fits[0]
            widened_util += (w * t[w]) / period
            log(f"    {net}: singleton {one:.2f} > window {window:.1f} -> must widen; "
                f"{w}c gives {t[w]:.2f} ms, costing {(w * t[w]) / period:.2f} cores")
        else:
            widened_util += one / period
    if n_miss_baseline == 0:
        log("    NOTHING AT STAKE: every net fits on one core; the baseline cannot miss")
        ok = False
    log(f"    baseline utilisation {singleton_util:.2f} cores; "
        f"with the required widening {widened_util:.2f} of 8")
    if widened_util > 8.0:
        log("    OVER CAPACITY even at the required widths — no schedule can exist")
        ok = False
    elif widened_util > 6.5:
        log("    tight: the solver must choose widths rather than take the widest")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/toplevel/scaling")
    ap.add_argument("--check", action="store_true",
                    help="only report the feasibility band analysis; write nothing")
    a = ap.parse_args()
    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)

    def log(s):
        print(s, flush=True)

    written, bad = [], []
    for name, nets in LADDER.items():
        log(f"== {name} ({len(nets)} nets)")
        good = check(name, nets, log)
        if not good:
            bad.append(name)
            log("    -> NOT in the band; not written")
            continue
        spec = {
            "_comment": (f"{name}: {len(nets)}-net K1 ladder rung. Windows sized from "
                         f"MEASURED board times so the singleton baseline misses and a "
                         f"measured wider implementation fits. Generated by "
                         f"scripts/make_scaling_workloads.py -- do not hand-edit."),
            "hardware": json.loads(json.dumps(HARDWARE)),
            "scheduler": json.loads(json.dumps(SCHEDULER)),
            "networks": {}, "edges": [],
        }
        for i, (net, (period, window, inst)) in enumerate(nets.items()):
            spec["networks"][net] = {
                "id": i, "identifier": net,
                "dispatch_deps_path": GRAPH.format(net=net),
                "period": period, "window_duration": window, "num_instances": inst,
            }
        if not a.check:
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, f"{name}.json")
            json.dump(spec, open(p, "w"), indent=1)
            written.append(os.path.relpath(p, REPO))
            log(f"    -> wrote {os.path.relpath(p, REPO)}")
        else:
            log("    -> in the band (--check: not written)")
    log(f"\n{len(LADDER) - len(bad)}/{len(LADDER)} rungs in the band"
        + (f"; rejected {bad}" if bad else ""))
    for p in written:
        log(f"  {p}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
