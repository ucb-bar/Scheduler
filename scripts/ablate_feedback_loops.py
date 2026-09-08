#!/usr/bin/env python3
"""Ablate the two feedback loops separately: inner (offline search) and outer (measured).

THE CLAIM THIS IS FOR. The paper says the channel closes at two distances -- an inner
loop that runs entirely offline (the scheduler asks the compiler for a different
implementation and re-solves) and an outer loop that closes through measurement (the
board returns what it actually cost, and the scheduler re-solves against that). Showing
one workload where the pair helps does not separate them, and does not say whether
either alone would have done. This runs the 2x2.

    cell   inner (levers)   outer (solve on measured costs)
    ----   --------------   -------------------------------
    A      no               no                (the naive deployment)
    B      YES              no                (offline co-design, deployed blind)
    C      no               YES               (measure-and-re-solve, no co-design)
    D      YES              YES               (both)

EVERY CELL IS SCORED ON BOARD COSTS, because that is what silicon does. The cells differ
in what the SCHEDULER KNEW, not in how they are judged:

  * A and B are solved against predicted costs and then RE-COST on the measured board
    multipliers with their assignment held fixed -- exactly what deploying them means.
  * C and D are solved with --board-calibration, so their own costs are already the
    board's; re-costing them again would apply the multiplier twice.

Scoring is instance-level (`xpu-rt/schedule_eval.py`): an instance misses when its last
dispatch ends past `inst*period + window`. That is the number the figures use.

Usage:
  scripts/ablate_feedback_loops.py --workloads data/toplevel/a.json ... \\
      --calibration results/codesign_feedback/k1_board_calibration.json \\
      --solver greedy --out-dir results/loop_ablation
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)

import schedule_eval  # noqa: E402

CELLS = (("A", False, False, "neither"),
         ("B", True, False, "inner only (levers, deployed blind)"),
         ("C", False, True, "outer only (measured re-solve, no levers)"),
         ("D", True, True, "both"))


def sh(cmd, timeout=None):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          timeout=timeout)


def solve(spec_path, solver, time_limit, calibration, tag, log):
    """Solve one spec, optionally with the measured board multipliers."""
    stem = os.path.splitext(os.path.basename(spec_path))[0]
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec_path,
           "--profiled", "--max-periodic-iters", "1"]
    sfx = "greedy_profiled"
    if solver == "cpsat":
        cmd += ["--solver", "milp", "--scheduler", "cpsat"]
        sfx = "cpsat_profiled"
    else:
        cmd += ["--solver", "greedy"]
    if time_limit:
        cmd += ["--time-limit", str(time_limit)]
    if calibration:
        cmd += ["--board-calibration", calibration]
    r = sh(cmd)
    sched = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}.json")
    if not os.path.exists(sched):
        log(f"    {tag}: SOLVE FAILED — {((r.stderr or '') + (r.stdout or ''))[-300:]}")
        return None
    return sched


def recost(sched, spec_path, calibration, out, log):
    """Re-cost a schedule under measured board costs with its assignment held fixed.

    This is what deploying a predicted-cost schedule means: the placement is already
    decided, and every dispatch takes what the board makes it take.
    """
    r = sh([PY, "scripts/recost_schedule_on_board.py", "--schedule", sched,
            "--spec", spec_path, "--calibration", calibration, "--out", out])
    if r.returncode != 0 or not os.path.exists(out):
        log(f"    recost failed: {((r.stderr or '') + (r.stdout or ''))[-300:]}")
        return None
    return out


def inner_search(workload, args, out_dir, log):
    """Run the loop's offline lever search; return the spec it converged on."""
    stem = os.path.splitext(os.path.basename(workload))[0]
    loop_out = os.path.join(out_dir, "inner", stem)
    cmd = [PY, "scripts/run_codesign_loop.py", "--workload", workload,
           "--solver", args.solver, "--max-rounds", str(args.max_rounds),
           "--objective", args.objective, "--out-dir", loop_out]
    if args.time_limit:
        cmd += ["--time-limit", str(args.time_limit)]
    if args.replay:
        cmd += ["--replay"]
    r = sh(cmd, timeout=args.timeout)
    rep_paths = glob.glob(os.path.join(loop_out, "*", "loop_report.json"))
    if not rep_paths:
        log(f"    inner search produced no report "
            f"({((r.stderr or '') + (r.stdout or ''))[-200:]})")
        return None, []
    rep = json.load(open(rep_paths[0]))
    levers = rep.get("levers_applied") or []
    # The loop writes every candidate spec it solved; the accepted one for the last
    # round is the converged spec. Prefer the recorded path, fall back to the newest.
    specs = sorted(glob.glob(os.path.join(loop_out, "*", "specs", "*.json")),
                   key=os.path.getmtime)
    final = None
    if levers:
        want = f"_{levers[-1]}.json"
        for p in reversed(specs):
            if p.endswith(want):
                final = p
                break
    if final is None:
        final = next((p for p in specs if p.endswith("_r0_baseline.json")), None)
    return final, levers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", required=True)
    ap.add_argument("--calibration",
                    default="results/codesign_feedback/k1_board_calibration.json")
    ap.add_argument("--solver", choices=["greedy", "cpsat"], default="greedy")
    ap.add_argument("--objective", default="auto")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--replay", action="store_true", default=True,
                    help="(default) run the inner search deterministically")
    ap.add_argument("--no-replay", dest="replay", action="store_false")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out-dir", default="results/loop_ablation")
    a = ap.parse_args()

    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cal = a.calibration if os.path.isabs(a.calibration) else os.path.join(
        REPO, a.calibration)
    if not os.path.exists(cal):
        print(f"no calibration at {cal}", file=sys.stderr)
        return 2
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    rows = []
    for i, w in enumerate(a.workloads, 1):
        stem = os.path.splitext(os.path.basename(w))[0]
        log(f"[{i}/{len(a.workloads)}] {stem}")
        t0 = time.time()
        base_spec = w if os.path.isabs(w) else os.path.join(REPO, w)
        opt_spec, levers = inner_search(w, a, out_dir, log)
        if opt_spec is None:
            rows.append({"workload": w, "status": "inner_search_failed"})
            continue
        log(f"    inner search: levers {levers or '(none)'}")
        spec_for = {False: base_spec, True: opt_spec}
        cells = {}
        for name, inner, outer, label in CELLS:
            sp = spec_for[inner]
            if inner and not levers:
                # No lever was accepted, so B and D are A and C by construction. Say so
                # rather than reporting a "second" measurement of the same schedule.
                cells[name] = {"label": label, "inner": inner, "outer": outer,
                               "note": "no lever accepted; identical to the "
                                       "inner-off cell"}
            sched = solve(sp, a.solver, a.time_limit, cal if outer else None,
                          f"{name}", log)
            if sched is None:
                cells.setdefault(name, {}).update(
                    {"label": label, "inner": inner, "outer": outer,
                     "status": "solve_failed"})
                continue
            spec_obj = json.load(open(sp))
            if outer:
                # already solved on board costs
                scored, how = sched, "solved on measured costs"
            else:
                out = os.path.join(out_dir, f"{stem}_{name}_recost.json")
                scored = recost(sched, sp, cal, out, log)
                how = "solved on predicted costs, then re-cost on measured"
                if scored is None:
                    cells.setdefault(name, {}).update({"status": "recost_failed"})
                    continue
            s = schedule_eval.summary(scored, spec_obj)
            cells.setdefault(name, {}).update(
                {"label": label, "inner": inner, "outer": outer, "scored_how": how,
                 "spec": os.path.relpath(sp, REPO),
                 "schedule": os.path.relpath(scored, REPO), **s})
            log(f"    {name} {label:<44} misses={s['instance_misses']:<3} "
                f"worst_late={(s['worst_lateness_ms'] or 0):8.3f} ms  "
                f"makespan={(s['makespan_ms'] or 0):7.2f} ms")
        rows.append({"workload": w, "status": "ok", "levers": levers,
                     "seconds": round(time.time() - t0, 1), "cells": cells})

    # ---- aggregate -----------------------------------------------------------
    ok = [r for r in rows if r.get("status") == "ok"]

    def vals(cell, key):
        out = []
        for r in ok:
            c = (r["cells"] or {}).get(cell) or {}
            v = c.get(key)
            if isinstance(v, (int, float)):
                out.append(v)
        return out

    agg = {}
    for name, inner, outer, label in CELLS:
        m = vals(name, "instance_misses")
        agg[name] = {
            "label": label,
            "n": len(m),
            "workloads_with_zero_misses": sum(1 for x in m if x == 0),
            "total_instance_misses": sum(m),
            "median_instance_misses": statistics.median(m) if m else None,
            "median_worst_lateness_ms": (statistics.median(vals(name, "worst_lateness_ms"))
                                         if vals(name, "worst_lateness_ms") else None),
            "median_makespan_ms": (statistics.median(vals(name, "makespan_ms"))
                                   if vals(name, "makespan_ms") else None),
        }
    only_both = []
    for r in ok:
        c = r["cells"]
        try:
            if (c["D"]["instance_misses"] == 0 and c["B"]["instance_misses"] > 0
                    and c["C"]["instance_misses"] > 0):
                only_both.append(os.path.basename(r["workload"]))
        except (KeyError, TypeError):
            continue

    summary = {
        "schema": "loop_ablation/v1",
        "cells": {n: l for n, _i, _o, l in CELLS},
        "scoring": ("instance-level misses on BOARD costs for every cell; A/B are "
                    "solved on predicted costs then re-cost with the assignment fixed, "
                    "C/D are solved with --board-calibration"),
        "solver": a.solver, "calibration": os.path.relpath(cal, REPO),
        "n_workloads": len(rows), "n_ok": len(ok),
        "aggregate": agg,
        "workloads_only_both_clears": only_both,
        "runs": rows,
    }
    json.dump(summary, open(os.path.join(out_dir, "ablation_summary.json"), "w"),
              indent=1)
    open(os.path.join(out_dir, "ablation.log"), "w").write("\n".join(lines) + "\n")

    log("\n=== aggregate over %d workload(s), scored on board costs ===" % len(ok))
    log(f"{'cell':<5}{'inner':<7}{'outer':<7}{'0-miss':<8}{'tot miss':<10}"
        f"{'med worst late':<16}{'med makespan':<13}")
    for name, inner, outer, label in CELLS:
        g = agg[name]
        log(f"{name:<5}{str(inner):<7}{str(outer):<7}"
            f"{g['workloads_with_zero_misses']}/{g['n']:<6}"
            f"{g['total_instance_misses']:<10}"
            f"{(g['median_worst_lateness_ms'] or 0):<16.3f}"
            f"{(g['median_makespan_ms'] or 0):<13.2f}")
    if only_both:
        log(f"\nworkloads only the PAIR clears (B>0, C>0, D==0): {only_both}")
    log(f"\nsummary: {os.path.relpath(os.path.join(out_dir, 'ablation_summary.json'), REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
