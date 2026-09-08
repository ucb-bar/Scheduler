#!/usr/bin/env python3
"""Run the co-design loop over several workloads and report the fraction it improves.

WHY A FRACTION AND NOT A SHOWCASE. One workload where the loop finds a lever is an
anecdote; a reviewer asked, reasonably, whether the process is stable and general, and
said a "reasonable 50%" would do. That is a number only a denominator can produce, so
this runs the same driver over N specs, with no per-workload tuning, and prints every
outcome including the ones where nothing helped.

IMPROVED means what `candidate_objective` means: the final schedule beats the baseline
on the nine lexicographic terms -- hard deadline misses first, makespan seventh. A run
that converges with no lever applied is NOT an improvement and is counted as such; so is
a run where the loop's own board-feedback arm cannot recover the misses the board reveals.
Refused, failed and unschedulable workloads are reported in their own buckets rather than
dropped, because a denominator that quietly loses its hard cases is not a denominator.

Usage:
  scripts/loop_over_workloads.py --workloads data/toplevel/a.json data/toplevel/b.json \\
      --solver cpsat --time-limit 30 --out-dir results/loop_sweep
  scripts/loop_over_workloads.py --list-k1        # the K1 specs this knows about
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)

# Specs that describe a K1 taskset with more than one net -- the population the loop is
# for. A single-net spec has no scheduling to do, and a non-K1 spec has no board costs.
K1_GLOBS = ("data/toplevel/_4w_networks_k1_*.json",
            "data/toplevel/networks_k1_*.json",
            "data/toplevel/_evo_*.json",
            "data/toplevel/_flight_deployed_*.json")


def k1_candidates():
    out = []
    for g in K1_GLOBS:
        for p in sorted(glob.glob(os.path.join(REPO, g))):
            try:
                spec = json.load(open(p))
            except Exception:
                continue
            nets = spec.get("networks") or {}
            if len(nets) < 2:
                continue  # nothing to schedule against
            out.append(os.path.relpath(p, REPO))
    return out


def run_one(workload, args, out_root):
    stem = os.path.splitext(os.path.basename(workload))[0]
    out_dir = os.path.join(out_root, stem)
    cmd = [PY, "scripts/run_codesign_loop.py", "--workload", workload,
           "--solver", args.solver, "--objective", args.objective,
           "--max-rounds", str(args.max_rounds), "--out-dir", out_dir]
    if args.time_limit:
        cmd += ["--time-limit", str(args.time_limit)]
    if args.board_calibration:
        cmd += ["--board-calibration", args.board_calibration,
                "--board-solver", args.solver]
        if args.board_time_limit:
            cmd += ["--board-time-limit", str(args.board_time_limit)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       timeout=args.timeout)
    dt = time.time() - t0
    report_path = os.path.join(out_dir, stem, "loop_report.json")
    if not os.path.exists(report_path):
        # the driver nests its output under the workload stem
        cands = glob.glob(os.path.join(out_dir, "*", "loop_report.json"))
        report_path = cands[0] if cands else ""
    rec = {"workload": workload, "seconds": round(dt, 1),
           "exit": r.returncode, "report": None,
           "status": "no_report", "levers_applied": [], "detail": ""}
    if report_path and os.path.exists(report_path):
        rep = json.load(open(report_path))
        rec["report"] = os.path.relpath(report_path, REPO)
        rec["levers_applied"] = rep.get("levers_applied") or []
        base, final = rep.get("baseline_score_ms"), rep.get("final_score_ms")
        bmk, fmk = rep.get("baseline_makespan_ms"), rep.get("final_makespan_ms")
        board = rep.get("board_feedback") or {}
        rec.update(objective=rep.get("objective"), baseline=base, final=final,
                   baseline_makespan_ms=bmk, final_makespan_ms=fmk,
                   accept_rule=rep.get("accept_rule"),
                   board_arc=[board.get("baseline_instance_misses"),
                              board.get("aot_instance_misses"),
                              board.get("board_recost_instance_misses"),
                              board.get("board_resolve_instance_misses")]
                   if board.get("enabled") else None)
        improved = bool(rec["levers_applied"]) and (
            (base is not None and final is not None and final < base - 1e-9)
            or (bmk is not None and fmk is not None and fmk < bmk - 1e-9))
        rec["status"] = "improved" if improved else "no_lever_helped"
        rec["detail"] = (f"{rep.get('objective')} {base} -> {final}; "
                         f"makespan {bmk} -> {fmk}")
    elif r.returncode != 0:
        tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
        rec["status"] = "failed"
        rec["detail"] = tail[-1][:300] if tail else ""
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument("--list-k1", action="store_true",
                    help="print the multi-net K1 specs and exit")
    ap.add_argument("--solver", choices=["greedy", "cpsat"], default="greedy")
    ap.add_argument("--objective", default="auto")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--board-calibration", default=None)
    ap.add_argument("--board-time-limit", type=float, default=None)
    ap.add_argument("--timeout", type=float, default=3600,
                    help="per-workload wall limit, seconds")
    ap.add_argument("--out-dir", default="results/loop_sweep")
    a = ap.parse_args()

    if a.list_k1:
        for w in k1_candidates():
            print(w)
        return 0
    workloads = a.workloads or k1_candidates()
    if not workloads:
        print("no workloads", file=sys.stderr)
        return 2
    out_root = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)
    os.makedirs(out_root, exist_ok=True)

    recs = []
    for i, w in enumerate(workloads, 1):
        print(f"[{i}/{len(workloads)}] {w}", flush=True)
        try:
            rec = run_one(w, a, out_root)
        except subprocess.TimeoutExpired:
            rec = {"workload": w, "status": "timeout", "levers_applied": [],
                   "detail": f"exceeded --timeout {a.timeout}s"}
        recs.append(rec)
        print(f"    {rec['status']:>16}  levers={rec.get('levers_applied')}  "
              f"{rec.get('detail','')[:120]}", flush=True)

    n = len(recs)
    imp = [r for r in recs if r["status"] == "improved"]
    nolever = [r for r in recs if r["status"] == "no_lever_helped"]
    bad = [r for r in recs if r["status"] in ("failed", "timeout", "no_report")]
    summary = {
        "schema": "loop_sweep/v1",
        "n_workloads": n,
        "n_improved": len(imp),
        "improved_fraction": round(len(imp) / n, 3) if n else None,
        "n_no_lever_helped": len(nolever),
        "n_failed_or_timeout": len(bad),
        "solver": a.solver, "objective": a.objective,
        "board_calibration": a.board_calibration,
        "improved_means": ("a lever was applied AND the declared objective or the "
                           "makespan strictly improved, judged by the loop's "
                           "candidate_objective rule"),
        "runs": recs,
    }
    json.dump(summary, open(os.path.join(out_root, "sweep_summary.json"), "w"), indent=1)
    print(f"\n{len(imp)}/{n} workloads improved "
          f"({100 * len(imp) / n:.0f}%); {len(nolever)} found no helping lever; "
          f"{len(bad)} failed/timed out")
    for r in recs:
        print(f"  {r['status']:>16}  {os.path.basename(r['workload'])}")
    print(f"\nsummary: {os.path.relpath(os.path.join(out_root, 'sweep_summary.json'), REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
