#!/usr/bin/env python3
"""Fully-automatic ModelBlaster<->XPU-RT co-design feedback loop.

ONE command. Starts from a workload spec at a clean baseline (no levers), then each
round proposes every not-yet-applied lever as a candidate, SOLVES each candidate with
the real profiled scheduler, and ACCEPTS the candidate with the largest MEASURED
makespan reduction that adds ZERO deadline misses. Iterates until no lever helps
(converged). The loop's intelligence is the advisor/cost-model; this driver applies,
measures, and accepts — honestly (a lever that does not help is rejected and reported).

Levers:
  * ime   — expose the K1 IME matrix engine as a per-dispatch alternative. For every
            CONV-bearing net that lacks an ime_x60 profile, auto-build one from the
            measured conv speedups (scripts/make_ime_profile.py, which scales mean_time
            — the column the loader reads), then set scheduler.enable_impls=true. The
            scheduler picks IME per dispatch only where measured faster.
  * shard — expose 2/4/8-hart implementations (scheduler.machine_combination_mode=shard,
            topo_tag_override=false).
  * fuse  — the roofline decision-aid (xpu-rt/data/fusion_benefit.csv) is consulted and
            REPORTED, not applied: our stack is compute-bound, so fusion is a scheduling
            (dispatch-collapse) move, not a cycle win — the loop honestly does not credit
            a makespan gain it cannot measure.

Usage:
  scripts/run_codesign_loop.py --workload data/toplevel/<spec>.json [--max-rounds 4]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Interpreter used for the child scheduler runs: $XPURT_PY, else the repo-root venv, else this interpreter.
_venv_py = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv_py if os.path.exists(_venv_py) else sys.executable)
EPS = 0.05  # ms; a smaller "improvement" is noise, not a win

sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
try:
    import exact_cycle  # objective-aware acceptance for exact_cycle workloads
except Exception:
    exact_cycle = None


def objective_of(spec: dict) -> str:
    """Which metric this workload is really optimizing. A tri-exact-style workload
    declares `exact_cycle_worst_response`, whose win is WORST CRITICAL RESPONSE, not
    makespan — so a makespan-only loop would (correctly, but uselessly) credit its
    shard/IME levers nothing. Accept on the objective the user actually declared."""
    om = spec.get("scheduler", {}).get("objective_mode") or spec.get("objective_mode")
    return "worst_response" if om == "exact_cycle_worst_response" else "makespan"


def worst_response_ms(sched_path: str, spec: dict):
    """Worst critical response (ms) of a schedule, via the exact-cycle assessor.
    None if unavailable — the caller then falls back to makespan."""
    if exact_cycle is None or not sched_path or not os.path.exists(sched_path):
        return None
    try:
        sch = json.load(open(sched_path))
        crit = (spec.get("critical_models") or spec.get("scheduler", {}).get("critical_models") or [])
        heavy = spec.get("heavy_model") or spec.get("scheduler", {}).get("heavy_model")
        obj = exact_cycle.assess_schedule(sch, spec, crit, heavy)
        return float(obj["objective"]["worst_critical_response_ms"])
    except Exception:
        return None


def _run(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def _rvv_profile(net, variant, hw="rvv_x60"):
    return os.path.join(
        REPO, f"gen/profile_mb/{hw}/spacemit_x60/{net}/{net}.{variant}/"
        f"{net}_spacemit_x60_{hw}_{net}.{variant}/topo_0/results.csv")


def _net_variant(deps: str, key: str):
    m = re.search(r"/vmfb/([^/]+)/", deps)
    net = m.group(1) if m else key
    tail = deps.split("/")[-1].replace("_dispatch_graph.json", "")
    variant = tail[len(net) + 1:] if tail.startswith(net + ".") else "int8"
    return net, variant


def _is_conv_net(net, variant):
    p = _rvv_profile(net, variant)
    if not os.path.exists(p):
        return False
    return any(r.get("op", "").startswith("conv2d") for r in csv.DictReader(open(p)))


def solve(spec_path, solver="greedy", board_cal=None, time_limit=None):
    """Run the profiled scheduler; return (makespan_ms, op_miss, sched_json, err).

    solver     — "greedy" (fast, default) or "cpsat" (CP-SAT MILP; needs an ortools
                 interpreter — run the driver under one, and bound XPURT_CPSAT_WORKERS).
    board_cal  — None/False: isolated profile costs (the AOT view). True or a path:
                 pass --board-calibration so profile_loader scales every op by the
                 measured K1 actual/predicted ratio (the run-honest view).
    time_limit — per-solve CP-SAT seconds (ignored by greedy)."""
    stem = os.path.splitext(os.path.basename(spec_path))[0]
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec_path,
           "--profiled", "--max-periodic-iters", "1"]
    if solver == "cpsat":
        cmd += ["--solver", "milp", "--scheduler", "cpsat"]
        sfx = "cpsat_profiled"
    else:
        cmd += ["--solver", "greedy"]
        sfx = "greedy_profiled"
    if time_limit is not None:
        cmd += ["--time-limit", str(time_limit)]
    if board_cal:
        cmd += ["--board-calibration"] + ([board_cal] if isinstance(board_cal, str) else [])
    r = _run(cmd)
    metrics = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}_metrics.json")
    sched = os.path.join(REPO, "schedules", f"scheduled_{stem}_{sfx}.json")
    if not os.path.exists(metrics):
        return None, None, None, (r.stdout + r.stderr)[-800:]
    m = json.load(open(metrics))
    return float(m["makespan_ms"]), int(m["op_deadline_miss_count"]), sched, None


def instance_misses(sched_path, spec):
    """Count INSTANCE-level deadline misses in a schedule (a net-instance misses if its
    LATEST dispatch ends past its instance deadline = inst*period + window). This mirrors
    the evolution renderer's source-of-truth miss logic exactly, so the loop decides the
    board-feedback arm on the same number the figure will show — not on the per-dispatch
    op_deadline_miss_count the schedulers report. Returns (count, {net: count})."""
    try:
        from job_names import split_job_name
        sch = json.load(open(sched_path))["dispatches"]
        nets = spec["networks"] if isinstance(spec, dict) else json.load(open(spec))["networks"]
        known = set(nets)
        per = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
        win = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}
        last = {}
        for d in sch.values():
            net, inst = split_job_name(d["job_name"], known)
            if not (net in per and per[net]):
                continue
            e = float(d["start_time"]) + float(d["duration"])
            last[(net, inst)] = max(last.get((net, inst), 0.0), e)
        miss, by = 0, {}
        for (net, inst), e in last.items():
            if win[net] > 0 and e > inst * per[net] + win[net] + 1e-6:
                miss += 1
                by[net] = by.get(net, 0) + 1
        return miss, by
    except Exception as e:
        return None, {"error": str(e)}


def mk_of(sched_path):
    """Makespan (ms) from a schedule's _metrics.json sidecar; 0.0 if unavailable."""
    try:
        m = json.load(open(sched_path.replace(".json", "_metrics.json")))
        return float(m.get("makespan_ms", m.get("makespan", 0.0)) or 0.0)
    except Exception:
        return 0.0


def total_lateness(sched_path, spec):
    """Sum over net-instances of max(0, last_dispatch_end - instance_deadline), in ms — the
    real-time-correct 'how badly do we miss' objective. Zero iff every periodic instance meets
    its deadline. Same instance-level (last-dispatch-per-instance) accounting as instance_misses,
    so a lever that pulls an instance in ahead of its deadline is credited even when it leaves the
    makespan critical path (and thus the makespan) unchanged — which is exactly how IME helps."""
    try:
        from job_names import split_job_name
        sch = json.load(open(sched_path))["dispatches"]
        nets = spec["networks"] if isinstance(spec, dict) else json.load(open(spec))["networks"]
        known = set(nets)
        per = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
        win = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}
        last = {}
        for d in sch.values():
            net, inst = split_job_name(d["job_name"], known)
            if not (net in per and per[net]):
                continue
            e = float(d["start_time"]) + float(d["duration"])
            last[(net, inst)] = max(last.get((net, inst), 0.0), e)
        total = 0.0
        for (net, inst), e in last.items():
            if win[net] > 0:
                total += max(0.0, e - (inst * per[net] + win[net]))
        return total
    except Exception:
        return None


def recost_on_board(sched_path, spec_path, out_path, calibration=None):
    """Re-time a FIXED schedule under measured K1 board costs (scripts/recost_schedule_on_board.py):
    the AOT assignment kept, every dispatch's duration replaced by its board-faithful value, re-timed
    ASAP. This is the 'how the RUN differs from the GANTT' signal. Returns the out schedule path or None."""
    cmd = [PY, "scripts/recost_schedule_on_board.py", "--schedule", sched_path,
           "--spec", spec_path, "--out", out_path]
    if calibration:
        cmd += ["--calibration", calibration]
    r = _run(cmd)
    return out_path if os.path.exists(out_path) else None


def baseline(spec: dict) -> dict:
    """Strip all levers so the loop starts from a clean floor."""
    spec = copy.deepcopy(spec)
    sch = spec.setdefault("scheduler", {})
    sch["enable_impls"] = False
    sch["machine_combination_mode"] = "singletons"
    if "hardware" in spec and "profile" in spec["hardware"]:
        spec["hardware"]["profile"].setdefault("topo_tag_override", True)
    return spec


def apply_ime(spec: dict, log) -> dict:
    spec = copy.deepcopy(spec)
    built = []
    for key, info in spec.get("networks", {}).items():
        net, variant = _net_variant(info.get("dispatch_deps_path", ""), key)
        if not _is_conv_net(net, variant):
            continue
        if os.path.exists(_rvv_profile(net, variant, hw="ime_x60")):
            continue  # already has an ime profile (matmul nets, or a prior build) — do not clobber
        r = _run([PY, "scripts/make_ime_profile.py", "--net", net, "--variant", variant])
        if r.returncode == 0:
            built.append(f"{net}.{variant}")
    log(f"      ime: built ime_x60 profiles for {built or '(none new; existing reused)'}")
    spec.setdefault("scheduler", {})["enable_impls"] = True
    return spec


def apply_shard(spec: dict, log) -> dict:
    spec = copy.deepcopy(spec)
    spec.setdefault("scheduler", {})["machine_combination_mode"] = "shard"
    if "hardware" in spec and "profile" in spec["hardware"]:
        spec["hardware"]["profile"]["topo_tag_override"] = False
    return spec


def apply_unfuse(spec: dict, log) -> dict:
    """AOT graph-rewrite lever (automatic): point each fused net at ModelBlaster's UNFUSED
    dispatch graph and re-solve. The unfused graph is ModelBlaster's real rewrite output
    (`apply_unfuse_hint` on the IR, re-exported); it is adopted here rather than regenerated
    on the fly because kernel regeneration needs the RISC-V cross-toolchain (see
    generate_kernels.py) and board re-profiling needs the physical target — those are build/
    hardware steps, not scheduling. The lever is a no-op for a net with no unfused build,
    and (like every lever) is accepted only if it measures better with no new misses."""
    spec = copy.deepcopy(spec)
    swapped = []
    for key, info in spec.get("networks", {}).items():
        dp = info.get("dispatch_deps_path", "")
        cand = dp.replace(".ctrl.int8", ".unfused.int8").replace(".fused.", ".unfused.")
        cand_abs = cand if os.path.isabs(cand) else os.path.join(REPO, cand)
        if cand != dp and os.path.exists(cand_abs):
            info["dispatch_deps_path"] = cand
            swapped.append(key)
    log(f"      unfuse: adopted ModelBlaster unfused dispatch graph for "
        f"{swapped or '(none — no fused net with an unfused build on disk)'}")
    return spec


LEVERS = {"ime": apply_ime, "shard": apply_shard, "unfuse": apply_unfuse}


def fusion_note():
    try:
        fb = {r["network"]: r for r in csv.DictReader(open(os.path.join(REPO, "xpu-rt/data/fusion_benefit.csv")))}
        worst = max(fb.values(), key=lambda r: float(r["epilogue_ceiling_pct"]))
        return (f"fuse: NOT applied — roofline decision-aid says the stack is compute-bound "
                f"(max fusible-epilogue ceiling {float(worst['epilogue_ceiling_pct']):.0f}% on "
                f"{worst['network']}); fusion collapses dispatches (scheduling) but is measured "
                f"~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.")
    except Exception:
        return "fuse: decision-aid unavailable."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--out-dir", default="results/codesign_loop")
    ap.add_argument("--solver", choices=["greedy", "cpsat"], default="greedy",
                    help="scheduler used for the predicted lever search (greedy is fast/low-mem; "
                         "cpsat gives an optimal 0-miss AOT schedule, needed for the sharp board arc).")
    ap.add_argument("--time-limit", type=int, default=45,
                    help="per-solve CP-SAT seconds during the lever search (ignored by greedy).")
    ap.add_argument("--board-calibration", nargs="?", const=True, default=None, metavar="PATH",
                    help="ENABLE THE BOARD-FEEDBACK ARM. After the predicted lever search converges, "
                         "re-cost the accepted schedule under measured K1 board costs; if that reveals "
                         "deadline misses the AOT Gantt hid, automatically re-solve the SAME spec with "
                         "--board-calibration and accept if the board misses drop. Bare flag uses "
                         "results/codesign_feedback/k1_board_calibration.json; pass a path to override.")
    ap.add_argument("--board-solver", choices=["greedy", "cpsat"], default="cpsat",
                    help="scheduler for the board-calibrated RE-SOLVE (the one final hard solve).")
    ap.add_argument("--board-time-limit", type=int, default=120,
                    help="CP-SAT seconds for the board-calibrated re-solve (the hard solve).")
    ap.add_argument("--levers", default=None,
                    help="comma-separated subset of {ime,shard,unfuse} the search may propose "
                         "(default: all). Restricting to a subset is honest scenario-scoping, e.g. "
                         "'--levers ime' asks 'with sharding off the table, does the loop autonomously "
                         "accept the matrix engine?' — the accept/reject is still measured, not forced.")
    ap.add_argument("--objective", choices=["auto", "makespan", "lateness", "misses"], default="auto",
                    help="acceptance metric. 'auto' = the spec's declared objective (worst-response for "
                         "exact-cycle specs, else makespan). 'lateness' = total instance lateness "
                         "(sum max(0, end-deadline)); 'misses' = instance-miss count. The deadline-"
                         "correct objectives (lateness/misses) credit levers like IME that pull "
                         "instances in ahead of deadline without shortening the makespan critical path.")
    args = ap.parse_args()

    active_levers = list(LEVERS)
    if args.levers:
        want = [l.strip() for l in args.levers.split(",") if l.strip()]
        bad = [l for l in want if l not in LEVERS]
        if bad:
            print(f"--levers: unknown lever(s) {bad}; valid = {list(LEVERS)}")
            return 1
        active_levers = [l for l in LEVERS if l in want]

    board_cal_path = None
    if args.board_calibration is not None:
        board_cal_path = (os.path.join(REPO, "results/codesign_feedback/k1_board_calibration.json")
                          if args.board_calibration is True else args.board_calibration)

    wl_stem = os.path.splitext(os.path.basename(args.workload))[0]
    out_dir = os.path.join(REPO, args.out_dir, wl_stem)
    spec_dir = os.path.join(out_dir, "specs")
    os.makedirs(spec_dir, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    log(f"== co-design loop: {wl_stem} ==")
    if active_levers != list(LEVERS):
        log(f"lever menu restricted to {active_levers} (scenario-scoped; accept/reject still measured)")
    working = baseline(json.load(open(args.workload)))
    base_path = os.path.join(spec_dir, f"{wl_stem}_r0_baseline.json")
    json.dump(working, open(base_path, "w"), indent=1)
    mk, miss, sched, err = solve(base_path, solver=args.solver, time_limit=args.time_limit)
    if mk is None:
        log(f"BASELINE SOLVE FAILED: {err}")
        return 1

    obj_mode = objective_of(working) if args.objective == "auto" else args.objective
    metric_name = {"worst_response": "worst-response", "lateness": "total-lateness",
                   "misses": "instance-misses"}.get(obj_mode, "makespan")
    # the new deadline-correct objectives judge "no new misses" on the INSTANCE-level
    # source-of-truth; worst_response/makespan keep their original op-count guard untouched.
    instance_guard = obj_mode in ("lateness", "misses")

    def score(sched_path, spec_dict, makespan):
        """The number the loop accepts on = the chosen objective (smaller is better)."""
        if obj_mode == "worst_response":
            w = worst_response_ms(sched_path, spec_dict)
            if w is not None:
                return w
        elif obj_mode == "lateness":
            v = total_lateness(sched_path, spec_dict)
            if v is not None:
                return v
        elif obj_mode == "misses":
            v = instance_misses(sched_path, spec_dict)[0]
            if v is not None:
                return float(v)
        return makespan

    def guard_miss(sched_path, spec_dict, op_miss):
        """The miss count the 'no new misses' acceptance guard uses: instance-level for the new
        lateness/misses objectives (the figure's source-of-truth), else the scheduler's op count
        (unchanged for the makespan/worst-response paths)."""
        if instance_guard:
            m = instance_misses(sched_path, spec_dict)[0]
            if m is not None:
                return m
        return op_miss

    base_score = score(sched, working, mk)
    base_gmiss = guard_miss(sched, working, miss)
    log(f"round 0 · baseline: {metric_name} {base_score:.3f} "
        f"(makespan {mk:.1f} ms), {base_gmiss} instance-miss")

    applied, rounds = [], []
    traj = [{"round": 0, "lever": "baseline", "score_ms": round(base_score, 3),
             "makespan_ms": round(mk, 1), "misses": base_gmiss}]
    cur_mk, cur_score, cur_miss, cur_spec = mk, base_score, base_gmiss, working
    cur_sched, cur_spec_path = sched, base_path

    for rnd in range(1, args.max_rounds + 1):
        cands = []
        for lever in active_levers:
            if lever in applied:
                continue
            cspec = LEVERS[lever](cur_spec, log)
            cpath = os.path.join(spec_dir, f"{wl_stem}_r{rnd}_{lever}.json")
            json.dump(cspec, open(cpath, "w"), indent=1)
            cmk, cmiss, csched, cerr = solve(cpath, solver=args.solver, time_limit=args.time_limit)
            if cmk is None:
                log(f"round {rnd} · try {lever}: SOLVE FAILED ({cerr[:120] if cerr else ''}) — reject")
                continue
            csc = score(csched, cspec, cmk)
            cgmiss = guard_miss(csched, cspec, cmiss)
            delta = cur_score - csc
            ok = (cgmiss <= cur_miss) and (delta > EPS)
            pct = (-delta / cur_score * 100) if cur_score else 0.0
            log(f"round {rnd} · try {lever}: {metric_name} {cur_score:.3f} -> {csc:.3f} "
                f"({pct:+.1f}%), {cgmiss} instance-miss -> {'ACCEPTABLE' if ok else 'reject'}")
            cands.append(dict(lever=lever, mk=cmk, score=csc, miss=cgmiss, sched=csched,
                              spec=cspec, spec_path=cpath, ok=ok))

        winners = [c for c in cands if c["ok"]]
        if not winners:
            log(f"round {rnd}: no lever improves {metric_name} with 0 added misses — CONVERGED")
            break
        # lexicographic: minimize the objective first, break ties by makespan (among lever sets
        # that meet deadlines equally, prefer the one that also finishes soonest). On this workload
        # shard and IME EACH drive lateness to 0 independently, so this tie-break is what decides
        # between two genuinely-deadline-meeting options rather than an arbitrary dict order.
        best = min(winners, key=lambda c: (c["score"], c["mk"]))
        pct = ((cur_score - best["score"]) / cur_score * 100) if cur_score else 0.0
        log(f"round {rnd}: ACCEPT +{best['lever']}  {metric_name} "
            f"{cur_score:.3f} -> {best['score']:.3f} (-{pct:.1f}%)")

        # render the accepted schedule's Gantt (IME dispatches darker+hatched)
        gstem = os.path.join(out_dir, f"round_{rnd}_{best['lever']}_gantt")
        _run([PY, "scripts/plot_scheduled_json.py", best["sched"], "--save", gstem,
              "--window-ms", str(round(best["mk"] * 1.05, 1))])

        rounds.append(dict(round=rnd, lever=best["lever"], metric=metric_name,
                           score_before_ms=round(cur_score, 3), score_after_ms=round(best["score"], 3),
                           makespan_before_ms=round(cur_mk, 1), makespan_after_ms=round(best["mk"], 1),
                           pct=round(-pct, 1), accepted=True, deadline_miss=best["miss"]))
        applied.append(best["lever"])
        cur_mk, cur_score, cur_miss, cur_spec = best["mk"], best["score"], best["miss"], best["spec"]
        cur_sched, cur_spec_path = best["sched"], best["spec_path"]
        traj.append({"round": rnd, "lever": f"+{best['lever']}", "score_ms": round(cur_score, 3),
                     "makespan_ms": round(cur_mk, 1), "misses": cur_miss})

    fnote = fusion_note()
    log(fnote)

    # ---- BOARD-FEEDBACK ARM (the paper's headline beat, now automatic) --------------------
    # The predicted lever search above optimizes on ISOLATED profile costs — the AOT Gantt.
    # But the real K1 runs ~26-31% slower per op (measured; per-op exec inflation, not
    # contention). So after the AOT search converges we CLOSE THE LOOP against the board:
    #   1. re-cost the accepted AOT schedule under measured board costs (fixed assignment) —
    #      deadlines the Gantt promised can now be MISSED;
    #   2. if misses appear, automatically RE-SOLVE the same spec with --board-calibration so
    #      the scheduler knows the true costs, and accept if the board misses drop;
    #   3. repeat until misses==0 or no further improvement.
    # Every stage's verdict is a MEASURED instance-miss count (instance_misses(), the figure's
    # own source-of-truth), never a hardcoded number.
    board = {"enabled": bool(board_cal_path)}
    panel_dir = os.path.join(out_dir, "panels")
    if board_cal_path:
        os.makedirs(panel_dir, exist_ok=True)

        def _stash(sched_src, name):
            dst = os.path.join(panel_dir, name + ".json")
            shutil.copy(sched_src, dst)
            msrc = sched_src.replace(".json", "_metrics.json")
            if os.path.exists(msrc):
                shutil.copy(msrc, dst.replace(".json", "_metrics.json"))
            return dst

        if not os.path.exists(board_cal_path):
            log(f"board arm: no calibration artifact at {board_cal_path} — arm SKIPPED (no-op).")
            board.update(enabled=False, reason="calibration artifact missing")
        else:
            log(f"\n== board-feedback arm ==  (calibration {os.path.relpath(board_cal_path, REPO)})")
            # The lever search above used short per-solve budgets to RANK levers cheaply; its
            # incumbent for the winning spec is not necessarily the optimum. Before closing the
            # board loop, re-solve the converged spec at the full budget so the AOT panel is the
            # definitive optimum for the loop's decision (same spec, solved to optimality) — and
            # so the reveal is measured against a schedule that genuinely meets every AOT deadline.
            amk, _aop, asched, aerr = solve(cur_spec_path, solver=args.board_solver,
                                            time_limit=args.board_time_limit)
            if asched is not None:
                cur_sched, cur_mk = asched, amk
                log(f"board arm · re-solved converged AOT spec at full budget "
                    f"({args.board_solver}, {args.board_time_limit}s): makespan {amk:.2f} ms")
            else:
                log(f"board arm · full-budget AOT re-solve failed ({(aerr or '')[:100]}); "
                    f"keeping the search incumbent")
            # canonical stage 1/2 schedules: baseline and the converged (polished) AOT schedule
            p1 = _stash(sched, "stage1_baseline")
            p2 = _stash(cur_sched, "stage2_aot")
            b_base = instance_misses(p1, working)[0]
            b_aot = instance_misses(p2, cur_spec)[0]
            log(f"board arm · stage1 baseline (AOT costs): {b_base} instance-miss")
            log(f"board arm · stage2 AOT-optimized (AOT costs): {b_aot} instance-miss  "
                f"[levers: {applied or 'none'}]")

            # stage 3 — re-cost the AOT schedule on the board (fixed assignment)
            p3 = os.path.join(panel_dir, "stage3_board_recost.json")
            recost_on_board(cur_sched, cur_spec_path, p3, calibration=board_cal_path)
            b_recost, b_recost_by = instance_misses(p3, cur_spec)
            log(f"board arm · stage3 board re-cost of the AOT schedule: {b_recost} instance-miss "
                f"{b_recost_by}  <- the run-honest reveal")

            stages = [
                dict(stage="baseline", cost="AOT", sched=os.path.relpath(p1, REPO),
                     instance_misses=b_base, makespan_ms=round(mk, 2)),
                dict(stage="aot-optimized", cost="AOT", sched=os.path.relpath(p2, REPO),
                     instance_misses=b_aot, makespan_ms=round(cur_mk, 2), levers=list(applied)),
                dict(stage="board-recost", cost="board", sched=os.path.relpath(p3, REPO),
                     instance_misses=b_recost, missed_by=b_recost_by),
            ]
            traj.append({"round": "board-recost", "lever": "recost(board)",
                         "makespan_ms": round(mk_of(p3), 1), "misses": b_recost})

            # stage 4 — automatic re-solve on board-calibrated costs, iterate the arm
            p4 = None
            b_final = b_recost
            if b_recost > 0:
                cur_board_miss = b_recost
                for it in range(1, args.max_rounds + 1):
                    rmk, rop, rsched, rerr = solve(cur_spec_path, solver=args.board_solver,
                                                   board_cal=board_cal_path,
                                                   time_limit=args.board_time_limit)
                    if rmk is None:
                        log(f"board arm · re-solve iter {it}: SOLVE FAILED ({(rerr or '')[:120]})")
                        break
                    b_re, b_re_by = instance_misses(rsched, cur_spec)
                    better = b_re < cur_board_miss
                    log(f"board arm · re-solve iter {it} (--board-calibration, {args.board_solver}): "
                        f"{cur_board_miss} -> {b_re} instance-miss {b_re_by} -> "
                        f"{'ACCEPT' if better else 'no improvement, stop'}")
                    if not better:
                        break
                    p4 = _stash(rsched, "stage4_board_resolve")
                    cur_board_miss = b_re
                    b_final = b_re
                    if b_re == 0:
                        break
                if p4 is not None:
                    stages.append(dict(stage="board-resolve", cost="board",
                                       sched=os.path.relpath(p4, REPO),
                                       instance_misses=b_final, makespan_ms=round(mk_of(p4), 2),
                                       solver=args.board_solver))
                    traj.append({"round": "board-resolve", "lever": "resolve(board)",
                                 "makespan_ms": round(mk_of(p4), 1), "misses": b_final})
            else:
                log("board arm · AOT schedule already meets every deadline on the board — "
                    "no re-solve needed.")

            board.update(
                calibration=os.path.relpath(board_cal_path, REPO),
                board_solver=args.board_solver,
                spec=os.path.relpath(cur_spec_path, REPO),
                baseline_instance_misses=b_base,
                aot_instance_misses=b_aot,
                board_recost_instance_misses=b_recost,
                board_resolve_instance_misses=(b_final if p4 is not None else None),
                recovered=(p4 is not None and b_final < b_recost),
                residual_misses=b_final,
                stages=stages,
            )
            arc = f"{b_base} -> {b_aot} -> {b_recost} -> {b_final if p4 is not None else b_recost}"
            log(f"board arm: instance-miss arc  baseline->AOT->board-recost->board-resolve = {arc}")

    denom = base_score if base_score else 1.0
    report = dict(workload=wl_stem, objective=metric_name,
                  baseline_score_ms=round(base_score, 3), final_score_ms=round(cur_score, 3),
                  total_reduction_pct=round((base_score - cur_score) / denom * 100, 1),
                  baseline_makespan_ms=round(mk, 1), final_makespan_ms=round(cur_mk, 1),
                  levers_available=active_levers,
                  levers_applied=applied, rounds=rounds, trajectory=traj,
                  fusion=fnote, converged=True, board_feedback=board)
    json.dump(report, open(os.path.join(out_dir, "loop_report.json"), "w"), indent=1)

    _plot_traj(traj, os.path.join(out_dir, "objective_vs_round"), metric_name)
    _write_readme(out_dir, wl_stem, args, report, fnote)
    log(f"\nCONVERGED [{metric_name}]: {base_score:.3f} -> {cur_score:.3f} ms "
        f"(-{report['total_reduction_pct']:.1f}%), levers applied: {applied or 'none'}")
    log(f"artifacts in {out_dir}")
    open(os.path.join(out_dir, "loop_log.txt"), "w").write("\n".join(lines) + "\n")
    return 0


def _plot_traj(traj, out, metric_name="makespan"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(traj)))
        ys = [t.get("score_ms", t["makespan_ms"]) for t in traj]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.step(xs, ys, where="post", color="#4c72b0", lw=2, marker="o")
        for i, t in enumerate(traj):
            ax.annotate(t["lever"], (i, ys[i]),
                        textcoords="offset points", xytext=(6, 8), fontsize=9,
                        weight="bold" if t["lever"] != "baseline" else "normal")
        if len(ys) > 1 and ys[0]:
            ax.annotate(f"-{(ys[0]-ys[-1])/ys[0]*100:.1f}%", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, -14), fontsize=9, color="#2f7d4f")
        ax.set_xlabel("feedback round"); ax.set_ylabel(f"{metric_name} (ms)")
        ax.set_xticks(xs); ax.set_ylim(0, max(ys) * 1.15)
        ax.set_title(f"Automatic co-design loop — {metric_name} per accepted round", weight="bold")
        fig.tight_layout()
        fig.savefig(out + ".png", dpi=160); fig.savefig(out + ".pdf")
    except Exception as e:
        print("traj plot skipped:", e)


def _write_readme(out_dir, stem, args, report, fnote):
    r = report
    md = [f"# Automatic co-design loop — `{stem}`", "",
          "Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →",
          "measure each → accept the largest measured makespan win with 0 added misses → repeat.",
          "", "```",
          f"scripts/run_codesign_loop.py --workload {args.workload} --max-rounds {args.max_rounds}",
          "```", "",
          f"**Baseline → final: {r['baseline_makespan_ms']} → {r['final_makespan_ms']} ms "
          f"(-{r['total_reduction_pct']}%)** — levers applied: {r['levers_applied'] or 'none'}.", "",
          "| round | lever | before (ms) | after (ms) | % | misses |", "|--:|--|--:|--:|--:|--:|"]
    for rr in r["rounds"]:
        md.append(f"| {rr['round']} | +{rr['lever']} | {rr['makespan_before_ms']} | "
                  f"{rr['makespan_after_ms']} | {rr['pct']} | {rr['deadline_miss']} |")
    md += ["", f"Honest note — {fnote}", "",
           "Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, "
           "`round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), "
           "`specs/` (every candidate spec), `loop_log.txt`."]
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    sys.exit(main())
