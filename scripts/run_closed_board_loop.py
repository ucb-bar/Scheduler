#!/usr/bin/env python3
"""The outer loop with the board actually in it, feeding new AOT decisions.

WHAT WAS OPEN. Everything called "board feedback" in this repo so far consumed ONE
static calibration table, fit to 60 runs of one taskset, and used it only to RE-SOLVE a
spec the AOT stage had already committed to. Two consequences, both structural:

  * a workload containing a network that taskset never ran -- `yolov8_nano_64x96`, the
    net that dominates the 5-net rung -- was costed by the op-kind fallback, i.e.
    extrapolated. The table says so itself.
  * the board could fix a PLACEMENT and never a TRANSFORMATION. The lever and rewrite
    search ran on isolated profile costs, so a lever the silicon punishes was adopted
    anyway (on `w5_ffn_dronet_yolo` the AOT search credits `shard` with 11 -> 5 misses;
    scored on board costs `shard` is rejected, worst lateness 38.62 -> 40.38 ms).

This closes both. One command, four stages, no hand-editing:

  1. AOT   run_codesign_loop.py with no calibration            -> converged spec+schedule
  2. BOARD ModelBlaster/scripts/run_xpurt_k1.sh, N repeats     -> measured traces
  3. FIT   emit_board_calibration.py over those traces         -> calibration for THIS
                                                                  workload, measured
  4. AOT'  run_codesign_loop.py --search-calibration <that>    -> decisions re-taken

and then reports which transformations the measured costs added or dropped, which is the
only output that answers "would the board have chosen differently".

Stage 3 is what makes stage 4 honest: the calibration covers exactly the networks that
just ran, so nothing in stage 4 is extrapolated for this workload. `coverage.nets_exact`
in the emitted table records that, and this script checks the spec's networks against it
and says plainly which -- if any -- are still uncovered.

Requires the cross toolchain (`eval "$(scripts/setup_spacemit_toolchain.sh)"`, fetched
automatically here) and a reachable board (ssh alias `k1`, or $MODELBLASTER_K1_HOST).

  scripts/run_closed_board_loop.py \\
      --workload data/toplevel/scaling/w3_ffn_dronet.json \\
      --repeats 3 --out-dir results/codesign_feedback/closed_w3
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)
MB = os.environ.get("MB_ROOT") or os.path.join(REPO, "ModelBlaster")

#: Networks whose IR is supplied rather than extracted, as `net -> repo-relative DIR`.
#: `yolov8_nano_64x96` is the `yolov8_nano` module built at a different input size and
#: registered under its own name, so `extract_graph --model yolov8_nano_64x96` has
#: nothing to extract -- the runner needs --staged-ir for it. Build the IR with
#: `MODELBLASTER_YOLOV8N_INPUT=64x96 python -m pipeline.extract_graph --model
#: yolov8_nano --quant int8 --out-dir <dir>`; the size comes from that env var, not from
#: the name.
#:
#: A DIRECTORY, and made absolute below. `--staged-ir` wants `<net>:<dir>` containing
#: graph.json, weights.npz and io.npz, and resolves it against ModelBlaster's own root
#: rather than this repo's -- so a repo-relative path silently points somewhere else.
NEEDS_STAGED_IR = {
    "yolov8_nano_64x96": "gen_mb/ir/yolov8_nano_64x96/int8",
}
#: Files `--staged-ir` requires in that directory, checked here so the failure names the
#: missing artifact instead of arriving from inside a shell script.
STAGED_IR_FILES = ("graph.json", "weights.npz", "io.npz")

#: The board build needs GCC 14.3. 13.2 -- what CROSS defaults to via chipyard --
#: miscompiles the RVV intrinsics into a SIGILL with no stdout; see
#: scripts/setup_spacemit_toolchain.sh for the reordered vsetvl.
CORE_KINDS = "rvv,ime,rvv_c1"

#: `fused_full` is the stateful FP16 sensor-fusion net, and extracting it needs the REAL
#: calibration set -- it is quantised against recorded sensor data, not synthetic input,
#: and its low-dimensional branch stays float. Every documented board run of a workload
#: containing it sets these three. Without the pkl the extraction still runs and produces
#: a differently-quantised model, which is the worst kind of failure: the run succeeds
#: and the numbers are not the published ones.
FUSED_CALIB_PKL = os.environ.get(
    "MB_FUSED_CALIB_PKL",
    "/scratch/dima/rose-infra/RoSE/experiments/rose_nav_cosim/calib/calib_real.pkl")
FUSED_ENV = {"MB_FUSED_LOWDIM_FLOAT": "1", "NUM_CALIBRATION": "32"}


def sh(cmd, env=None, cwd=None, log=None, stream=False):
    e = dict(os.environ)
    e.update(env or {})
    if log:
        log(f"  $ {' '.join(str(c) for c in cmd)}")
    if stream:
        r = subprocess.run(cmd, cwd=cwd or REPO, env=e)
        return subprocess.CompletedProcess(cmd, r.returncode, "", "")
    return subprocess.run(cmd, cwd=cwd or REPO, env=e, capture_output=True, text=True)


def toolchain_prefix(log):
    """The GCC 14.3 cross prefix, fetching the toolchain if it is not there yet."""
    r = sh(["bash", os.path.join(REPO, "scripts/setup_spacemit_toolchain.sh"), "--path"])
    for line in reversed((r.stdout or "").strip().splitlines()):
        if line.strip().endswith("-"):
            return line.strip()
    log(f"could not resolve the cross toolchain: {(r.stderr or '')[-300:]}")
    return None


def board_python(explicit, log):
    """An interpreter the BOARD BUILD can use, which is not necessarily this one.

    Stage 2 runs ModelBlaster's `extract_graph`, and that imports torch (and ultralytics
    for the detector). This repo's venv is installed `--no-deps` on purpose -- the docs
    say so, because only the model-extraction path needs either -- so the venv that runs
    the scheduler cannot run the extractor. The failure is four stages in and reads as
    `ModuleNotFoundError: No module named 'torch'` from inside a shell script, which is
    not where anyone looks.

    A candidate has to satisfy BOTH conditions: torch imports, and `modelblaster`
    resolves inside THIS checkout. The second is not paranoia -- there is a second
    ModelBlaster clone on this machine and some envs carry an editable install pointing
    at it, so an interpreter can import a DIFFERENT pipeline under the same module names.
    `run_xpurt_k1.sh` refuses in that case; this checks first so the reason is legible.
    """
    cands = [c for c in (explicit, os.environ.get("MB_PY"), PY,
                         "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python",
                         shutil.which("python3")) if c]
    probe = ("import torch, modelblaster.pipeline.backends as b; "
             "print(b.__file__)")
    tried = []
    for c in cands:
        if not os.path.exists(c):
            continue
        e = dict(os.environ)
        e["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(MB, "src"), MB] + ([e["PYTHONPATH"]] if e.get("PYTHONPATH")
                                             else []))
        r = subprocess.run([c, "-c", probe], capture_output=True, text=True, env=e,
                           cwd=MB)
        where = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
        if r.returncode == 0 and where.startswith(MB):
            log(f"  board build will use {c}")
            return c
        why = ("torch/modelblaster import failed"
               if r.returncode != 0 else f"modelblaster resolves to {where}")
        tried.append(f"{c}: {why}")
    log("  no interpreter can run the board build. Stage 2 needs torch (and "
        "ultralytics for the detector) AND a `modelblaster` that resolves inside "
        f"{os.path.relpath(MB, REPO)}. Tried:")
    for t in tried:
        log(f"    {t}")
    log("  pass --board-py <interpreter>, or set MB_PY.")
    return None


def loop(workload, out_dir, log, search_cal=None, solver="greedy", extra=()):
    """One run of the co-design driver; returns its loop_report.json."""
    cmd = [PY, "-u", os.path.join(REPO, "scripts/run_codesign_loop.py"),
           "--workload", workload, "--solver", solver, "--objective", "lateness",
           "--out-dir", out_dir, *extra]
    if search_cal:
        cmd += ["--search-calibration", search_cal]
    r = sh(cmd, log=log)
    stem = os.path.splitext(os.path.basename(workload))[0]
    rep = os.path.join(out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir),
                       stem, "loop_report.json")
    if r.returncode != 0 or not os.path.exists(rep):
        log(f"  co-design loop FAILED (rc={r.returncode})")
        log("  " + ((r.stderr or r.stdout or "")[-500:]).replace("\n", "\n  "))
        return None
    return json.load(open(rep))


def converged_spec(report, out_dir, workload):
    """The spec the AOT stage converged on -- the one the board should execute.

    The report records the accepted candidates; the last accepted round's spec IS the
    converged spec. Falling back to the baseline when nothing was accepted is correct
    and not a failure: a workload the levers cannot help is still a workload whose real
    costs we want to measure.
    """
    stem = os.path.splitext(os.path.basename(workload))[0]
    d = os.path.join(out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir),
                     stem, "specs")
    accepted = [r for r in (report.get("rounds") or []) if r.get("accepted")]
    if accepted:
        last = accepted[-1]
        for key in ("spec_path", "spec"):
            p = last.get(key)
            if isinstance(p, str):
                p = p if os.path.isabs(p) else os.path.join(REPO, p)
                if os.path.exists(p):
                    return p
        # the round records the lever, and the driver names the spec after it
        cand = os.path.join(d, f"{stem}_r{last['round']}_{last['lever']}.json")
        if os.path.exists(cand):
            return cand
    base = os.path.join(d, f"{stem}_r0_baseline.json")
    return base if os.path.exists(base) else None


def _codesign_module():
    """Import the loop driver so its `solve()` is reused, not reimplemented.

    The first version of this script rebuilt the solve command by hand and got it wrong
    in a way that only surfaced after the AOT stage had already run: greedy is
    `--solver greedy` with NO `--scheduler`, and the schedule it writes is suffixed
    `_greedy_profiled`. Both facts live in run_codesign_loop.solve(), together with the
    CP-SAT budget trap (`--time-limit` is MILP-only), so calling it is the only way this
    stage cannot drift from the stage that chose the spec.
    """
    import importlib.util
    path = os.path.join(REPO, "scripts", "run_codesign_loop.py")
    spec = importlib.util.spec_from_file_location("run_codesign_loop", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_codesign_loop"] = mod
    spec.loader.exec_module(mod)
    return mod


def solve_for_board(spec_path, out_dir, solver, time_limit, log):
    """Solve the converged spec and return the schedule the board will execute."""
    cds = _codesign_module()
    os.environ.setdefault("XPURT_NO_COMPACT", "1")
    mk, miss, sched, err = cds.solve(spec_path, solver=solver, time_limit=time_limit)
    if sched is None or not os.path.exists(sched):
        log(f"  solve produced no schedule: {(err or '')[-400:]}")
        return None
    log(f"  solved: makespan {mk:.2f} ms, {miss} dispatch-level deadline misses")
    shutil.copy2(sched, os.path.join(out_dir, os.path.basename(sched)))
    return sched


#: Ops whose weights are PACKED PER SHARD at codegen time, so one generated model
#: cannot carry two layouts for one dispatch. Mirrors
#: `ModelBlaster/pipeline/schedule_shards._PACKED_WEIGHT_SHARD_OPS`; a linear is absent
#: because its row-major weights are sliced at runtime from the entry's own pool width.
PACKED_WEIGHT_OPS = {"conv2d_s8", "conv2d_batchnorm2d_s8",
                     "conv2d_batchnorm2d_silu_s8", "conv2d_silu_s8"}


def undeployable_widths(sched_path, log):
    """`{net: {dispatch_id: [widths]}}` a ModelBlaster build cannot express.

    WHY THIS IS CHECKED HERE. XPU-RT's `shard` mode lets every periodic INSTANCE of a
    dispatch pick its own aligned core block, and for a convolution that is not
    buildable: the packed weight array is materialized per shard while generating the
    skeleton, so the width has to be one value per dispatch. The scheduler does not know
    that constraint, so it produces schedules that are valid for the runtime and
    impossible for the compiler -- on the 5-net rung, `dronet` dispatches 0, 3, 8 and 9
    each take two or three different widths across their five instances.

    Discovering it inside the board build costs the whole build: it dies at stage 1 of 5,
    after extracting and generating for every model, with an error raised from a shell
    script. Checking the schedule first costs milliseconds and names the dispatches.
    """
    sched = json.load(open(sched_path))
    packed, widths = {}, {}
    for e in (sched.get("dispatches") or {}).values():
        job = str(e.get("job_name", ""))
        net = job.rstrip("0123456789") or job
        did = int(e["id"])
        # The op is not its own field; it is embedded in `module_name`, shaped
        # `<net>$dispatch_<id>_<backend>_<op>_<SHAPE>` -- so match the op name
        # delimited by underscores rather than trying to split the whole thing. The
        # packed names do not prefix one another (`conv2d_batchnorm2d_s8` does not
        # contain `conv2d_s8`), so a delimited substring test is exact here.
        mod = str(e.get("module_name") or "")
        packed[(net, did)] = any(f"_{op}_" in mod for op in PACKED_WEIGHT_OPS)
        n = len([x for x in str(e.get("hardware_target", "")).split("+") if x.strip()])
        widths.setdefault((net, did), set()).add(n)
    bad = {}
    for (net, did), ws in sorted(widths.items()):
        if len(ws) > 1 and packed.get((net, did)):
            bad.setdefault(net, {})[did] = sorted(ws)
    return bad


def run_on_board(sched, nets, repeats, out_dir, cross, mb_py, log,
                 warnings=None):
    """Execute the schedule on the K1 `repeats` times; return the pulled trace paths.

    Each repeat's trace is copied out immediately. The runner writes to a path derived
    from the schedule name, so a later repeat overwrites an earlier one -- copying per
    iteration is what makes `--repeats` mean anything.
    """
    warnings = warnings if warnings is not None else []
    staged = []
    for n in nets:
        if n not in NEEDS_STAGED_IR:
            continue
        d = os.path.join(REPO, NEEDS_STAGED_IR[n])
        missing = [f for f in STAGED_IR_FILES if not os.path.exists(os.path.join(d, f))]
        if missing:
            log(f"  {n} needs a staged IR at {os.path.relpath(d, REPO)} and "
                f"{missing} are missing -- see NEEDS_STAGED_IR for how to build it")
            return []
        staged.append(f"{n}:{d}")
    models = ",".join(n for n in nets if n not in NEEDS_STAGED_IR)
    cmd = ["bash", "scripts/run_xpurt_k1.sh", "--schedule", os.path.relpath(sched, MB),
           "--backends", "rvv_x60,ime_x60,rvv_x60", "--jobs", "4"]
    if models:
        cmd += ["--models", models]
    for s in staged:
        cmd += ["--staged-ir", s]
    if any(n.startswith("fused_") for n in nets):
        if os.path.exists(FUSED_CALIB_PKL):
            extra_env = dict(FUSED_ENV, MB_FUSED_CALIB_PKL=FUSED_CALIB_PKL)
            log(f"  fused net present: quantising against {FUSED_CALIB_PKL}")
        else:
            extra_env = {}
            log(f"  WARNING: {[n for n in nets if n.startswith('fused_')]} present but "
                f"no calibration set at {FUSED_CALIB_PKL} -- the extraction will "
                f"succeed and produce a DIFFERENTLY QUANTISED model than every "
                f"published run. Set MB_FUSED_CALIB_PKL.")
            warnings.append("fused net quantised without the real calibration set")
    else:
        extra_env = {}
    env = {**extra_env, "CORE_KINDS": CORE_KINDS, "CROSS": cross, "PY": mb_py,
           "MODELBLASTER_KERNEL_CC": cross + "gcc",
           # SCHED_FIFO 80: the measured runs this project publishes are RT-pinned, and
           # a CFS run measures the Linux scheduler as much as the kernel -- the
           # non-RT trace set shows 1.35x where the RT set shows 0.92x on the same
           # schedule, which is preemption, not execution.
           "MODELBLASTER_K1_RT_PRIORITY": "80"}
    if staged:
        log(f"  staged IR for {[s.split(':')[0] for s in staged]} "
            f"(no --model can regenerate these)")
    traces = []
    for i in range(repeats):
        log(f"  board run {i + 1}/{repeats}")
        r = sh(cmd, env=env, cwd=MB, log=(log if i == 0 else None), stream=True)
        if r.returncode != 0:
            log(f"  board run {i + 1} FAILED (rc={r.returncode}) -- see the runner's "
                f"output above")
            break
        stem = os.path.splitext(os.path.basename(sched))[0]
        gen = os.path.join(MB, "build/k1_xpurt/_gen", stem)
        src = os.path.join(gen, f"{stem}_trace.csv")
        if not os.path.exists(src):
            log(f"  board run {i + 1} produced no trace at {src}")
            break
        dst = os.path.join(out_dir, f"board_{i}_trace.csv")
        shutil.copy2(src, dst)
        traces.append(dst)
        # KEEP THE RUNNER'S OWN STDOUT. It carries the two facts that decide whether a
        # trace is worth calibrating from and neither is in the CSV: `entries_done`,
        # which is 0 when every worker refused every entry (a completed run that
        # executed nothing), and the ingest's profile-database staleness warning.
        blog = os.path.join(gen, f"{stem}_stdout.txt")
        if os.path.exists(blog):
            shutil.copy2(blog, os.path.join(out_dir, f"board_{i}_stdout.txt"))
            txt = open(blog, errors="replace").read()
            done = [ln for ln in txt.splitlines() if "entries_done" in ln]
            if done:
                log(f"    {done[-1].strip()}")
            if "entries_done=0" in txt:
                log("    WARNING: entries_done=0 -- the run completed having executed "
                    "nothing (core_kind vs backend-tag mismatch); this trace is all "
                    "zeros and must not be calibrated from")
                warnings.append(f"run {i}: entries_done=0")
    return traces


def fit_calibration(traces, workload_desc, out_path, log):
    cmd = [PY, os.path.join(REPO, "scripts/emit_board_calibration.py"),
           "--workload", workload_desc, "--out", out_path,
           "--validate-against", "results/codesign_feedback/k1_board_calibration.json"]
    for t in traces:
        cmd += ["--trace", t]
    r = sh(cmd, log=log)
    for line in (r.stdout or "").splitlines():
        log("  " + line)
    if not os.path.exists(out_path):
        log(f"  calibration fit FAILED: {(r.stderr or '')[-300:]}")
        return None
    return json.load(open(out_path))


def decisions_of(report):
    """`(levers, misses_before, misses_after, score_before, score_after)`."""
    bt = report.get("baseline_terms") or {}
    ft = report.get("final_terms") or {}
    return (list(report.get("levers_applied") or []),
            bt.get("01_hard_deadline_misses"), ft.get("01_hard_deadline_misses"),
            report.get("baseline_score_ms"), report.get("final_score_ms"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--repeats", type=int, default=3,
                    help="board executions of the AOT schedule; the calibration is the "
                         "mean over all of them")
    ap.add_argument("--solver", default="greedy", choices=["greedy", "cpsat"])
    # int, because run_codesign_loop.py's --time-limit is an int and argparse
    # rejects "90.0" -- a float here fails four stages in, after the solve.
    ap.add_argument("--time-limit", type=int, default=90)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loop-args", default="",
                    help="extra arguments passed verbatim to run_codesign_loop.py in "
                         "BOTH search stages, e.g. \"--levers ime\"")
    ap.add_argument("--board-py", default=None,
                    help="interpreter for the board build's ModelBlaster steps; it "
                         "needs torch, which this repo's venv deliberately does not "
                         "have. Auto-detected when omitted.")
    ap.add_argument("--skip-board", action="store_true",
                    help="reuse board_*_trace.csv already in --out-dir; for re-fitting "
                         "and re-deciding without another board slot")
    a = ap.parse_args()

    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(REPO, a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    spec = json.load(open(a.workload))
    nets = list((spec.get("networks") or {}).keys())
    stem = os.path.splitext(os.path.basename(a.workload))[0]
    log(f"== closed board loop: {stem} ==")
    log(f"networks: {nets}")

    cross = toolchain_prefix(log)
    mb_py = None if a.skip_board else board_python(a.board_py, log)
    if not a.skip_board and (cross is None or mb_py is None):
        return 1

    # ---- stage 1: AOT, isolated profile costs -----------------------------------
    log("\n[1/4] AOT search on isolated profile costs")
    aot_dir = os.path.join(out_dir, "aot")
    loop_extra = ["--time-limit", str(a.time_limit)] + a.loop_args.split()
    rep_aot = loop(a.workload, aot_dir, log, solver=a.solver, extra=loop_extra)
    if rep_aot is None:
        return 1
    lv, mb_, mf_, sb, sf = decisions_of(rep_aot)
    log(f"  AOT decided: {lv or 'nothing'}   misses {mb_} -> {mf_}   "
        f"lateness {sb} -> {sf} ms")

    cspec = converged_spec(rep_aot, aot_dir, a.workload)
    if cspec is None:
        log("  could not locate the converged spec -- nothing to execute")
        return 1
    log(f"  converged spec: {os.path.relpath(cspec, REPO)}")

    # ---- stage 2: execute it on the board ----------------------------------------
    log(f"\n[2/4] execute that schedule on the K1 x{a.repeats}")
    board_warnings = []
    traces = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                    if f.startswith("board_") and f.endswith("_trace.csv"))
    if a.skip_board:
        log(f"  --skip-board: reusing {len(traces)} trace(s) already here")
    else:
        sched = solve_for_board(cspec, out_dir, a.solver, a.time_limit, log)
        if sched is None:
            return 1
        bad = undeployable_widths(sched, log)
        if bad:
            log("  NOT DEPLOYABLE: this schedule gives a packed-weight (convolution) "
                "dispatch different core widths in different periodic instances, and "
                "one generated model cannot encode two packed layouts for one dispatch:")
            for net, dids in bad.items():
                for did, ws in dids.items():
                    log(f"    {net} dispatch {did}: widths {ws}")
            log("  This is a scheduler/compiler mismatch, not a bad schedule: XPU-RT's "
                "shard mode lets every INSTANCE pick its own aligned block and does not "
                "know the codegen constraint. Re-run restricting the levers to ones "
                "that do not vary width per instance (e.g. --loop-args '--levers ime'), "
                "or pin a single shard width per net.")
            json.dump({"undeployable_widths": bad,
                       "schedule": os.path.relpath(sched, REPO)},
                      open(os.path.join(out_dir, "deployability.json"), "w"), indent=1)
            return 2
        traces = run_on_board(sched, nets, a.repeats, out_dir, cross,
                              mb_py, log, board_warnings)
    if not traces:
        log("  no traces -- the loop cannot be closed without them")
        return 1

    # ---- stage 3: fit the calibration to THIS workload ---------------------------
    log(f"\n[3/4] fit a calibration to this workload's own {len(traces)} run(s)")
    cal_path = os.path.join(out_dir, f"k1_calibration_{stem}.json")
    cal = fit_calibration(traces, f"{stem}: {'+'.join(nets)}", cal_path, log)
    if cal is None:
        return 1
    covered = set(cal["coverage"]["nets_exact"])
    uncovered = [n for n in nets if n not in covered]
    if uncovered:
        log(f"  STILL EXTRAPOLATED: {uncovered} did not appear in the traces, so they "
            f"are costed by op kind or by the aggregate")
    else:
        log(f"  every network in this workload is MEASURED -- nothing extrapolated")

    # ---- stage 4: re-take the AOT decisions on measured costs --------------------
    log("\n[4/4] AOT search again, scored on those measured costs")
    brd_dir = os.path.join(out_dir, "board")
    rep_brd = loop(a.workload, brd_dir, log, search_cal=cal_path, solver=a.solver,
                   extra=loop_extra)
    if rep_brd is None:
        return 1
    lv2, mb2, mf2, sb2, sf2 = decisions_of(rep_brd)
    log(f"  board-honest decided: {lv2 or 'nothing'}   misses {mb2} -> {mf2}   "
        f"lateness {sb2} -> {sf2} ms")

    # ---- the only output that answers the question -------------------------------
    added = [l for l in lv2 if l not in lv]
    dropped = [l for l in lv if l not in lv2]
    log("\n=== would the board have chosen differently? ===")
    log(f"  AOT-cost search   : {lv or 'nothing'}")
    log(f"  board-cost search : {lv2 or 'nothing'}")
    if added or dropped:
        if dropped:
            log(f"  DROPPED on measured costs: {dropped} -- the AOT search would have "
                f"deployed these and the silicon does not support the decision")
        if added:
            log(f"  ADDED on measured costs: {added} -- worth doing only once the real "
                f"cost of the ops involved is known")
    else:
        log("  same transformations either way: the AOT decision is robust to the cost "
            "model on this workload")
    if isinstance(sf, (int, float)) and isinstance(sf2, (int, float)) and sf > 0:
        log(f"  residual lateness the AOT view reports: {sf} ms; measured-cost view: "
            f"{sf2} ms ({sf2 / sf:.2f}x)")

    summary = dict(
        workload=stem, networks=nets, solver=a.solver, repeats=len(traces),
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        calibration=os.path.relpath(cal_path, REPO),
        calibration_aggregate=cal.get("aggregate_multiplier"),
        board_warnings=board_warnings,
        nets_measured=sorted(covered), nets_still_extrapolated=uncovered,
        aot=dict(levers=lv, misses_before=mb_, misses_after=mf_,
                 lateness_before_ms=sb, lateness_after_ms=sf),
        board=dict(levers=lv2, misses_before=mb2, misses_after=mf2,
                   lateness_before_ms=sb2, lateness_after_ms=sf2),
        levers_dropped_on_measured_costs=dropped,
        levers_added_on_measured_costs=added,
        traces=[os.path.relpath(t, REPO) for t in traces],
    )
    json.dump(summary, open(os.path.join(out_dir, "closed_loop_summary.json"), "w"),
              indent=1)
    open(os.path.join(out_dir, "closed_loop_log.txt"), "w").write("\n".join(lines) + "\n")
    log(f"\nartifacts in {os.path.relpath(out_dir, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
