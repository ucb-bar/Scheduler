#!/usr/bin/env python3
"""Rebuild results/codesign_feedback/hil_flights_master.csv from every outcome-bearing flight CSV.

Unions the worktree HIL CSVs + the canonical perc_crash sweeps, tags each row with a `source` and a
`regime`, and adds a `success` flag. Regimes keep distinct experiments from being silently pooled.
Excludes derived/duplicate files (the master itself, the v1 archive, the v2 staging file) and the
live campaign dir. Idempotent — safe to re-run whenever new CSVs land.
"""
import csv, glob, os, shutil
from collections import OrderedDict, Counter

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_repo, "results/codesign_feedback")
CANP = "/scratch/agustin/projects/DIMA/XPU-RT/results/perc_crash"
SCRATCH2 = "/scratch2/agustin/XPU-RT/results/codesign_feedback"
EXCLUDE = {"hil_flights_master.csv", "hil_ablation_v1_120flights_fixedgain.csv", "hil_ablation_v2.csv"}


def regime(src):
    if src.startswith("perc_crash_"): return "perc_freshness/safety (canonical; separate experiment)"
    if src == "hil_ablation": return "envelope (figure source)"
    if src == "hil_ablation_courseB": return "envelope course-B (generalization; do not pool with A)"
    if src.startswith("crash_verify_new"): return "showdown (calibrated gain, ZOH latency)"
    if src.startswith("hil_dense") or src.startswith("hil_pilot"): return "calibrated-gain grid"
    return "misc/verification"


def collect():
    srcs = []
    for f in sorted(glob.glob(RES + "/**/*.csv", recursive=True)):
        if "strengthen_campaign" in f or os.path.basename(f) in EXCLUDE:
            continue
        rows = list(csv.DictReader(open(f)))
        if rows and "outcome" in rows[0]:
            srcs.append((os.path.relpath(f, RES)[:-4].replace("/", "_"), rows))
    for f in sorted(glob.glob(CANP + "/*.csv")):
        rows = list(csv.DictReader(open(f)))
        if rows and "outcome" in rows[0]:
            srcs.append(("perc_crash_" + os.path.basename(f)[:-4], rows))
    return srcs


def main():
    srcs = collect()
    cols = OrderedDict()
    for _, rows in srcs:
        for c in rows[0]:
            cols.setdefault(c, None)
    base = list(cols); out = ["source"] + base + ["regime", "success"]
    master = os.path.join(RES, "hil_flights_master.csv")
    n = 0; byreg = Counter()
    with open(master, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out); w.writeheader()
        for src, rows in srcs:
            for r in rows:
                rec = {c: r.get(c, "") for c in base}
                rec["source"] = src; rec["regime"] = regime(src)
                rec["success"] = int(r.get("outcome", "") == "success")
                w.writerow(rec); n += 1; byreg[regime(src)] += 1
    try:
        shutil.copy(master, os.path.join(SCRATCH2, "hil_flights_master.csv"))
    except Exception as e:
        print("(scratch2 copy skipped:", e, ")")
    print(f"wrote {master} -> {n} flights, {len(out)} cols")
    for k, v in sorted(byreg.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>4}  {k}")


if __name__ == "__main__":
    main()
