#!/usr/bin/env python3
"""Recompute every quantitative claim on warehouse_showdown_story from the CSVs.

Run this before quoting the figure. It reproduces the success counts, tests every pairwise
rate contrast with Fisher's exact test, and separates the contrast the figure ANNOTATES
from the contrast its headline IMPLIES -- which are not the same contrast, and only one of
them is supported:

  * 25 -> 50 Hz  is +36.7 pts, p = 2.7e-07 on course A. This is the "+37 pts" annotation,
    and it holds. There is a real control-rate FLOOR between 33 and 50 Hz.
  * 50 -> 100 Hz is -3.3 pts, p = 0.85 on course A and +3.3 pts, p = 1.00 on course B.
    Above the floor, rate does not measurably matter -- so a headline contrasting a 100 Hz
    arm against a 50 Hz arm is not supported by this data, on either course.

The defensible claim is the floor, not the 50-vs-100 comparison.
"""
import csv, collections, itertools
from scipy.stats import fisher_exact

def load(p):
    rows = list(csv.DictReader(open(p)))
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        hz = round(float(r["eff_cmd_hz"]))
        by[hz][1] += 1
        by[hz][0] += (r.get("outcome") == "success")
    return by, len(rows)

for name, label in (("hil_ablation.csv", "COURSE A"), ("hil_ablation_courseB.csv", "COURSE B")):
    by, n = load(f"results/codesign_feedback/{name}")
    print(f"\n{'='*74}\n{label}  ({n} flights)\n{'='*74}")
    for hz in sorted(by):
        s, t = by[hz]
        print(f"  {hz:>4} Hz   {s:>3}/{t:<4} = {s/t:.3f}")
    print(f"\n  {'comparison':<22} {'delta pts':>10} {'Fisher p':>12}   verdict")
    for a, b in itertools.combinations(sorted(by), 2):
        sa, ta = by[a]; sb, tb = by[b]
        d = (sb/tb - sa/ta) * 100
        _, p = fisher_exact([[sb, tb-sb], [sa, ta-sa]])
        v = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {a:>3} Hz -> {b:>3} Hz      {d:>+9.1f}  {p:>12.2e}   {v}")

    # the claim the figure makes vs the claim the headline makes
    print()
    if 50 in by and 100 in by:
        s5, t5 = by[50]; s1, t1 = by[100]
        _, p = fisher_exact([[s1, t1-s1], [s5, t5-s5]])
        print(f"  HEADLINE TEST  100 Hz vs 50 Hz: {s1}/{t1}={s1/t1:.3f} vs {s5}/{t5}={s5/t5:.3f}  "
              f"delta {(s1/t1-s5/t5)*100:+.1f} pts, p={p:.3f}")
    # pooled: under-rate (25,33) vs feasible (50,100) -- what the shaded bands imply
    lo = [sum(by[h][i] for h in by if h < 42) for i in (0, 1)]
    hi = [sum(by[h][i] for h in by if h >= 42) for i in (0, 1)]
    _, p = fisher_exact([[hi[0], hi[1]-hi[0]], [lo[0], lo[1]-lo[0]]])
    print(f"  POOLED (what the bands imply)  under-rate {lo[0]}/{lo[1]}={lo[0]/lo[1]:.3f} vs "
          f"feasible {hi[0]}/{hi[1]}={hi[0]/hi[1]:.3f}  delta {(hi[0]/hi[1]-lo[0]/lo[1])*100:+.1f} pts, p={p:.2e}")
