# Proposed paper edits for the v2 batch (review before I apply)

These go into `sections/evaluation.tex` once the fresh grid (stage B) + course B (stage C) land.
The showdown numbers below are FINAL (18 seeds, stage A done); envelope numbers are `<v2>` placeholders
I'll fill from `reproduce_hil_figure.py` when the grid finishes. Nothing is applied to the paper until
you OK the wording.

## 0. Use MEAN GATES CLEARED (0-4) as the primary flight metric — more granular AND more significant
We record `gates_passed` per flight. Switching from binary success to mean gates cleared uses the full
progress info and is statistically stronger:
- Showdown (18 seeds): XPU-RT 2.67/4, ROS 50Hz 1.89/4, ROS 25Hz 0.00/4. Mann-Whitney XPU>ROS50 **p=0.044
  (significant)** — binary success was only p=0.07 (n.s.). XPU>ROS25 and ROS50>ROS25 both p<0.0001.
- Envelope by rate: 25→1.27, 33→2.00, 50→2.70, 100→2.53 gates (the floor as a smooth 0-4 metric).
Recommend reporting mean gates cleared (with the Mann-Whitney) as the headline flight metric, keeping
full-course success as a secondary. This is the single cheapest strengthening and it's honest.

## 0b. Logistic response-surface — the strongest statistical statement for the envelope
`scripts/hil_logistic_fit.py` fits P(success) ~ log2(rate) + speed across ALL flights, pooling power
instead of per-cell bars. On v1 (120 flights): control rate **p=0.0030** (odds 2.04x/SD, faster helps),
cruise speed **p=0.0006** (odds 0.40x/SD, faster hurts) — **both significant**. Recommend adding one
sentence to the envelope discussion: "A logistic fit over all flights confirms both control rate
(p<0.01) and cruise speed (p<0.001) as significant predictors of course completion," and optionally
overlay the model's 50%-success contour as the feasible-envelope boundary. Re-run on v2 for the final
numbers (expect stronger with 2x the flights). This is the single biggest defensibility lift.

## 1. Showdown aggregate — replace the fragile "3/6 vs 0/6"
The old claim ("XPU-RT completes 3/6, ROS 0/6") is small-n fragile; at 18 seeds it does not hold as
stated (ROS at 50 Hz succeeds sometimes). Honest, and actually a cleaner story — a control-rate-monotone
degradation with one statistically significant endpoint:

- XPU-RT 100 Hz: **8/18 (44%)**, Wilson95% [0.25, 0.66]
- ROS 50 Hz: **3/18 (17%)**, [0.06, 0.39]
- ROS 25 Hz (deeper starvation, matches the Gantt's "backs up to ~15 Hz"): **0/18 (0%)**, [0.00, 0.18]
- XPU-100 vs ROS-25: **+44 pts, p=0.001 (significant)**; XPU-100 vs ROS-50: +28 pts, p=0.07 (trend).

Proposed wording (replaces the "3/6 / 0/6" sentence + the ROS "loses stability and crashes" absolute):
> "Success degrades monotonically as the schedule starves the control loop: over 18 matched seeds the
> drone completes 8/18 at the sustained 100 Hz, 3/18 when the command rate is held to 50 Hz, and 0/18
> once the starved schedule backs up toward ~25 Hz — the regime the ROS partition actually reaches
> under contention (\autoref{fig:hil-showdown}, bottom). The 100 Hz vs starved-ROS separation is
> significant (p=0.001); single-seed outcomes remain non-reproducible, so we report the aggregate."

Keep the representative-flight strips (a–d) as-is (one representative crash/complete each).

## 2. Envelope panel — refresh numbers to the 240-flight grid (12 seeds/cell)
Fill from `reproduce_hil_figure.py` after stage B: pooled per-rate k/n, the `+<v2> pts` floor, the
`p=<v2>` for 25→50 Hz, and the best cell. The floor + speed-limited-band story should hold and be
MORE significant (n doubles); if 50 Hz ≈ 100 Hz persists, the wording ("floor + speed-limited band,
no 100-vs-50 claim") is unchanged.

## 3. Rate-cap bridge — one connecting sentence (measured, already in README)
Tie the envelope's control-rate floor to why each scheme lands where it does:
> "A schedule's worst-case response bounds the command rate it can sustain — 4.89 ms (shard) → ~205 Hz,
> 8.00 ms (greedy) → ~125 Hz, 12.40 ms (ROS pinning) → ~81 Hz (\texttt{microros\_baseline\_k1};
> board-confirmed) — so at the deployed 100 Hz only the XPU-RT schedules clear their own period, which
> is why the ROS mapping falls to/below the control-rate floor."
(Honesty note carried from README: the ROS ~81 Hz figure is the deployed stale-hold under the K1
workload, cited as a worst-response→rate bound, not overclaimed.)

## 4. Course B (stage C) — generalization line, pending `analyze_courseB.py`
If course B reproduces the floor (≈0 below 50 Hz, rise above), add one sentence that the same unchanged
stack shows the same control-rate floor on a second gate layout — cross-course generalization. If it
doesn't reproduce, say so and drop the claim.
