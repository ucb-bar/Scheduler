# HIL flight-envelope data — what it is, how to reproduce the figure, and its scope

This directory holds the raw per-flight data behind the warehouse HIL figure (the flight-envelope
error-bar panel and the single-flight showdown). Every plotted number is regenerated from these
CSVs by one script — nothing in the figure is hand-entered.

## Reproduce every plotted number
```
<env_isaaclab>/python scripts/reproduce_hil_figure.py           # prints the stats
<env_isaaclab>/python scripts/reproduce_hil_figure.py --render  # also re-renders the panel PNG
```
It prints the per-cell counts, the pooled-over-speed trend with Wilson 95% intervals, the caption
checks (`+37 pts`, `p=0.0006` for the 25→50 Hz rise, best cell `4/6`), and the showdown `3/6` / `0/6`.

## The two data sources (they are DIFFERENT experiments — read this before comparing them)
Every flight is one real headless Isaac-Lab run of `sims/scripts/sweep_rate_demo.py` (RL/MLP
controller `nav_fused_v12_cnn.pt`, YOLOv8n perception, fixed 4-gate warehouse course). What differs:

| | **Envelope** (`hil_ablation.csv`) | **Showdown** (`crash_verify/new_*.csv`) |
|---|---|---|
| what it is | the error-bar panel | the single-flight `3/6` vs `0/6` |
| control rate set by | clean ZOH **decimation** (`sched_latency_ms` 8/18/28/38 → 100/50/33/25 Hz) | real worst-case **`sched_latency_ms`** (XPU 4.89, ROS50 12.40, ROS25 35.58 ms) |
| controller gain | **fixed** `moment_scale=0.0055` | **calibrated** `moment_scale = 0.5/eff_hz` (0.005@100, 0.01@50, 0.02@25) |
| grid | 5 speeds × 4 rates × 6 seeds = 120 | cruise 1.4, 6 seeds each |

**Why the envelope's 50 Hz cell (3/6) ≠ the showdown ROS (0/6):** they are not the same condition.
The showdown ROS run is the *starved, jittered* schedule (worst-case latency ZOH), which is strictly
harder than a clean decimated 50 Hz sweep point, and it uses the calibrated gain. So a lower success
count there is expected, not a contradiction. The XPU 100 Hz `3/6` vs the envelope's `4/6@1.4/100`
is seed noise at n=6. **Do not pool the two sources or read them cell-for-cell.**

## What the data does and does not claim (honest scope)
- The **defensible, significant** result is the *pooled* trend: a control-rate **floor** — 25 Hz ≈ 3%
  rising to 50 Hz ≈ 40% (25→50 Hz, `p<0.001`, n=30/rate) — and success falling with cruise speed.
- **Not** claimed: 100 Hz > 50 Hz. Pooled they are a plateau (0.40 vs 0.33, CIs overlap, n.s.). The
  figure and caption say "floor + speed-limited band," never "100 beats 50."
- **n = 6 per cell** → per-cell intervals are wide; lean on the pooled n=30 marginals, not single cells.
- **Single-seed outcomes are not individually reproducible** (Isaac/CUDA nondeterminism); only the
  aggregate over seeds is stable — hence we always report k/n, never a single representative success.
- The envelope's 25 Hz collapse is *partly* under-authority from the fixed gain; the pure-rate version
  (calibrated-gain dense grid) is not part of this figure.

## Data versions (Sept 2026)
- `hil_ablation_v1_120flights_fixedgain.csv` — the original **120-flight** envelope (6 seeds/cell),
  fixed gain, produced by the earlier sim. **Archived permanently** (sha `706fd92…`); never deleted.
- A **fresh full grid at 12 seeds/cell (240 flights)** is being regenerated from the *canonical*
  repo with one consistent sim version (→ `hil_ablation_v2.csv`); once verified it **supersedes**
  `hil_ablation.csv` as the figure source. `reproduce_hil_figure.py` recomputes whatever is current,
  so the printed numbers (and the caption) update with the new data. v1 stays for provenance.
- `hil_ablation_courseB.csv` — the same sweep on a *second gate course* (`WAREHOUSE_COURSE=b`), for
  cross-course generalization. **Never pool it with course A** (different layout).

## Why the rate axis has a floor — measured scheduler rate-caps (the bridge)
The envelope shows *that* success needs a control-rate floor; the scheduler measurements show *why
each scheme lands where it does* on that axis. The critical loop must finish inside one flight-loop
period, so a schedule's worst-case response bounds the command rate it can sustain
(`microros_baseline_k1/flyfaster_crash_band.json`, corroborated by real-board runs in
`k1_feedback_exact/board_result.json` and the 1.26× board calibration `k1_board_calibration.json`):

| schedule | worst-case response | sustainable rate |
|---|---|---|
| ROS per-net pinning | 12.40 ms | ~81 Hz |
| XPU-RT greedy | 8.00 ms | ~125 Hz |
| XPU-RT shard (feedback) | 4.89 ms | ~205 Hz |

At the deployed **100 Hz**, only the XPU-RT schedules clear their own period; the ROS pinning cannot,
which is why it falls to/below the envelope's control-rate floor and crashes. This is the principled
link from the flight envelope to the scheduling contribution — both measured, not asserted.
*Honesty note:* the ROS 12.40 ms → ~81 Hz figure reflects the deployed per-net-pinning stale-hold
under the K1 multi-model workload, not a fresh per-flight derivation; cite it as a worst-response→rate
bound under that workload, and pair it with the board evidence above rather than overclaiming.

## Files
- `hil_ablation.csv` — the envelope panel (the committed figure source; v1 today, v2 after the refresh).
- `hil_flights_master.csv` — **435 flights** unioned across every experiment, with `source`, a
  `regime` column, and a `success` flag. Regimes: envelope (120), calibrated-gain grid (96), showdown
  (18), perception-freshness/safety (116, from the canonical `perc_crash/` sweeps), verification (85).
  The `regime` column exists so these are **never silently pooled** — the figure uses only the
  envelope + showdown regimes; `perc_crash` is a separate perception-freshness study. *(Checked:
  within `perc_crash`, perception hold 22 ms vs 65 ms did NOT separate success — fresh22 11/16 vs
  stale65 11/16, xpu_safe 10/16 vs ros_safe 10/16 — a null on that knob. Kept for breadth; no claim
  is built on it.)*
- `crash_verify/new_xpu.csv`, `new_ros50.csv`, `new_ros.csv` — the showdown seeds (6 each; 18 after refresh).
- `crash_verify/run_newshowdown.sh`, `run_ros50.sh` — the exact showdown invocations.
- `scripts/hil_ablation_grid.sh` — the envelope grid driver (regenerates `hil_ablation.csv`).

## Column dictionary (`hil_ablation.csv`; the master adds `source` + `success`)
- `seed` — RNG seed (also drives the randomized obstacle/people/crate field this flight sees).
- `cruise_speed` — commanded forward speed (m/s).
- `sim_dt`, `decimation`, `control_dt_ms` — physics step, control decimation, resulting control period.
- `sched_latency_ms` — worst-case command latency; the motor command is held (ZOH) between refreshes.
- `hold_steps` — control steps per refresh = `ceil(latency / control_dt)`.
- `eff_cmd_hz` — effective command rate = `1000 / (control_dt_ms × hold_steps)` (the x-axis).
- `moment_scale` — controller action→moment gain (0.0055 fixed for the envelope).
- `gates_passed` — gates cleared (of 4); `steps` — sim steps survived; `K` — total gates.
- `outcome` — `success` (all gates) / `crash` / `timeout`.
- master extras: `walk_speed, gust, motor_tau, pipeline_zoh, percep_*` (blank unless that run logged them),
  `crash_type`, `success` (1 iff `outcome==success`).
