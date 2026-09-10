# Reproducing the measurements and the ablations

This is the operational companion to [`feedback_loop_reference.md`](feedback_loop_reference.md),
which explains what the co-design loop *is* and what the study *claims*. This document is
about how to **re-run it and check it**: the K1 measurements, the ablation of the
experiments (which rungs, and why those), and the ablation of the feedback (which half of
the loop is doing the work).

Related, and not duplicated here:

| for | read |
|---|---|
| what the loop is, and the argument | [`feedback_loop_reference.md`](feedback_loop_reference.md) |
| board access, toolchain, one model at a time | [`k1_board.md`](k1_board.md) |
| the loop stage by stage, and the graph-rewrite arm | [`codesign_loop_reproduction.md`](codesign_loop_reproduction.md) |
| porting to a target that is not the K1 | [`REPRODUCE.md`](REPRODUCE.md) |
| the w4/w5 rungs specifically | [`w4_w5_inner_outer.md`](w4_w5_inner_outer.md) |

Everything below runs from a checkout at any path. Two interpreters are involved:
`$REPO/.venv/bin/python` for the scheduler and the analysis (override with `XPURT_PY`),
and a separate Isaac interpreter for the flight sims (`ISAAC_PY`), which is only needed
for the HIL figures and not for anything in this document.

---

## 0. The short version

```bash
# the inner loop alone -- no board, deterministic, ~2 minutes
.venv/bin/python scripts/run_codesign_loop.py \
    --workload data/toplevel/scaling/s5_solvable_reveal.json \
    --solver greedy --max-rounds 4 --replay

# the whole outer loop on real silicon -- needs the K1, ~40 minutes
scripts/run_board_arc.sh --workload data/toplevel/scaling/s5_solvable_reveal.json \
    --stem s5_arc --models mlp_control,fused_full,ffn_block,dronet,yolov8_nano_64x96

# the 2x2 feedback ablation, both solvers
.venv/bin/python scripts/ablate_feedback_loops.py \
    --workloads data/toplevel/scaling/{s5_solvable_reveal,w4_ffn_dronet_sensor}.json \
    --calibration results/codesign_feedback/k1_cal_s5_measured.json \
    --solvers cpsat,greedy --cpsat-time-limit 150 --out-dir results/loop_ablation_postfix
```

`--replay` is the mode someone without our hardware can check: it pins
`XPURT_CPSAT_WORKERS=1` and refuses anything that would touch the board, so two runs of
the same inputs produce byte-identical schedules.

---

## 1. Reproducing the K1 measurements

### 1.1 The chain, and why it is a script

`scripts/run_board_arc.sh` runs five steps, each needing the previous one's output path.
It is a script rather than a runbook because doing it by hand is where the `--backends`
arity and the `--staged-ir` flags kept getting fumbled.

```
  0. inner loop            run_codesign_loop.py         -> a converged spec
  1. solve                 run_xpurt_schedule.py        -> a schedule
  2. execute on the K1     ModelBlaster/run_xpurt_k1.sh -> a trace CSV
  3. calibrate             emit_board_calibration.py    -> measured multipliers
  4. attribute             attribute_board_misses.py    -> execution vs queueing
  5. re-cost and re-solve  run_codesign_loop.py         -> the four-beat arc
```

Steps 2–4 are the outer loop proper. Steps 0–1 are the inner loop, and can be run alone
on any machine.

### 1.2 Four things the chain has to get right

Each of these was a bug before it was a rule, which is why each is enforced in code
rather than described in prose.

**The converged spec is the last ACCEPTED round, not the newest file.** The loop writes a
candidate spec for every lever it *tries*, so the final round — the one that accepts
nothing and thereby ends the search — leaves the most recent files on disk, and those are
rejects. Picking by mtime once sent a *rejected* round-3 candidate (`shard:dronet`, which
the accept rule had just refused) to the board. The driver reads `loop_report.json`.

**`--backends` takes one entry per core kind, not per model.** `rvv_x60,rvv_x60` is
`cpu_p,cpu_e`, and it stays two entries whether you run one network or five.

**Calibrate from the run's own trace, every time.** A calibration measured on a different
rung is an extrapolation. Reusing `b5z`'s table for `s5` mispredicted it by 1.66× — which
is exactly the class of error the outer loop exists to catch, so importing it at the
calibration step defeats the purpose. There are nine calibrations in
`results/codesign_feedback/` for this reason, one per board run.

**Pass `--schedule` to the calibrator.** The runner records zero-cost ops with
`dispatch_id -1` and numbers the rest from zero, while the schedule numbers all of them,
so on a network with zero-cost ops the two numberings diverge. Ratios are computed within
a row and are always right; only the *key* can be wrong, and that is invisible in the
emitted table. `--schedule` turns on the alignment check, and a network that fails it
loses its per-dispatch keys rather than keeping plausible-looking wrong ones.

### 1.3 What was measured

Nine board runs, each calibrated from its own trace, all alignment-verified:

| rung | dispatches | mean inflation | median | nets exact | trace |
|---|---:|---:|---:|---:|---|
| `c2` | 109 | 1.132× | 1.069× | 2 | `c2_aot_sched_trace.csv` |
| `c3` | 289 | 1.323× | 1.134× | 3 | `c3_aot_sched_trace.csv` |
| `b4` | 394 | 1.312× | 1.149× | 4 | `b4_aot_sched_trace.csv` |
| `b5x` | 333 | 1.429× | 1.202× | 4 | `b5x_aot_sched_trace.csv` |
| `b5y` | 333 | 1.434× | 1.204× | 4 | `b5y_aot_sched_trace.csv` |
| `b5z` | 333 | 1.448× | 1.183× | 4 | `b5z_aot_sched_trace.csv` |
| `s5` | 207 | 1.728× | 1.329× | 4 | `s5_arc_sched_trace.csv` |
| `w4` | 394 | 1.304× | 1.135× | 4 | `w4_aot_sched_trace.csv` |
| `w5` | 484 | 1.348× | 1.221× | 4 | `w5_shard_sched_trace.csv` |

Read the **median** as the typical dispatch and the **mean** as the one the scheduler has
to survive; they differ because inflation has a tail. The mean is what the calibration
applies, deliberately.

Two properties of these numbers worth stating precisely, because they bound what the
calibration means:

* **Execution only.** Queue delay is carried separately by the trace and excluded. A
  dispatch that *waited* is not a dispatch that *ran slowly*, and folding queueing in
  would charge the scheduler twice for its own placement.
* **Three tiers, degrading honestly.** `network/dispatch_id` where it was measured, op
  kind where it was not, aggregate as the floor. A net absent from `coverage.nets_exact`
  is costed by its op kinds, which is a *prediction* about that net rather than a
  measurement of it — and the JSON says so in its own `coverage.note`.

Timing is `rdtime` at 24 MHz. The pooled op tier and the aggregate are floored at
predicted ≥ 0.1 ms (2400 ticks) so timer granularity cannot inflate a pooled multiplier.

### 1.4 The four-beat arcs

The shape the study is about: **baseline misses → AOT clears → the board reveals →
the board re-solve clears again.**

| rung | arc | levers | ends at |
|---|---|---|---|
| `sensor_evo_auto` | 2 → 0 → **6** → **0** | shard + IME | **all met** |
| `sensor_evo_ime` | 2 → 0 → 1 → 0 | IME | all met |
| `s5_solvable_reveal` | 1 → 0 → 6 → **2** | shard | 2 residual |
| `b4_board_sized` | 10 → 3 → 4 → 4 | shard | 4 residual |
| `w4_ffn_dronet_sensor` | 10 → 5 → 4 → 4 | shard | 4 residual |
| `w5_ffn_dronet_yolo` | 11 → 7 → 16 → 14 | shard | 14 residual |

`sensor_evo_auto` is the one the figure uses: it is the only arc that applies **both**
levers (12 dispatches at width 2, 61 at width 4, plus IME) *and* returns to all-met.

The rows that do not return to zero are kept, not hidden, and step 4 of the chain says
why each one does not. `s5`'s residual is one execution-bound yolo instance — no
placement recovers a dispatch that cannot fit its window at any width — and `w5`'s is
larger because at 492 dispatches CP-SAT never proves phase 1 (see §2.3), so the re-solve
is optimising against an unconverged bound.

```bash
.venv/bin/python scripts/attribute_board_misses.py \
    --trace <trace.csv> --spec data/toplevel/scaling/s5_solvable_reveal.json \
    --json-out results/codesign_feedback/s5_miss_attribution.json
```

`execution_bound_instances: 0` means every remaining miss is queueing, and the loop is
accountable for it. Anything above zero is a compiler gap wearing a scheduler's clothes.
Measured: `b5y` 0, `s5` 1, `b4` 2, `b5x` 6, `w5` 6.

---

## 2. Ablation of the experiments: which rungs, and why

### 2.1 The problem the ladders solve

The 24 runnable K1 specs are bimodal, and neither mode tests a scheduler. Twelve have a
baseline that already meets every deadline — nothing at stake, so the loop can only win
on terms nobody reads. Most of the rest ask for something no schedule can deliver:
`yolov8_nano_64x96` needs 23.95 ms at 8 cores against a 22 ms window and its core scaling
has saturated (4→8 cores buys 1.6%), so 44 of the 50 residual misses in the full ablation
were infeasible by construction. Almost nothing sat in the band where a scheduler decides
whether the deadline is met.

`scripts/make_scaling_workloads.py` builds rungs that do. Every rung is constructed so
the singleton baseline **misses**, and a schedule that meets every deadline **exists**
using implementations already measured on the board — so a failure is the loop's, not
physics'.

```bash
.venv/bin/python scripts/make_scaling_workloads.py --out-dir data/toplevel/scaling --check
```

### 2.2 Five ladders, each isolating one variable

| ladder | varies | sized from | why it exists |
|---|---|---|---|
| `LADDER` (w2–w5) | net count | profiles | the first ladder; confounded (see below) |
| `LADDER_COMPOSITION` (c2–c5) | net count, `dronet` at the top only | profiles | isolates `dronet` to one row |
| `LADDER_BOARD` (b4, b5) | net count | **board** | profiles were wrong about what fits |
| `LADDER_BOARD_SLACK` / `_HOLDS` (b5x, b2y–b5y) | slack, cold start | board, 1-core cold | a window is only meetable at the width the solver *picks* |
| `LADDER_SOLVABLE` (s5) | dispatch count | board + CP-SAT tractability | reachable *and* solvable |

Each successor exists because its predecessor was wrong in an identifiable way, and the
docstrings in `make_scaling_workloads.py` carry the arithmetic. In brief:

* **`LADDER` confounded two things.** It introduces `dronet` at rung 3, and `dronet`'s conv
  dispatches take different core widths across their instances, so `shard` — the only
  lever in the whole ablation that ever clears a deadline — was refused as *unbuildable*
  on w4 and w5. All ten contract violations were `dronet`. The ladder therefore varied
  "more networks" and "contains the network that blocks our best lever" together, and the
  flat w4/w5 rungs were read as a scaling limit when the evidence pointed at codegen.
  `LADDER_COMPOSITION` puts `dronet` in exactly one rung so its effect is a single row.
* **Profile-sized windows do not survive execution.** `ffn_block` misses a 10 ms window on
  the board at its *fastest* measured width (10.03–10.34 ms against a 7.72 ms profile),
  and yolo misses 26 ms by 1.64×. The "0 infeasible misses" classification said otherwise
  only because it trusted profiles. `LADDER_BOARD` re-sizes from measurements.
* **A window is meetable only at the width the loop actually converges on.** `b5x` reaches
  0 predicted misses and does not survive execution: its windows were sized from
  measurements taken while `dronet` and `yolo` were *widened*, but the loop converged on
  `shard:ffn_block` alone and left both at one core. `LADDER_BOARD_HOLDS` sizes from
  1-core cold measurements — the conservative width, the one the loop falls back to.
* **Cold start is not free and is not a scheduling failure.** The first instance of a
  network pays it: `fused_full` runs 2.66–2.74× its warm median on instance 0 (11.7 ms
  against 4.28 warm). A 5 ms period cannot absorb that in any schedule. Deploying at that
  rate needs a warm-up pass, not a better scheduler; the rungs state it as a window.

### 2.3 Sizing for the solver, not only for the board

A scan of the 204 CP-SAT certificates in this repo says model size decided more of our
results than any scheduling question:

* a phase-1 objective of **0 is OPTIMAL in 55 of 55 runs** — when a zero-miss schedule
  exists the bound is matched trivially. All 121 FEASIBLE runs have objective > 0 and a
  genuine open gap. CP-SAT is fast at *confirming* an achievable target and slow at
  *disproving* an unachievable one;
* **n = 492 dispatches is never OPTIMAL (0/33)**, and above 300 only 2 of 73 files ever
  proved phase 1. The single yolo instance — 98 dispatches on its own — is exactly what
  takes w4 (394) to w5 (492);
* between those, node count is a weak predictor: n = 214 is mostly FEASIBLE while the
  larger n = 217 and n = 254 are mostly OPTIMAL. Hardness, not node count.

Dispatches per instance, measured: `mlp_control` 7, `fused_full` 15, `ffn_block` 5,
`dronet` 21, `yolov8_nano_64x96` 98. `s5_solvable_reveal` is sized to 215 dispatches from
these — the size at which the working rung sits.

**The consequence for reading w5:** its re-solve does not pay off, and the honest
statement is not that the outer loop fails at five networks. It is that at 492 dispatches
CP-SAT never converges, so the re-solve is optimising against an unproven bound. That was
an experiment-sizing mistake, and `s5` is the same five networks sized so the solver can
actually answer.

### 2.4 Gate a rung before running it

```bash
.venv/bin/python scripts/gate_rung.py \
    --spec data/toplevel/scaling/s5_solvable_reveal.json \
    --calibration results/codesign_feedback/k1_cal_s5_measured.json \
    --sweep-net ffn_block --windows 34,28,24,21,20,18 \
    --pin yolov8_nano_64x96=30.0 \
    --shard-sets yolov8_nano_64x96 yolov8_nano_64x96,ffn_block
```

Two gates, both empirical:

1. **at stake** — the solved baseline, on profile costs, misses at least one deadline;
2. **reachable** — some lever set, on *measured board* costs, reaches zero.

Both must pass on the same row. Gate 2 also buys tractability, per §2.3: a reachable rung
is also a solvable one.

**Why this is empirical and not arithmetic.** Three analytical gates in a row got it
wrong, each with correct arithmetic over a wrong model of the baseline:

* `one > window` ignored that a window may be wide for an unrelated reason (absorbing the
  t=0 cold-start burst), which then also clears the baseline;
* `one > period` is false on a multicore machine — successive *instances* are independent
  and the scheduler puts them on different harts. CP-SAT duly returned a zero-miss
  baseline by spreading `ffn_block`'s three instances;
* comparing `net_times()` to the window ignores that the scheduler parallelises a
  network's *dispatches* across cores even at width 1 each, so yolo's 47.73 ms serial sum
  never lands on a single core at all.

Solve it and look.

One more distinction the tool enforces, because hand-tuning kept breaking it: **period and
window are different knobs.** The period is what puts a rung at stake — `ffn_block` at
period 20 ms against a 26.61 ms single-core time means one core cannot sustain the release
rate, so widening is forced whatever the window says. The window is what makes it
reachable — 34 ms, wide enough that the widened schedule survives the t=0 burst when all
five networks release together. An earlier version sized the window at 28 to absorb the
burst, which also put it above the 26.61 ms single-core cost, so the baseline fitted on
one core and the band check rejected the rung as "nothing at stake".

---

## 3. Ablation of the feedback: which loop is doing the work

### 3.1 The design

Showing one workload where the pair helps does not separate them. `ablate_feedback_loops.py`
runs the 2×2, per solver:

| cell | inner (levers) | outer (solve on measured costs) | |
|---|---|---|---|
| **A** | no | no | the naive deployment |
| **B** | **yes** | no | offline co-design, deployed blind |
| **C** | no | **yes** | measure-and-re-solve, no co-design |
| **D** | **yes** | **yes** | both |

* **inner** = AOT co-design with ModelBlaster. Graph rewrites and the per-dispatch
  implementation choice, decided offline against isolated per-dispatch profiles. The board
  may be used here, but only as a **profiler**.
* **outer** = HIL. Costs observed while the whole schedule runs in real time, returned as
  measured multipliers and re-solved against. The board here is the **runtime**.

**Every cell is scored on board costs**, because that is what silicon does. The cells
differ in what the *scheduler knew*, not in how they are judged: A and B are solved
against predicted costs and then re-cost on the measured multipliers with their assignment
held fixed — exactly what deploying them means — while C and D are solved with
`--board-calibration`, so re-costing them again would apply the multiplier twice.

Scoring is instance-level (`xpu-rt/schedule_eval.py`): an instance misses when its last
dispatch ends past `inst*period + window`.

### 3.2 Running it

```bash
.venv/bin/python scripts/ablate_feedback_loops.py \
    --workloads data/toplevel/scaling/s5_solvable_reveal.json \
                data/toplevel/scaling/w4_ffn_dronet_sensor.json \
    --calibration results/codesign_feedback/k1_cal_s5_measured.json \
    --solvers cpsat,greedy --cpsat-time-limit 150 --repeats 3 \
    --out-dir results/loop_ablation_postfix
```

`--repeats` applies only to a solve whose result can *move*: a CP-SAT solve returning
OPTIMAL has a unique objective and runs once; one returning FEASIBLE is budget-truncated
and is repeated, reported as median with spread. Greedy is deterministic and always runs
once. `--one-per-family` collapses the 25 K1 specs, which contain byte-identical
duplicates and variants that return bit-identical results — the family is the real unit.

### 3.3 Which grid to read

There are **thirteen** ablation directories in `results/`, because the answer moved as
bugs were fixed underneath it and the superseded grids are the record of that.

**Read `results/loop_ablation_postfix/`.** It is the only grid run after the accept-path
fixes — the measurement tolerance no longer blurs a deterministic miss count, and the
winner is ranked by misses rather than by makespan (term 7 of 9) — and after the CP-SAT
budget split. The others: `loop_ablation` is the broad 24-workload greedy sweep,
`_ladder*` are the scaling rungs, `_v4_*`/`_v5_big` vary model size, and `_pair`,
`_stable` and `_final*` are intermediate grids kept so a changed number can be traced to
the change that caused it.

Each `ablation_summary.json` carries its own `cpsat_time_limit_s`, `cpsat_workers`,
`solve_env` and `codegen_contract`, so a cell cannot be silently compared against one
solved under different conditions. Check them before comparing across directories.

### 3.4 What it found

Instance misses on board costs, `loop_ablation_postfix`:

| workload | solver | A | B | C | D | inner levers chosen |
|---|---|---:|---:|---:|---:|---|
| `s5_solvable_reveal` | greedy | 1 | **0** | 1 | **0** | `shard` |
| `s5_solvable_reveal` | cpsat | 1 | 1 | 1 | 1 | *(none accepted)* |
| `w4_ffn_dronet_sensor` | greedy | 10 | **4** | 10 | 8 | `shard:ffn_block` |
| `w4_ffn_dronet_sensor` | cpsat | 23 | 10 | 28 | 26 | `shard:dronet`, `shard:fused_full` |

Three things this says, stated as plainly as they deserve:

**The inner loop carries this result.** B beats A on every row that moves. On these two
at-stake workloads, offline co-design against isolated profiles is what clears deadlines.

**The outer loop alone does approximately nothing here.** C ≈ A throughout. Re-solving
against measured costs without any lever to pull does not help, which is unsurprising in
retrospect: better cost estimates do not create an implementation that fits.

**D is not better than B, and on `w4`/greedy it is worse (8 vs 4).** We do not have an
explanation we are confident in, and it is reported rather than smoothed. Note the
tension with §1.4, where the *arcs* show the outer loop recovering board-revealed misses
on `sensor_evo_auto` and `s5` — the arc and the ablation ask different questions (does the
re-solve recover what the board revealed, vs. does the pair beat the inner loop alone),
and the honest summary of the pair, on this population, is that it does not.

**CP-SAT loses to greedy in every cell** (`exact_vs_greedy_per_cell`: B, C, D all 0 wins /
2 losses; A 1–1). At these sizes the exact solver does not earn its cost against the
heuristic. That is a statement about these two workloads at this budget and this
dispatch count, not about exact scheduling.

**n = 2.** The at-stake population is two workloads. These are directional findings from
a small sample, and the ablation reports `aggregate_at_stake_only` separately from
`aggregate_achievable_only` precisely so a reader can see how few rows carry the claim.

### 3.5 Separating a miss the loop could have cleared from one it could not

`achievable_means` in the summary: *a miss is ACHIEVABLE when the net's fastest measured
implementation fits its window; otherwise no schedule can meet that deadline and the miss
is a compiler gap, not a scheduling one.* `total_infeasible_misses` is 0 across every cell
in `postfix`, which is what makes those rungs fair tests — and is exactly the property the
profile-sized rungs lacked (§2.2).

---

## 4. What these measurements do not establish

* **Single board, single unit.** All nine runs are one SpaceMiT K1. Nothing here separates
  silicon-to-silicon variation from anything else.
* **The calibration is not transferable.** Measured per rung, and demonstrated not to
  transfer: `b5z`'s table mispredicted `s5` by 1.66×. Treat a multiplier as a statement
  about the workload it was measured on.
* **Op-kind and aggregate tiers are predictions.** Only `coverage.nets_exact` is measured
  per dispatch. A net costed by its op kinds is being predicted, and the JSON labels it.
* **Queueing is excluded from the calibration by construction.** It is in the trace and in
  the attribution, but not in the multipliers.
* **The ablation population is two at-stake workloads.** See §3.4.
* **ROS was never executed on the board, and "ROS" means three different things.** This is
  the largest gap in the comparison and is stated first because a reader will find it.
  All ten board traces are XPU-RT schedules; `find ModelBlaster/tmp -name '*_trace.csv'`
  returns zero ROS runs. The three arms are:

  1. *The scheduling baseline* (`scripts/ros_pinning_generic.py`,
     `ros_pinning_periodic.py`) — a **policy model, not ROS software**. It takes XPU-RT's
     real measured per-dispatch durations and re-lays them out under ROS's serialization
     policy: one node per network, pinned to one hart, whole dispatch graph sequential,
     periodic timer releases. Every `schedules/cmp_*ros*_board.json` is this, and the
     `_board` suffix means board-*calibrated costs*, not board-*executed*.
  2. *The flight-sim arm* (envelope, showdown, crash demo) — in-sim ZOH latency injection.
     The knobs are the CSV's own columns: `sched_latency_ms`, `percep_latency_ms`,
     `percep_hold_ms`, `eff_cmd_hz`, `pipeline_zoh`. A latency model parameterised to
     stand for ROS, flown in Isaac.
  3. *micro-ROS on FPGA* — a different target (FireSim), not the K1.

  **What this does support.** Because the baseline pays the *same measured per-op costs*
  as XPU-RT, the comparison isolates the serialization policy: it cannot be dismissed as
  a slow ROS build or unoptimised kernels. On the compute side it is a best case for ROS.
  The defensible claim is *"static per-node serial pinning loses to global scheduling at
  equal per-op cost"*.

  **What it does not support.** *"ROS 2 loses to XPU-RT on the K1"* — we have not measured
  that. The model omits DDS serialization, message copies, executor wake-up latency and
  callback jitter, which would make real ROS **worse** than our arm; but it also assumes
  single-threaded per-node execution, and a real deployment using a multithreaded executor,
  callback groups or composed nodes would be **better** than our arm. That second
  direction is the one that cuts against us, and we cannot currently bound it.

  Related: the `12.40 ms` ROS figure is a **literal** in `scripts/hil_ablation_phase.py`
  (`SCHEDS = [("ROS · per-net pinning", 12.40, ...)]`), not derived at plot time, so it
  will not move if the schedule it came from changes.

  **The cheap experiment that would close most of this.** The ROS-pinned schedules are
  valid schedule JSONs, so `ModelBlaster/scripts/run_xpurt_k1.sh --schedule` can execute
  one. That makes the baseline board-*executed* rather than board-*costed* and captures
  the real queueing and contention it currently only predicts. It is still not ROS
  middleware — for that, ROS 2 nodes have to run on the board's Bianbu Linux — but it
  removes the weaker of the two objections. Neither has been done.
* **`yolov8_nano_64x96` core scaling is measured only in contention.** In the five-network
  runs yolo was sharded per dispatch (48 dispatches at width 4, 17 at width 2, 33 at width
  1 — never width 8) while contending with four other networks. That is not the standalone
  1/2/4/8-core sweep that Figure 14's extrapolated multipliers would need to be confirmed
  against, and it should not be presented as one. `MB_CORES` in
  `ModelBlaster/scripts/run_model_k1.sh` supports the clean experiment; it has not been run.

---

## 5. Environment and gotchas

| variable | what it does |
|---|---|
| `XPURT_PY` | interpreter for the scheduler and analysis (default `$REPO/.venv/bin/python`) |
| `ISAAC_PY` | interpreter for the flight sims (HIL figures only) |
| `XPURT_NO_COMPACT=1` | keep the schedule as solved; the miss count is the solver's answer, not a post-pass's |
| `XPURT_UNIFORM_PACKED_WIDTH=1` | a packed-weight dispatch takes one width across its instances — the codegen contract's requirement |
| `XPURT_CPSAT_WORKERS` | CP-SAT workers. 4 for cells, 1 under `--replay` for bit-exact reruns; **mixing the two inside one row confounds inner-vs-outer with worker count** |
| `XPURT_SHARD_ONLY_NETS` | restrict the shard lever to named nets. Published *before* any solve — a lever published only at the CP-SAT call site never reaches `--solver greedy`, which made per-net shard candidates come out byte-identical |
| `MODELBLASTER_K1_HOST` | ssh config entry for the board (default `k1`) |
| `CROSS` | SpaceMiT cross toolchain, set by `scripts/setup_spacemit_toolchain.sh` |

Two more that cost real time:

**Do not use `pgrep -f` to wait on your own jobs.** It matches its own command line. It
killed the shell mid-heredoc three times in this study and lost a script. Use PID files.

**Absolute paths under `results/` are records, not instructions.** They say what was
actually run. The paths in `scripts/` are all repo-relative; if you find one that is not,
that is a bug.

---

## 6. Where the artifacts are

```
results/codesign_feedback/
  k1_cal_<rung>_measured.json     the nine board calibrations (§1.3)
  <rung>_miss_attribution.json    execution-bound vs queueing (§1.4)
  <arc>/                          per-arc loop output, panels, Gantts
  loop_overview_2band.{png,pdf}   the figure, with its _metrics.json sidecar
results/loop_ablation_postfix/    the 2x2 grid to read (§3.3)
results/loop_ablation*/           twelve superseded grids, kept as the record
results/loop_sweep/<workload>/    25 workloads taken through the loop end to end
data/toplevel/scaling/            the generated rungs
```

Every `loop_report.json` names the round, the lever, the objective terms that decided and
whether the candidate was accepted — so a claim of the form "the loop chose `shard` here"
can be checked against the round that made the choice rather than taken on trust. Every
figure has a `_metrics.json` sidecar carrying the numbers it drew.
