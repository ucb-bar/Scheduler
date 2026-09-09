# The co-design feedback loop: a reference for writing it up

Written so someone who did not run these experiments can choose what to claim. Every
number here is from an artifact in this repo; where a number is uncertain or where an
earlier claim turned out to be wrong, it says so, because those are the places a reviewer
will push.

Contents: what the study asks · the inner loop in detail · the outer loop in detail · the
platform and workloads · results · what can and cannot be claimed · corrections made.

---

## 1. What the study asks

A robotic workload is several neural networks that must each finish inside a repeating
time window on one heterogeneous SoC. Two questions decide whether it can be deployed:

* **Which compiler transformation should be applied?** Widen a network across cores, move
  its matrix work to an accelerator, split or fuse its graph. This is a *compiler*
  decision but its payoff is only visible in a *schedule*.
* **Is the answer still right on real silicon?** Offline decisions rest on per-operator
  profiles. If the board is slower than the profile, a schedule that met every deadline
  in prediction can miss on hardware.

The system closes a loop for each. The **inner loop** decides transformations offline
against predicted costs. The **outer loop** executes the result on the board and returns
measurements. The study measures what each contributes, separately and together.

---

## 2. The inner loop, in detail

Offline. No hardware in the path, so it is bounded by compile and profile time.

### 2.1 Candidate generation

Each round proposes one *candidate design* per available lever, applied on top of the
current design:

| lever | what it changes | how |
|---|---|---|
| `shard:<net>` | widens ONE network across harts, holding the others at one core | spec: `machine_combination_mode: shard` plus `scheduler.shard_only_networks` |
| `shard` | widens every network at once | spec, global |
| `ime` | allows matrix dispatches on the K1's IME engine | spec: `enable_impls` |
| `unfuse` | points a network at ModelBlaster's unfused dispatch graph | swaps `dispatch_deps_path` |
| rewrite verbs | genuinely edits the IR — fuse / split | ModelBlaster rewrite arm; regenerated, host-verified bit-exact, re-profiled |

`shard` being **per network** matters. It began as one global switch, which forced the
loop to widen everything or nothing. On the five-network rung that made the lever
unusable: widening also widened YOLO — 191.6 core-ms that monopolises all eight harts
while the 5 ms-period networks wait — so worst-case lateness went 24.67 → 34.87 ms and
the whole lever was rejected, including the part that helped. Per-network candidates
compose across rounds, so the loop *discovers* the set `{ffn_block, dronet}` rather than
guessing it.

### 2.2 Evaluation: candidates are solved, not estimated

Every candidate is scheduled by the real solver (CP-SAT or the greedy list scheduler) and
scored on the resulting timeline. Nothing is judged by a cost proxy.

### 2.3 The acceptance rule

`xpu-rt/candidate_objective.py:accept()`. Nine terms, strict lexicographic order:

1. hard deadline misses  2. max deadline lateness  3. frequency compliance
4. p99 response of critical tasks  5. heavy-model max latency  6. heavy-model throughput
7. **makespan**  8. utilisation  9. standalone kernel cycles

Deadline misses first, makespan **seventh** — a candidate that finishes sooner but misses
one more deadline loses. Each term carries a tolerance, and **a tie is a rejection**: a
transformation that cannot be shown to help is not worth the rebuild.

Two defects here were costing a lever each, and both are worth a sentence in the paper
because they are easy for a reader to suspect:

* **A measurement tolerance was applied to a deterministic count.** `miss_rate_frac` is 8%
  because seven repeated *board runs* of one schedule gave 7–9 misses of 38 — real
  execution jitter. But in the AOT search a candidate's misses are computed from a solved
  schedule against fixed costs: deterministic, with no run-to-run spread (two independent
  full loop runs are bit-identical). Applied there it cost exactly the improvements the
  loop exists to find — on one rung, 8% of 34 instances is 2.72 misses, so a candidate
  taking misses 5 → 3 was reported "indistinguishable on every term". Fixed with
  `DETERMINISTIC_TOLERANCES`, which zeroes the miss tolerance and keeps the continuous
  ones (lateness, p99, makespan do move between identical CP-SAT runs under a time limit).
* **The winner among accepted candidates was ranked by makespan**, the seventh term, while
  the comment claimed it minimised "the objective". Fixing the tolerance exposed it: with
  more candidates accepted per round, one rung's final went from 7 misses to 9. Ranking
  now mirrors the nine-term order.

### 2.4 The buildability veto

Before performance is considered, `xpu-rt/codegen_contract.py:violations()` rejects
candidates the code generator cannot build. The rule that bites is
**`uniform_width_across_instances`**: weights are packed per core width, and one generated
model cannot hold two layouts, so a dispatch cannot run at width 1 in one periodic
instance and width 4 in another.

This silently discarded the best schedules. On one rung greedy's shard candidate — 5
misses against the baseline's 10, worst lateness 3.09 ms against 17.95 — was thrown away
over **three** dronet dispatches. Two mechanics are worth knowing:

* The width must be chosen by **measured** cost. "Widest" is wrong on real data (YOLO's
  OC=2 detect-head convs measure *slower* on four cores than one) and "narrowest" silently
  undoes sharding.
* The width must be **priced out**, not flagged. The list scheduler never reads
  `infeasible_combinations` at all — flagging alone left dronet mixing widths [1, 4].
  `processing_times` is what both schedulers read, and CP-SAT folds a sentinel cost back
  into its exclusions, so one edit binds both arms.

A related trap: **a warm start that violates a constraint is worse than no warm start.**
HEFT computes its hint before these exclusions exist, so it hints combinations that are
then illegal; CP-SAT starts from an infeasible point and spends the budget repairing it.
On one rung this returned *no schedule at all* at 400 s. The hint is now projected onto
the constraints before it is handed over.

### 2.5 Convergence

The accepted candidate becomes the base for the next round, so transformations compose.
The loop stops when a round accepts nothing.

---

## 3. The outer loop, in detail

### 3.1 Execute the schedule on the board

`ModelBlaster/scripts/run_xpurt_k1.sh` ingests the schedule and, per network, extracts or
accepts staged IR, generates the skeleton and kernels, emits a `dispatch_table.{c,h}` and
an `xpurt_main.c` walker, cross-builds a RISC-V binary, deploys it, runs it pinned, and
pulls back stdout and a trace CSV. Each network's output is checked against a reference:
int8 networks come back bit-exact (`max_abs_err=0`), the float `fused_full` within 1.8e-4.

Practical notes that cost time to rediscover:

* `--backends` takes one entry per **core kind**, not per model — two on the K1 regardless
  of how many networks are in the schedule.
* Graph extraction needs torch, which the solver venv lacks, so board runs reuse built IR
  via `--staged-ir <net>:<dir>` rather than re-extracting. The build tree is
  `ModelBlaster/build/k1_xpurt/`, not `build/`.
* A schedule using the `ime` lever needs an `ime_x60` backend; an RVV-only build dies with
  `FATAL entry 0 of ffn_block asks for impl 'ime'`.

The trace carries, per dispatch: network, instance, dispatch id, op, core kind, hart,
predicted start and duration, and **actual start and end in rdtime cycles** (24 MHz).

### 3.2 Turn the trace into costs

`scripts/emit_board_calibration.py` computes `actual / predicted` per dispatch and emits
three tiers, matching the lookup order in `profile_loader._board_calibration_mult`:

1. `per_dispatch_multiplier["net/dispatch_id"]` — exact, measured
2. `per_op_multiplier["op_kind"]` — pooled fallback
3. `aggregate_multiplier` — last resort

Two design choices to state if the paper describes this:

* **Execution only; queueing excluded.** A dispatch that waited is not a dispatch that ran
  slowly, and folding queueing into a per-op cost would charge the scheduler twice for its
  own placement decisions.
* **Dispatch-id alignment is checked** against the schedule. The runner records zero-cost
  ops (`chunk`, `split`, `slice`) as `-1` and renumbers the rest, so on a network
  containing them the two numberings diverge; YOLO traces ids the schedule does not have.
  Where they diverge, per-dispatch keys are dropped and that network falls back to op-kind
  keys, which are still measured from its own rows.

This tool also closed a provenance gap: the deployed calibration table was an **orphaned
output** — the script that produced it was never committed, so it could be used but not
regenerated for a new workload, and its own metadata admitted YOLO was extrapolated.

### 3.3 Re-cost: the reveal

Re-scoring the **same** schedule on measured costs is the step that carries information,
because it separates *the model was optimistic* from *the schedule was wrong*. On one
five-network rung the offline model predicted 7 misses and the board showed **16** —
including five in a network the offline solve believed was entirely fine.

### 3.4 Feed back, at two depths

* **Re-solve**: the scheduler re-places dispatches against measured costs.
* **Re-search**: the lever search itself runs against measured costs, so the *choice* of
  transformation is made on measured rather than predicted data (`--search-calibration`).

### 3.5 Attribute the residual

`scripts/attribute_board_misses.py` splits measured misses into what a scheduler could fix
and what it cannot, by summing each instance's own dispatch cycles:

* **execution-bound** — the instance's own execution already exceeds its window, so no
  placement helps. Reported separately when it is only the *first* instance, which is
  **cold start**: a network's first instance runs up to 2.74× its warm median, and the
  warm steady-state profile models none of it.
* **queueing** — everything else, which *is* the scheduler's to fix.

This is arguably the outer loop's most useful output: it converts "the loop failed" into
"the target was not achievable, here is why".

---

## 4. Platform and workloads

**SpaceMiT K1**: 8 RVV X60 harts in two clusters (4 "P", 4 "E"), plus the IME matrix
engine on cluster 0 (cluster 1 traps on it). Timing from `rdtime` at 24 MHz.

**Networks**, with measured whole-net board times:

| network | 1 core | best | speedup | dispatches/instance |
|---|---:|---:|---:|---:|
| mlp_control | 0.08 ms | 0.08 (1c) | 1.00× | 7 |
| attn_block | 0.13 | 0.13 (1c) | 1.00× | 14 |
| fused_full | 3.62 | 3.62 (1c) | 1.00× | 15 |
| dronet | 8.33 | 5.25 (4c) | 1.59× | 21 |
| ffn_block | 26.61 | 7.72 (8c) | 3.45× | 5 |
| yolov8_nano_64x96 | 47.73 | 23.95 (8c) | 1.99× | 98 |

**Rungs** are sized so that (a) the baseline misses at least one deadline and (b) a
schedule meeting every deadline exists using implementations already measured on the
board. `scripts/make_scaling_workloads.py --check` reports whether a rung is in that band.

---

## 5. Results

### 5.1 Inner loop

Instance-level deadline misses, greedy, reproducible bit-for-bit across runs:

| rung | networks | baseline → after | levers found |
|---|---:|---|---|
| w2 | 2 | 5 → **0** | `shard` |
| w3 | 3 | 10 → **5** | `shard` |
| w4 | 4 | 10 → **5** (−21.7% makespan) | `shard:ffn_block` |
| w5 | 5 | 11 → **7** (−14.2%) | `shard:ffn_block`, `shard:dronet`, `ime` |
| b4 board-sized | 4 | 10 → **3** | `shard` |
| s5 | 5 | 1 → **0** (−30.4%) | `shard:yolo`, `shard:ffn_block` |

w4 and w5 previously reported 10 → 10 and 11 → 11. Nothing about the scheduler changed;
the four defects in §2.3 and §2.4 were removed.

### 5.2 Outer loop — the four-beat arcs

`baseline → AOT-optimised → board re-cost → board re-solve`, instance misses:

| rung | arc | reading |
|---|---|---|
| c2 (2 nets) | 5 → 0 → 0 → 0 | AOT model accurate; nothing to correct |
| b4 (4 nets) | 10 → 3 → 4 → 4 | board slightly harsher; greedy re-solve made it worse (4 → 8) and was correctly refused |
| b5y (5 nets) | 3 → **0 → 0** → 0 | zero misses that HOLD on hardware; zero execution-bound instances |
| w5 (5 nets) | 11 → 7 → **16** → 14 | the reveal: 9 misses the model did not predict, 5 in a network it thought was clean |
| sensor (5 nets) | 2 → 0 → **1** → 0 | reveal **and** fix — the arc the figure shows |
| s5 (5 nets) | 1 → 0 → **1** → 1 | reveal, and the residual is provably unfixable |

Nine board runs were taken, each calibrated from its own trace (aggregate multipliers
1.13–1.73, 109–484 dispatch samples).

### 5.3 Solver behaviour

From all 204 CP-SAT certificates in the repo (83 OPTIMAL / 121 FEASIBLE):

* **A reachable target is a tractable one.** Every run whose phase-1 objective is 0 is
  OPTIMAL — 55 of 55. All 121 FEASIBLE runs have objective > 0 and an open gap.
* **Size bites at the top.** n=492 dispatches is *never* OPTIMAL (0/33); above 300, only 2
  of 73 ever proved phase 1. At 215 dispatches phase 1 proves optimal in 18–31 s.
* Between those, node count is a weak predictor — n=214 is mostly FEASIBLE while the
  larger n=217 and n=254 are mostly OPTIMAL.

---

## 6. What can and cannot be claimed

**Supported:**

* The inner loop clears or reduces deadline misses on every rung tested, reproducibly, and
  composes multiple transformations (up to three levers over three rounds).
* The loop's decisions are gated by a buildability contract, so what it proposes can
  actually be generated — this is checkable, not asserted.
* The outer loop reveals misses no offline solve can find (7 predicted → 16 measured), and
  attributes residuals into schedulable and unschedulable.
* A five-network workload reaches zero misses that hold under measurement (`b5y`).
* The whole chain runs unattended: propose → solve → verify buildable → execute on
  hardware → calibrate → re-cost → re-solve.

**Not supported, and worth not implying:**

* *That the outer loop's re-solve reliably recovers what it reveals.* Across w4, w5 and
  b5z, with both solvers, re-solving on measured costs did **not** pay off; on b5z the
  minimum over the entire shard lever space was 3 misses while the blind schedule measured
  1. The reveal-and-attribute half is what consistently earns its place.
* *That CP-SAT beats greedy here.* It ties on small rungs and loses on large ones. It is
  the better *re-optimiser* (recovering 9 misses on one re-solve where greedy recovered 2)
  but the worse *solver* at these sizes.
* *That the loop scales smoothly with network count.* The b2y–b5y rungs all report 3 → 0
  with an identical makespan because one network dominates; they are four views of one
  problem, not a scaling curve.

---

## 7. Corrections made during the study

Worth knowing so they are not reintroduced.

1. **The automation was never committed** — 596 insertions sat dirty while the paper
   claimed an automatic process.
2. **CP-SAT ran at the wrong budget.** `--cpsat-time-limit` reached only one of two call
   sites, so `--cpsat-time-limit 300` really solved at `--time-limit 90`.
3. **The lexicographic loop abandoned its lower objectives.** When a phase failed to prove
   optimality it broke out, so lateness and makespan were never optimised — which is why
   CP-SAT returned a *worse makespan than greedy*. Unproven phases are now bounded rather
   than abandoned.
4. **The `fused_full` attribution was wrong.** Its misses were blamed on it being a
   single-width network; measurement showed it fits its window warm and the real
   execution-bound network was `ffn_block`, whose warm board execution (10.34 ms) exceeds
   its 10 ms window at its fastest measured width.
5. **The ladder's windows were sized from profiles the board contradicts** — ffn 1.34×,
   YOLO 1.79×. So "0 infeasible misses" overstated what was achievable, and rungs were
   asking for targets that do not exist.
6. **Three analytical "is this rung at stake?" gates were wrong**, each with correct
   arithmetic over a wrong model of the baseline: comparing single-core time to the window
   ignores that the window may be wide for cold-start reasons; comparing it to the period
   is false because successive *instances* run on different harts; and comparing a
   network's *serial sum* to its window ignores that the scheduler parallelises its
   *dispatches* across cores. The reliable test is to solve the baseline and look.
7. **A reachability gate calibrated from a different workload mispredicted by 1.66×** —
   the same extrapolation error the outer loop exists to catch, made while building the
   gate for it.
8. **The paper's Figure 5 caption claimed sharding that did not happen.** That run's loop
   report records `levers_available: ['ime']` and all 217 dispatches are width 1 in all
   four stages; the arc turns on four dispatches moving to IME, six after the re-solve.

---

## 8. Where things are

* Code and experiments: worktree `/scratch/agustin/xpurt-dev-sync`, branch
  `feat/codesign-loop-hardening`, 18 commits this session, **not pushed**.
* Figure: `results/codesign_feedback/loop_overview.{png,pdf}` plus a `_metrics.json`
  sidecar carrying every number on it. Its middle band still shows pre-fix data and needs
  regenerating.
* Paper: fork at `/scratch2/agustin/XPU-RT-paper-fork`, Overleaf branch merged and the
  figure installed as a double-column `figure*` in `sections/design.tex`. Six commits,
  **not pushed** — Overleaf will not see the merge until they are.
* Companion doc with the w4/w5 specifics: `docs/w4_w5_inner_outer.md`.
