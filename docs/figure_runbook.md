# Figure runbook — how each headline PNG is made (inputs + exact command)

Every figure below is reproducible from on-disk artifacts. Repo root = `/scratch2/agustin/XPU-RT`.
Scheduler figures use the XPU-RT venv; the two warehouse mega plots + the HIL scatter need the Isaac env
(`env_isaaclab`) to (re)generate their flight data. `export XPURT_CPSAT_WORKERS=0` for any CP-SAT solve.

Composition scripts marked `scripts/…` live in the repo. Scripts marked `(scratchpad)` are the newer
figure builders that should be copied into `scripts/` for a clean checkout (they only read on-disk artifacts).

---

## 1. `schedule_evolution_mega` — the co-design "Gantt after Gantt" (the 3rd mega plot)
**Story** (contended sensor-fusion workload; HERO metric = **network-instance deadline misses** — hard-real-time
control, so makespan is secondary context — trajectory **2 → 0 → 4 → 0**, each panel a real solve or a
K1-calibrated re-cost). Panel 4 re-solves on the calibrated costs to **34.16 ms / 0 misses**, essentially the
same makespan as panel 3 but with every instance deadline restored:
1. **og** — RVV singleton dispatches: misses two FFN instance deadlines.
2. **+ shard + IME** — AOT levers (multi-hart widths + matrix-engine routing): meets every deadline ON THE GANTT
   (2 → 0).
3. **runtime feedback** — panel-2 schedule RE-COST with heterogeneous K1 calibration (per-dispatch/per-op
   measurements where available, aggregate fallback otherwise): deadlines the optimistic Gantt promised
   are now MISSED (0 → 4). This is a calibrated model,
   not a direct execution trace. Panel tinted as a board-calibrated round.
4. **re-schedule on board-calibrated costs** — CP-SAT re-solves knowing the true costs: deadlines RECOVERED
   (4 → 0). This is the loop closing: adjust after runtime feedback, before any further optimization.

Idle stretches are compressed into grey break columns (real-ms x-axis kept); misses are ringed red.
- **Generate the sequence** (`XPURT_CPSAT_WORKERS=0`; CP-SAT solves for og/AOT/fix + a board re-cost):
  `XPURT_PY=$(command -v python) python scripts/gen_schedule_evolution.py`
  → writes the 4 panel schedules + `panels.json` under `results/codesign_feedback/sensor_evo/`.
  (The board re-cost is `scripts/recost_schedule_on_board.py` — re-times a fixed schedule using measured
  per-dispatch/per-op calibration where available and an aggregate fallback otherwise; the software twin of
  "how the run differs from the Gantt".)
- **Render**:
  `.venv/bin/python scripts/compose_schedule_evolution.py \
     --spec data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
     --panels-json results/codesign_feedback/sensor_evo/panels.json --layout grid`
  → `results/codesign_feedback/schedule_evolution_mega.{png,pdf}` plus a metrics sidecar. The default grid is
  a 2×2 four-stage layout authored at 7.16×4.35 in for the top of a two-column page; `--layout vertical`
  preserves the legacy single-column stack. All four grid panels share one time scale so the makespan change
  remains visually comparable.
- **Inputs**: the sensor workload spec `_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json` + its profiles
  under `gen_mb/…`; `k1_board_calibration.json` (drives both the re-cost and the board-calibrated re-solve).

## 2. `hil_ablation_phase` — HIL command-rate phase diagram (GPU for the grid)
- **Generate** the per-flight grid (real Isaac flights; 5 speeds × 4 rates × 6 seeds = 120 flights, GPU-hours).
  Needs the conda `env_isaaclab` python — pass it via `ISAAC_PY` (see `docs/REPRODUCE.md` §1):
  `ISAAC_PY=<env_isaaclab>/bin/python bash scripts/hil_ablation_grid.sh`
  → appends rows to `results/codesign_feedback/hil_grid/hil_ablation.csv` (via
  `sims/scripts/sweep_rate_demo.py --sweep-csv`; one row/episode:
  `seed,cruise_speed,sim_dt,decimation,control_dt_ms,sched_latency_ms,hold_steps,eff_cmd_hz,…,outcome`).
  Overridable env: `XPURT_REPO`, `HIL_OUTDIR`, `HIL_WEIGHTS`. The 120-flight CSV is committed at
  `results/codesign_feedback/hil_ablation.csv` so the diagram re-renders without the GPU sweep.
- **Render**:
  `.venv/bin/python scripts/hil_ablation_phase.py --csv results/codesign_feedback/hil_ablation.csv`
  → `results/codesign_feedback/hil_ablation_phase.{png,pdf}`. **Two panels sharing the command-rate axis**:
  LEFT ties each scheduler's worst-case loop response to the rate it sustains (rate = 1000/response — ROS
  12.40 ms→81 Hz, greedy 8.00→125, shard-feedback 4.89→204; the AOT feedback loop's −39 % response = +80 Hz
  headroom), against the shrinking loop budget; RIGHT is the flight phase diagram (a smooth safe→crash surface
  over speed×rate, crash frontier drawn, raw 6-seed cells) with those rates carried across (ROS 81 Hz on the
  frontier, XPU-RT 125/204 Hz clear). Honest: >100 Hz is hatched "not flight-tested" (no extrapolated surface).
  The scheduler worst-response numbers (12.40/8.00/4.89 ms) are hard-coded constants in the script. Replaces the old
  `hil_ablation_scatter` (kept for reference).

## 3–4. Warehouse figures — combined showdown + the two mega plots (GPU to regen flights)
- **Flight data** — TWO dumps: the XPU-RT successful weave and the ROS crash:
  `<env_isaaclab>/python sims/scripts/record_sensor_demo.py --headless --controller rl \
     --weights sims/models/warehouse/nav_fused_v12_cnn.pt --sched_latency_ms 12.40 --decimation 1 \
     --prop_density 0.35 --obstacle_level 8 --fixed_speed 1.2 --episodes 4 --seed 2000 \
     --dump_figure_data <xpu-dir>` (+ once with `--clean_overview --clean_out <xpu-dir>` for `clean_bg.npz`);
  the ROS crash dump is the same command with the ROS-rate latency (crashes ~y=10, past gate 1) → `<ros-dir>`.
- **`warehouse_showdown_board`** (combined overview) and **`warehouse_schedule_board`** (publication-sized
  schedule panel): both selected paths on one top-down aisle, 2 fixed-baseline + 2 XPU snapshots,
  IMU/goal/speed/velocity, and a separate matched-horizon schedule-model comparison:
  `<env_isaaclab>/python sims/scripts/compose_warehouse_showdown.py \
     --xpu-dir results/codesign_feedback/crash_demo/complete_figdata \
     --ros-dir results/codesign_feedback/crash_demo/crash_figdata --rot 0 \
     --out results/codesign_feedback/warehouse_showdown_board \
     --schedule-out results/codesign_feedback/warehouse_schedule_board`
  → both `.png` and `.pdf` variants plus metrics sidecars. The defaults are the matched 5/6/12-instance
  board-model schedules `scheduled__flight_deployed_matched_board_cpsat_profiled.json` (XPU-RT) and
  `scheduled_ros_partition_deployed_matched_board.json` (ROS-style fixed partition), comparing interior
  YOLO frame 3 against the adapted 23 ms period/deadline. Trace provenance and selection rules are recorded
  in `results/codesign_feedback/crash_demo/trace_manifest.json`. Use the standalone schedule panel in the
  main paper; treat the dense combined overview as a supplement/full-page asset.
- **`mega_warehouse_xpurt` / `mega_warehouse_ros`** (the per-scheduler mega plots) — Gantt strip via
  `scripts/plot_solver_gantt_annotated.py --sched schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json`
  (the CP-SAT deployed schedule: 40.4 ms 0-miss, balanced), then
  `<env_isaaclab>/python sims/scripts/compose_mega_figure.py --data-dir <xpu-dir> --gantt <gantt.png>
   [--crash-step 779] --out results/codesign_feedback/mega_warehouse_{xpurt,ros}`.
- **Notes**: people are projected at their real height (z≈0.85, not 2.0); moment markers are placed at the
  actual gate crossings; IMU is smoothed. `make_all_codesign_figures.sh §5` drives all three (set
  `WAREHOUSE_FIGDATA`/`WAREHOUSE_ROS_FIGDATA`).

## 5. `solver_win_sensor` — CP-SAT beats greedy (contended sensor workload)
`.venv/bin/python scripts/compose_solver_win.py \
   --greedy schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0_greedy_profiled.json \
   --cpsat  schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0_cpsat_profiled.json \
   --spec   data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0.json --window 32 \
   --out results/codesign_feedback/solver_win_sensor`
(greedy 4 misses / INFEASIBLE vs CP-SAT 0 misses / PROVEN OPTIMAL, 27.85 vs 30.64 ms).

## 6. Supporting Gantts / comparison
- `gantt_annotated_{cpsat,ros}` — `scripts/plot_solver_gantt_annotated.py --sched <schedule> --window-ms 44
  [--crash-note --yolo-deadline 22]` (ROS variant shows serial-YOLO backlog → crash).
- `warehouse_crash_speed` / `warehouse_solver_flight` / `solver_comparison_*` — scratchpad builders reading
  `schedules/*_metrics.json` + the Isaac sweep CSV.

## 7. `loop_ablation` — which feedback loop does the work (and does the exact solver earn its cost)

**Story.** The channel closes at two distances, and one showcase run cannot separate
them. Four cells — **A** neither, **B** inner only, **C** outer only, **D** both — with a
bar per solver arm in each. Read across the cells for which loop does the work; read
within a cell for whether CP-SAT beats greedy. *Inner* = AOT co-design with ModelBlaster
(graph rewrites + per-dispatch implementation choice, decided against isolated profiles;
the board appears only as a **profiler**). *Outer* = HIL (measured multipliers from
real-time runs of the whole schedule, returned and re-solved against; the board is the
**runtime**). HERO metric = network-instance deadline misses.

**Generate** (CPU only; no board needed — the calibration is a committed artifact):

```bash
export XPURT_NO_COMPACT=1                 # the script sets it too, and records it
$PY scripts/ablate_feedback_loops.py \
    --workloads $($PY scripts/loop_over_workloads.py --list-k1) \
    --solvers cpsat,greedy --cpsat-time-limit 900 --repeats 3 \
    --one-per-family --max-rounds 3 \
    --out-dir results/loop_ablation
```

Writes `results/loop_ablation/ablation_summary.json` (+ `ablation.log`). Greedy over all
families is ~11 min; a CP-SAT arm is budget-bound — its cost is set by
`--cpsat-time-limit`, not by the instance — so expect hours and run it `nice`d.

**Render:**

```bash
$PY scripts/plot_loop_ablation.py \
    --summary results/loop_ablation/ablation_summary.json \
    --out-dir results/codesign_feedback --stem loop_ablation
```

Outputs `results/codesign_feedback/loop_ablation.{png,pdf}` plus a metrics sidecar.
Caption numbers come from the sidecar via `scripts/emit_figure_numbers.py --prefix abl…`
(use a per-arm prefix — `ORDINALS` holds 8 entries and cell×arm overflows it). The
renderer reads the summary only and never re-solves.

**Inputs.** `data/toplevel/*.json` (the K1 specs), the profiles under `gen/profile_mb/`,
and `results/codesign_feedback/k1_board_calibration.json`.

**Honesty notes — read these before quoting the figure.**
- **The population is families, not files.** Two specs are byte-identical and several
  variants return bit-identical results; `--one-per-family` is the honest unit. The
  family table is in the summary's `population` block.
- **Only the at-stake stratum is plotted.** A workload whose naive deployment already
  meets every deadline cannot show a loop helping. The stratum (`cell A misses ≥ 1`) is
  pre-registered in code, and the count of excluded workloads is printed on the figure.
- **Every cell is scored on board costs.** A/B are solved on predicted costs and re-cost
  with their assignment fixed; C/D are solved with `--board-calibration` — re-costing
  those would apply the multiplier twice.
- **CP-SAT's result is not reproducible; the experiment is.** Repeats are run only where
  the solver did not prove optimality, and reported as median with the spread.
- **MOSEK is absent on purpose**: it exhausted ~89 GiB on the rich workload and no memory
  guard exists in code.

---
**Regenerate all** (from cached artifacts where possible): `bash scripts/make_all_codesign_figures.sh`.
