#!/usr/bin/env bash
# SCHEDULED energy experiment: waits until the current strengthening campaign frees the GPU, then
# flies ours-vs-baseline with wrench-logging (--dump_figure_data now carries the commanded wrench),
# and computes modeled propulsive energy. Safe to launch now; it holds until the GPU is idle.
set -u
CAN=/scratch/agustin/projects/DIMA/XPU-RT
WT=/scratch/agustin/xpurt-dev-sync; RES=$WT/results/codesign_feedback
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
W=$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt
ER=$RES/energy_runs; mkdir -p "$ER/tmp"; export TMPDIR="$ER/tmp"
LOG=$ER/ENERGY.log; : > "$LOG"
cd "$CAN" || exit 1
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

say "waiting for the strengthening campaign to finish so the GPU is free..."
while pgrep -f "run_strengthen_resume|sweep_rate_demo" >/dev/null 2>&1; do sleep 60; done
say "GPU free — starting energy runs (ours vs baseline, 3 seeds each, wrench-logged)"

run_cond(){ local cond=$1 lat=$2 mom=$3
  for s in 1000 1001 1002; do
    say "energy: $cond seed $s (lat=$lat mom=$mom)"
    timeout 900 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
      --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
      --sched_latency_ms "$lat" --percep_hold_ms 0 --moment_scale "$mom" \
      --cruise_speed 1.4 --walk_speed 0.0 --episodes 1 --seed "$s" --max_steps 1800 \
      --dump_figure_data "$ER/${cond}_s${s}" >> "$ER/${cond}.log" 2>&1
  done
}
run_cond xpu100 4.89 0.005    # XPU-RT, 100 Hz
run_cond ros50  12.40 0.01    # ROS,     50 Hz
run_cond ros25  35.58 0.02    # ROS,     25 Hz (starved, the measured-response regime)

say "computing modeled propulsive energy from the logged wrench..."
$PY $WT/scripts/flight_energy_model.py --glob "$ER/**/figure_data.npz" --out "$RES/flight_energy.csv" 2>&1 | tee -a "$LOG"
say "ENERGY EXPERIMENT DONE -> $RES/flight_energy.csv"
