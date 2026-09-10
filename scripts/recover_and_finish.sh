#!/usr/bin/env bash
# Robust recovery + finish, one job (the box keeps memory-killing separate tasks). Idempotent: if this
# is itself killed, just relaunch it — the resume skips completed cells and continues.
#   1) wait for any orphaned current flight to finish (avoid GPU contention on relaunch)
#   2) continue course B via the idempotent resume driver
#   3) run the scheduled energy experiment (GPU now free)
set -u
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # this repo, wherever it is checked out
export ISAAC_PY="${ISAAC_PY:-/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python}"
log(){ echo "=== $(date +%H:%M:%S) recover: $* ==="; }

log "waiting for any orphaned flight to finish before relaunching (avoid GPU contention)..."
while pgrep -f sweep_rate_demo >/dev/null 2>&1; do sleep 20; done

log "continuing course B (idempotent resume)"
bash "$WT/scripts/run_strengthen_resume.sh"

log "running the scheduled energy experiment"
bash "$WT/scripts/run_energy_experiment.sh"

log "RECOVER_AND_FINISH DONE"
