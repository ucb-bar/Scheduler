# Automatic co-design loop — `_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload /scratch/agustin/xpurt-dev-sync/data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json --max-rounds 4
```

**Baseline → final: 51.6 → 32.8 ms (-100.0%)** — levers applied: ['shard', 'ime'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +shard | 51.6 | 32.9 | -100.0 | 0 |
| 2 | +ime | 32.9 | 32.8 | -0.0 | 0 |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.
