# Automatic co-design loop — `networks_k1_tri_exact_100ms_feedback`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/networks_k1_tri_exact_100ms_feedback.json --max-rounds 4
```

**Baseline → final: 83.4 → 83.4 ms (-38.9%)** — levers applied: ['shard'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +shard | 83.4 | 83.4 | -38.9 | 0 |

Honest note — fuse: decision-aid unavailable.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.
