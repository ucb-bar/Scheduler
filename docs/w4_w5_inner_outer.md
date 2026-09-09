# The top rungs: what the inner and outer loops each do on w4 and w5

Both rungs used to report that no lever helps — w4 10 → 10 deadline misses, w5 11 → 11 —
and the ladder figure could show it without explaining it. Neither was a scheduling
limit. This is what they actually do, why they looked flat, and how to reproduce it.

## Result

Instance-level deadline misses, on the accept rule's own counter.

| rung | baseline | inner (AOT) | board re-cost | board re-solve | levers found |
|------|---------:|------------:|--------------:|---------------:|--------------|
| w4 · 4 nets | 10 | **5** | 4 | 4 (refused) | `shard:ffn_block` |
| w5 · 5 nets | 11 | **7** | **16** | 14 | `shard:ffn_block`, `shard:dronet`, `ime` |

Bars 1–2 are predicted costs; 3–4 are measured by executing bar 2's schedule on a real
K1. Figure: `results/codesign_feedback/inner_outer_arc.png`.

## Why they looked flat: two defects, neither of them scheduling

**The list scheduler could not be held to the codegen contract.** CP-SAT couples a
dispatch's instances with a constraint, so it can be *asked* for one width per
packed-weight dispatch. Greedy picks each op's combination as it walks the ready set,
with nothing to couple, so the contract could only be checked afterwards and the whole
candidate discarded. On w4 that threw away the best schedule anyone has produced for the
workload — 5 misses against 10, worst lateness 3.09 ms against 17.95 — over **three**
dronet dispatches whose instances took different widths. `codegen_contract.pin_uniform_widths`
now restricts each such dispatch to one width up front, chosen by *measured* cost.

Two traps worth knowing if you touch this:

* Choosing the *widest* usable width is wrong on real data — yolo's OC=2 detect-head
  convs measure slower on four cores than on one. Choosing the *narrowest* silently
  undoes sharding, which is the point of the lever. The width is picked by summed
  measured duration over the dispatch's instances.
* The width must be **priced out**, not flagged. Greedy never reads
  `infeasible_combinations`; flagging alone left dronet 0/8/9 still mixing widths
  `[1, 4]`. `processing_times` is what both schedulers read, and CP-SAT folds a sentinel
  cost back into its exclusions, so one edit binds both arms.

**Sharding was all-or-nothing across networks.** `machine_combination_mode: "shard"`
opens multi-hart combinations for every network at once. On w5 that also widens
`yolov8_nano_64x96` — 191.6 core-ms monopolising all eight harts while the 5 ms-period
networks wait — so worst lateness went 24.67 → 34.87 ms and the whole lever was rejected,
including the part that helps. `restrict_shard_to_networks` plus a per-network
`shard:<net>` lever lets a round widen one network and hold the rest at one core;
successive rounds compose, so the loop *discovers* the set. w5 converges on
`{ffn_block, dronet}` and correctly leaves yolo single-core.

## What the outer loop contributes, stated precisely

It is not "re-place the dispatches". On these rungs it does two things a purely AOT flow
cannot:

1. **It reveals what the cost model missed.** w5's AOT schedule predicts 7 misses; the
   board says **16**, and five of them are in `fused_full`, a network the AOT solve
   believed was entirely clean. That is a discrepancy no amount of offline solving finds.
2. **It attributes the residual.** Of those 16, five *cannot* be scheduled away:
   `net_times` reports a single measured core width for `fused_full` (against
   `{1,2,4,8}` for ffn_block/dronet/yolo), its window is 5.0 ms against a 3.62 ms
   singleton, and the board inflates its dispatches up to 3.12×, which puts it over.
   Adding `shard:fused_full` was tried and changes nothing — identical 16 misses, same
   breakdown — because sharding a single-width net is a no-op by construction. So the
   residual splits **5 unschedulable + 9 contention**, and the first five are an
   actionable ModelBlaster task (generate a multi-core `fused_full`), not a scheduling
   failure.

That also explains the weak fix: the greedy re-solve recovers 2 of the 9 revealed misses
because five of the nine were never recoverable by any scheduler.

## Known limits

* **w4 has no reveal.** Its board re-cost is 5 → 4: the AOT model was already right
  there, and the greedy re-solve makes it worse (4 → 8) and is correctly refused. Bar 4
  is drawn as the beat it kept, marked `[refused: nothing better]` — an empty bar there
  would read as zero misses, the opposite of the truth.
* **A list scheduler is the wrong re-solver.** It does not optimise deadline misses, so
  re-solving it against better costs moves the schedule without aiming at the metric.
  Two independent checks agree: board-aware greedy *search* gives 10 → 8 on w4 and
  11 → 12 on w5, both worse than solving blind and measuring afterwards.
* **The w5 board run excludes the `ime` lever.** The harness is built one backend per
  core kind, and the converged schedule's IME dispatches need an `ime_x60` backend that
  the RVV build lacks (`FATAL entry 0 of ffn_block asks for impl 'ime'`). The measured
  run is the shard-only spec; the IME lever is present in the AOT arc but not in the
  board beats.

## Reproduce

```bash
# inner loop (fast, deterministic)
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w4_ffn_dronet_sensor.json \
    --solver greedy --max-rounds 4 --out-dir results/w4_inner

# execute that schedule on a real K1 and calibrate from its own trace
eval "$(scripts/setup_spacemit_toolchain.sh)"
ModelBlaster/scripts/run_xpurt_k1.sh --schedule <the converged schedule>.json \
    --models mlp_control,fused_full,ffn_block,dronet \
    --staged-ir mlp_control:ModelBlaster/build/k1_xpurt/mlp_control/int8 \
    ... --backends rvv_x60,rvv_x60 --quant int8 --out-root /tmp/w4_board
scripts/emit_board_calibration.py --trace <...>_trace.csv --schedule <...>.json \
    --out results/codesign_feedback/k1_cal_w4_measured.json

# the four beats, scored on those measured costs
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w4_ffn_dronet_sensor.json \
    --solver greedy --max-rounds 4 \
    --board-calibration results/codesign_feedback/k1_cal_w4_measured.json \
    --board-solver greedy --out-dir results/w4_outer

scripts/plot_inner_outer_arc.py \
    --rung "w4 · 4 networks=results/w4_outer/w4_ffn_dronet_sensor/loop_report.json=data/toplevel/scaling/w4_ffn_dronet_sensor.json" \
    --rung "w5 · 5 networks=results/w5_outer/w5_ffn_dronet_yolo/loop_report.json=data/toplevel/scaling/w5_ffn_dronet_yolo.json"
```

`--backends` is one entry per **core kind**, not per model — two kinds on the K1, so two
entries regardless of how many networks are in the schedule. Graph extraction needs
torch, which the solver venv does not have, so board runs reuse built IR via
`--staged-ir` / `MB_IR` rather than re-extracting.
