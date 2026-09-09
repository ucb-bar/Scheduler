# Two gaps the loop found, with the numbers that locate them

Both came out of running the co-design loop over a 2->5 network ladder and asking why the
residual deadline misses were residual. Neither is a scheduling-search failure: one is a
kernel, one is the machine model. Every number here is measured on the physical K1 and
lives in the committed profiles under `gen/profile_mb/`.

---

## 1. `conv2d_batchnorm2d_silu_s8` runs at 26% parallel efficiency

`yolov8_nano_64x96`, whole net, measured across core widths:

| cores | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms | 47.73 | 29.56 | 24.35 | 23.95 |

That is **1.99x on 8 cores**, and it has saturated -- 4 to 8 cores buys 1.6%. Per op kind
at 8 cores:

| op | n | 1 core | 8 cores | speedup | share of the 8-core time |
|---|---|---|---|---|---|
| `conv2d_batchnorm2d_silu_s8` | 57 | 46.41 | 22.57 | **2.06x** | **94.2%** |
| `conv2d_s8` | 6 | 0.51 | 0.54 | 0.93x | 2.3% |
| `maxpool2d_s8` | 3 | 0.36 | 0.37 | 0.97x | 1.5% |

So one op kind is 94% of the runtime and it parallelises at 26% efficiency. For contrast,
`ffn_block`'s `linear_s8` gets **8.94x** on the same 8 cores, so the ceiling is the kernel,
not the hardware.

**What it costs.** This is what makes the deployed workloads infeasible rather than merely
hard. `yolov8_nano_64x96` needs **23.95 ms against a 22 ms window** -- a ratio of 1.089 --
so no scheduler, no lever, no solver and no amount of board feedback can meet that
deadline. In the full ablation over the existing corpus, **44 of the 50 residual misses**
were of exactly this kind, across five of the twelve at-stake workloads
(`networks_k1_flight_deployed` and variants, `networks_k1_deployed_rich_vithead`,
`_flight_deployed_matched_board`, `_flight_deployed_2frame`), plus 12 more on
`yolov8_nano` in `networks_k1_mb_3model_12hz` (176.37 ms, no shard profile at all).

Reporting those as loop failures is a category error. The honest statement is that the
loop clears what is clearable, and this kernel is the reason the rest is not.

**Where to look.** The op is 96% of yolo's dispatch graph and the fused convs are the ones
`docs/the_loop.md` records as needing per-shard repacked weights and parallel wrappers --
the B4 rung. `scripts/make_scaling_workloads.py` and
`scripts/ablate_feedback_loops.py::feasibility` both read these profiles, so when the
kernel improves, the infeasible set shrinks automatically and the ladder can be retightened.

---

## 2. No 8-wide machine combination exists, though the 8-hart profile does

A solved schedule from the loop (`results/codesign_feedback/evo_w2/a2_aot.json`) offers 21
combinations, and their widths are:

```
{1: 12, 2: 6, 4: 3}          # widest: [CPU_P#0..3] and [CPU_E#0..3]
cross-cluster (P and E in one combination)? False
```

`build_machine_combinations` (`xpu-rt/workload_factory.py`) builds blocks **per machine
kind** -- its docstring says so: *"Each combination only contains cores from the same
processor type."* On a genuinely heterogeneous target that is right. On the K1 it is a
modelling artifact: the spec declares `machines: {cpu_p: 4, cpu_e: 4}` while
`profile_hw` is **`rvv_x60` for both**, i.e. one homogeneous 8-core cluster -- and
`gen/profile_mb/.../topo_0_1_2_3_4_5_6_7/results.csv` is a real 8-hart board measurement
the scheduler can never use, because no combination has 8 cores.

**What it costs.** From w2's own AOT schedule, one `ffn_block` instance:

| dispatch | width chosen | ms |
|---|---|---|
| `layernorm_s8` | 1 | 0.732 |
| `linear_s8` M128xK256xN1024 | 4 | 3.644 |
| `gelu_s8` | 1 | 0.478 |
| `linear_s8` M128xK1024xN256 | 4 | 2.826 |
| `add_s8` | 1 | 0.170 |
| **instance total** | | **7.85** |

The per-dispatch choice is already good -- it declines to shard `layernorm` and `gelu`,
which get *slower* when widened (0.36x and 0.28x, i.e. ~3x worse). What it cannot do is
put the linears on 8 cores. Choosing the fastest measured width per dispatch, 8 included,
gives **4.66 ms**: a **1.68x** improvement on this net, from measurements already taken.
Across the ladder's nets:

| net | best uniform width | per-dispatch oracle incl. 8-wide | gain |
|---|---|---|---|
| `ffn_block` | 7.72 | 4.66 | **1.66x** |
| `dronet` | 5.25 | 4.21 | 1.25x |
| `yolov8_nano_64x96` | 23.95 | 23.03 | 1.04x |

yolo barely moves, for the reason in gap 1: there is nothing to re-balance when 94% of the
work is one op kind.

**Why this was NOT changed here.** It is more invasive than it looks and it would perturb
every existing schedule in the repo:

1. With `enable_impls` on -- which the loop's `ime` lever sets -- the live path is
   `build_machine_combinations_with_impls` in `xpu-rt/capabilities.py`, not the plain
   builder, so both need the change plus the K1 capability table that restricts IME to
   cluster 0.
2. `combo_hw` in `scripts/run_xpurt_schedule.py` is derived from
   `machine_type_prefix(combo[0])` -- the FIRST core's kind. A cross-kind combination
   would be silently attributed to one kind's backend. Harmless while both kinds are
   `rvv_x60`; wrong on the chipyard gemmini/opu targets.

So it needs a homogeneity guard (same `profile_hw` for every kind in the combination)
threaded through both builders, and it must be opt-in. `aligned_core_blocks` itself needs
nothing: pooled over 8 cores it already yields `[0..7]` and keeps the alignment property
that makes `combinations_overlap` honest.
