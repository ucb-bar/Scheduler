# Running the co-design feedback loop

This is the document to hand someone who asks "where is the automated script?". Every
command below is committed and runs from a fresh clone; the ones that need the physical
K1 board or the cross toolchain say so.

Paths are repo-relative. `$PY` is the repo venv (see [Environment](#environment)).

---

## What the loop is, in one table

| stage | what decides | script | needs |
|---|---|---|---|
| solve | the scheduler | `scripts/run_xpurt_schedule.py` | — |
| advise | the profile + the solved schedule | `scripts/emit_compile_advice.py` | — |
| bridge | one verb's hint from the advice | `scripts/advice_to_{fusion,unfuse,split,shard,kernel_choice}_hint.py` | — |
| apply | ModelBlaster rewrites the IR | `ModelBlaster/pipeline/apply_*_hint.py` | — |
| graph gate | did the dispatch graph change | `scripts/diff_dispatch_graph.py` | — |
| correctness gate | is the rewrite bit-exact | `ModelBlaster/scripts/verify_ir_rewrite_host.py` | — |
| rebuild + measure | the board | `ModelBlaster/scripts/run_model_k1.sh` | **toolchain + K1** |
| re-solve | the scheduler, on measured costs | `scripts/run_xpurt_schedule.py` | — |
| verdict | `candidate_objective.accept()` — nine terms | `scripts/compare_candidates.py` | — |

Two drivers chain those for you, and one composes the figure.

---

## 1. The scheduling levers, one command, no board

```bash
$PY scripts/run_codesign_loop.py \
    --workload data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
    --solver cpsat --time-limit 45 --objective lateness \
    --board-calibration results/codesign_feedback/k1_board_calibration.json \
    --board-solver cpsat --board-time-limit 200 \
    --out-dir results/codesign_loop
```

Starts from the workload with every lever stripped, proposes each not-yet-applied lever
(`shard`, `ime`, `unfuse`) as a candidate, solves each with the real profiled scheduler,
and accepts on **`candidate_objective.accept()`** — nine lexicographic terms, hard
deadline misses first and makespan seventh, each with a noise tolerance. Then the
board-feedback arm re-costs the accepted schedule under measured K1 multipliers and
re-solves against them until the misses clear.

What it prints, on that workload:

```
round 1 · try ime:    accepted -- hard deadline misses: ime=0 beats baseline=2
round 1 · try shard:  accepted -- hard deadline misses: shard=0 beats baseline=2
round 1 · try unfuse: lever changed nothing on this spec (inapplicable here)
round 1: ACCEPT +shard   total-lateness 3.226 -> 0.000
round 2 · try ime:    rejected -- max deadline lateness: shard=0 beats ime=11.29
round 2: no lever is accepted by the nine-term objective rule — CONVERGED
board arm: instance-miss arc  baseline->AOT->board-recost->board-resolve = 2 -> 0 -> 2 -> 0
```

Read that last line as the whole point: the offline loop clears the misses, the board
says two came back, and re-solving on the measured costs clears them again.

`loop_report.json` records, per round, the accepted lever, the **deciding term**, the
nine-term values, and every rejected candidate with its reason. `--accept-rule legacy`
reproduces pre-hardening runs, which judged on `misses_not_worse AND delta > 0.05 ms`
and could accept a lever with 93 deadline misses on a makespan win.

## 2. The graph-rewrite arm (advice → bridge → applier → gates)

```bash
$PY scripts/run_modelblaster_arm.py \
    --schedule schedules/scheduled_<stem>_greedy_profiled.json \
    --ir ffn_block=<ModelBlaster build>/k1_xpurt/ffn_block/int8/graph.json \
    --out-dir results/modelblaster_arm/<name>
```

Drives all five verbs and records, per verb, how far it got and **why it stopped** —
"the advisor did not fire", "the applier refused" and "the rewrite is not bit-exact" are
three different findings. Nothing is marked `eligible` without the bit-exact host verify
(`--no-require-verify` overrides, and the report then says so on every row).

`shard` is exempt from the numeric gate on purpose: `apply_shard_hint` is annotation-only
— same dispatch count, same ids, one extra `shard_factor` field — so numerics cannot
change and there is nothing for a numeric check to check. Its remaining gate is a board
reprofile at that width.

The IRs are ModelBlaster build outputs (`pipeline/extract_graph.py`), not committed.

## 3. A board round: rebuild, reprofile, re-solve, adjudicate — **needs the K1**

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"     # GCC 14.3 for riscv64
$PY scripts/run_board_round.py \
    --workload data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
    --net ffn_block \
    --baseline-ir  <build>/k1_xpurt/ffn_block/int8/graph.json \
    --candidate-ir results/modelblaster_arm/<name>/ffn_block.split.graph.json \
    --seed-build-dir <build>/k1_xpurt/ffn_block/int8 \
    --baseline-tag bA --candidate-tag splitM2 \
    --out-dir results/board_loop/ffn_split
```

Both arms are profiled in one session with the same curated kernels and filed under their
own model identity (`<net>_<tag>`). That is not cosmetic: `gen_mb/profile` is a symlink
the profiler writes through, so a shared basename means the second run overwrites the
costs the first was solved from and the comparison becomes a schedule against itself.

The round refuses to use costs from a board run that did not report `max_abs_err=0`, and
warns when a curated kernel fell back to the scalar reference — both produce timings that
describe something other than the kernel under test.

Result on `ffn_block`, whose `linear_s8` overruns its slot (the scheduler's own advice):

| | blocking dispatch | total |
|---|---|---|
| baseline | 360,803 cy | 685,262 cy |
| split along M, ×2 | 185,611 cy | 659,129 cy |

and the re-solve: makespan 53.55 → 45.37 ms, worst lateness 22.65 → 14.54 ms, 2 → 1
instance-misses. **Verdict: ACCEPT** on term 2 of 9.

## 4. The figure, and its caption

```bash
XPURT_CPSAT_WORKERS=6 $PY scripts/gen_schedule_evolution.py --from-loop
$PY scripts/emit_figure_numbers.py \
    --metrics results/codesign_feedback/schedule_evolution_auto_metrics.json \
    --prefix schedEvo --out results/codesign_feedback/schedule_evolution_numbers.tex
```

`--from-loop` runs the loop and builds the four panels from its **actual trajectory**,
so panel 2's title names the levers the loop chose. `emit_figure_numbers.py` turns the
figure's own metrics into `\newcommand`s for the caption to quote, because a hand-typed
caption and a regenerated figure drifted apart once already: the paper said panel 3 had
four misses at 34 ms while the auto variant of the same figure read 2 misses at 36 ms.
Both were true about *some* figure.

## 5. How general is it

```bash
$PY scripts/loop_over_workloads.py --solver greedy \
    --board-calibration results/codesign_feedback/k1_board_calibration.json \
    --out-dir results/loop_sweep
```

Runs the same driver over every multi-net K1 spec with no per-workload tuning and reports
the fraction it strictly improves, listing the workloads where no lever helped. A run
that converges with nothing applied is not an improvement and is counted as one of the
failures.

---

## Environment

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python --no-deps -e . -e ModelBlaster
uv pip install --python .venv/bin/python numpy scipy matplotlib pandas ortools \
    pyyaml pillow requests pytest cvxpy 'highspy==1.14.0'
git submodule update --init ModelBlaster
```

`--no-deps` for the two local packages on purpose: ModelBlaster declares `torch` and
`ultralytics`, which only the model-extraction path needs. `highspy` is pinned because
1.15's bundled HiGHS has a C++ ABI clash with ortools' `libortools` — import `cvxpy`
then `ortools` and it dies on an undefined symbol.

Then `$PY -m pytest -q` (771 passed) and, in `ModelBlaster`, `$PY -m pytest tests
pipeline/tests -q` (183 passed). CI runs both plus `examples/run_all.py`.

MOSEK is optional and not needed for any result here.

## Things that will bite you

- **`XPURT_CPSAT_WORKERS`.** `scheduler_cpsat` pins one worker and seed 42 so a cold
  rerun matches bit-exactly, and says in as many words that more workers under a time
  limit are not deterministic. The published solver numbers use `=0`/`=6`; the test
  suite pins `=1` via `conftest.py`. Record which you used.
- **A short profile CSV.** A dispatch absent from a profile that *exists* is costed
  0.0. It is reported now, listed in the schedule as `metadata.zero_costed_dispatches`,
  and fatal under `XPURT_STRICT_ZERO_COST=1`. On the sensor workload it flags 8 — yolo's
  `chunk2_c1` ops, which the codegen legitimately drops. A short CSV looks identical
  from inside the loader, which is why the list is in the artifact.
- **`--board-calibration` with a wrong path** exits 2 rather than silently producing an
  uncalibrated schedule under a calibrated name.
- **`pdb_hash` vs `solve_hash`.** `pdb_hash` fingerprints only the profile CSVs, so an
  additive solve and a calibrated re-solve of one spec share it — and `compare_candidates`
  refused that comparison, which is the one the board-feedback story rests on.
  `solve_hash` folds in the calibration content, solver and env.
- **Derived IME profiles.** `scripts/make_ime_profile.py` writes measured-RVV-divided-by-
  measured-speedup into the tree the loader reads as board measurement. Those rows carry
  `source=ime_derived(...)` and a `PROVENANCE.json` sidecar. They are predictions.
- **The board is shared.** Check before you take it (`docs/k1_board.md`). `ping` to the
  old address fails while `ssh k1` works, so test with ssh.
