# Experiment log

All runs use GPT-2 124M (123.5M parameters), FineWeb, `quick` preset unless stated otherwise:
131k tokens per step, 1200 steps, 157M baseline tokens. Logs are in [`results/`](results/).
Evaluate any folder of logs with `python experiments.py eval <folder>`.

**How runs are compared** (since v3): wall-clock time is only meaningful on the same GPU, and two T4s in
one Kaggle session can differ by 22%. So the comparison is

- **steps ratio** — steps Smart Filtering needs to reach the baseline's final val loss ÷ baseline steps
  (independent of the GPU),
- **step cost** — the filter run's filtered steps vs. its own unfiltered warm-up steps (same GPU, same
  process; step 127, which compiles the scoring graph, is excluded),
- **time ratio** = steps ratio × step cost. Below 1.0 the filter wins.

## Summary

| attempt | filter | baseline val @1200 | filter val @1200 | steps ratio | step cost | time ratio |
|---|---|---|---|---|---|---|
| Kaggle v2, 2026-11-20 | pool 1.25, score 256 tok | 4.1326 | **4.1163** | 0.979 | +10.0% | 1.077 ✗ |
| Kaggle v3, 2026-11-21 | pool 1.25, score 64 tok | 4.1401 | **4.1186** | 0.973 | +4.3% (+2.8% raw) | 1.015 ≈ tie |
| Kaggle v4, 2026-11-21 | pool 1.5, score 64 tok, GPUs swapped | cancelled — stuck in Kaggle's "Starting" | | | | |
| Kaggle v5, 2026-11-22 | pool 1.5, score 64 tok, GPUs swapped | 4.1420 | **4.1005** | 0.946 | +4.0% (+3.3% raw) | **0.984 ✓** |
| Kaggle full (v6) | full preset, filter only, pool 1.5 | prepared; a friend's account runs it | | | | |

The filter is better per step in every attempt (−0.016, −0.022 and −0.041 val loss at step 1200). With
pool 1.25 scoring cost about as much as it saved; with pool 1.5 (v5) the filter needs 5.4% fewer steps for
the same ~+4% step cost, a first small net win (−1.6% time). One seed, one quick run: v6 has to confirm it.

---

## Kaggle v1 — no GPU (2026-11-20)

The kernel started on a CPU machine (`nvidia-smi: not found`) although the metadata asked for T4 x2:
the account was not phone-verified, which Kaggle requires for GPUs and Internet. Nothing trained.

## Kaggle v2 — first attempt (2026-11-20, `results/kaggle-v2/`)

Code: commit `900eabc`. What changed against the previous version (commit `39703b2`):

- **Rotary fix**: the cos/sin cache was keyed on the last sequence length; with scoring (256) and training
  (1024) alternating, every micro-batch invalidated it, which under `torch.compile` means recompiles until
  the recompile limit and then a silent fallback to eager. Now a precomputed table is sliced.
- **No CPU↔GPU syncs** in the filter (statistics stay on the GPU instead of three `.item()` per micro-batch).
- **Pipeline**: a background thread reads/pins the data; scoring of micro-batch *m+1* runs on a side CUDA
  stream while *m* trains.
- **Resume** stores the data position of the last *trained* micro-batch.

Setup: baseline on GPU 0 and Smart Filtering (pool 1.25, score_len 256) on GPU 1, at the same time.

| | baseline (GPU 0) | Smart Filtering (GPU 1) |
|---|---|---|
| val loss @ step 1200 | 4.1326 | **4.1163** |
| wall time | 2.63 h | 3.16 h |
| s/step (steps 200–1199) | 7.83 | 9.48 |
| tokens read | 157 M | 192 M |

Reading at the time: the filter was ~20% slower per step. Half of that was already there in steps 0–127,
when nothing is filtered — so either GPU 1 was slower or the pipeline had overhead. v3 was built to tell.

## Kaggle v3 — second attempt (2026-11-21, `results/kaggle-v3/`)

Changes against v2:

1. **Diagnostics first** (phase 1, 60 steps each, steps 10–59 timed):
   - 1a: baseline on both GPUs → **GPU 0 8.49 s/step, GPU 1 6.60 s/step**: GPU 1 was 22% *faster* in
     this session (in v2 it had looked slower) — Kaggle hands out different machines and T4s vary.
   - 1b: the filter pipeline without filtering on GPU 0 → 8.56 s/step = **+0.9% overhead**: the pipeline
     itself is not the problem. GPU 1 ran the baseline again: 6.84 s/step (same GPU varies by ~3.5%).
2. **Cheaper scoring**: `score_len` 256 → **64** tokens (≈4× cheaper scoring forward).

Phase 2: baseline on GPU 0, Smart Filtering (pool 1.25, score_len 64) on GPU 1.

| | baseline (GPU 0) | Smart Filtering (GPU 1) |
|---|---|---|
| val loss @ step 400 | 5.1424 | **5.0911** |
| val loss @ step 800 | 4.4792 | **4.4778** |
| val loss @ step 1200 | 4.1401 | **4.1186** |
| steps to val 4.1401 | 1200 | **1168** (−2.7%) |
| unfiltered steps 10–126 | — | 7.11 s/step |
| filtered steps 200–1199 | — | 7.31 s/step (+2.8%; +4.3% after correcting for the baseline's drift) |
| **time ratio** | 1.000 | **1.00–1.015** |
| wall time (different GPUs, not comparable) | 2.87 h | 2.44 h |

Lessons:
- Never compare wall time across two GPUs; the runner's own "+6.6%" line (GPU-ratio corrected) is
  unreliable for the same reason. Use `experiments.py eval`.
- Scoring 64 tokens costs ~3–4% per step instead of ~10%, and the selection is still as good per step.
- An early hand calculation that called v3 a 2–3% win included step 127 (compiling the scoring graph,
  35 s instead of 7 s) in the unfiltered average. Corrected: a tie.

## Kaggle v4 / v5 — swapped GPUs, stronger filter (2026-11-21)

**v4 was cancelled because of a Kaggle-side problem:** it sat in "Starting…" for ~2.5 h without ever getting a
GPU (quota was fine: 3 h 41 min of 30 h used). Re-submitted unchanged as **v5**, which then waited in the queue
for ~2 h and was cancelled too. On 2026-11-22 v5 was pushed again from the same account and started at once
(version 6 of the notebook, ~2 h 50 min, COMPLETE). Logs: [`results/kaggle-v5/`](results/kaggle-v5/).

Setup:

- Smart Filtering on **GPU 0**, baseline on **GPU 1** (swapped against v2/v3), so a per-step gain can't
  come from one particular card.
- `filter_pool` 1.25 → **1.5** (24 candidates, eject 4 easy + 4 hard). Scoring 64 tokens is cheap, so a
  wider selection costs only ~1% more per step.
- No diagnostics phase; the evaluation is GPU-independent now.

Result:

| run | GPU | final val @1200 | steps to 4.1420 | steps ratio | step cost | time ratio |
|---|---|---|---|---|---|---|
| smartfilter_p15 | T4 (GPU 0) | **4.1005** | 1135 | 0.946 | +4.0% (+3.3% raw) | **0.984** |
| baseline | T4 (GPU 1) | 4.1420 | 1200 | 1.000 | – | 1.000 |

Val loss, filter vs. baseline: step 400 5.080 vs. 5.172, 600 4.663 vs. 4.743, 800 4.479 vs. 4.486,
1000 4.273 vs. 4.306, 1200 4.101 vs. 4.142.

- The wider pool roughly doubles the per-step gain of v3 (5.4% vs. 2.7% fewer steps) while the step cost
  stays the same (~+4%), so the time ratio drops from 1.015 to **0.984**.
- The gain is on GPU 0 this time (the filter was on GPU 1 in v2/v3), so it does not come from one card.
- One seed only; the difference to v3 (~3%) is larger than the step-time noise (~3.5%) only in steps, not
  necessarily in time. **Decision: pool 1.5 for the full run (v6).**

## Next

- **Kaggle full run** (`kaggle/full/`): Smart Filtering only, full preset (4768 steps, stop at 3.3821), with
  pool 1.5 (from v5). Both T4s train one run together (~22 h), spread over 2–3 sessions of 12 h that
  hand over a checkpoint. The baseline is not re-run: the official baseline reaches 3.3821 at step 4768 by
  definition, so steps ratio = filter steps / 4768; step cost comes from session 1. Less precise than a
  same-hardware baseline (fp16 on T4 vs. bf16 on a 4090), but fits the free quota.
