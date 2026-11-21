# AGENTS.md — running this repo through an AI agent / API

Instructions for coding agents (Claude Code, Codex, …) that launch and evaluate runs without a human
clicking through notebooks. Read [README.md](README.md) and [IDEA.md](IDEA.md) first.

## Goal and rules

- Train GPT-2 124M on FineWeb to val loss ≤ **3.3821** faster than the baseline (4768 steps, 2.5B tokens).
- Allowed: up to 5B training tokens, algorithmic ideas (data selection, losses). **Not allowed:** changing
  the model size, copying modded-nanogpt speedups (Muon etc.).
- Every claim must be against the baseline **from the same code**. Never compare wall-clock time between
  two different GPUs — use `python experiments.py eval <folder>` (steps ratio × same-process step cost).

## Layout

| path | role |
|---|---|
| `train_gpt2.py` | the only training script; `--smart_filter` off = exact upstream baseline |
| `experiments.py` | `PRESETS`, `train_cmd()`, `shards_needed()`, `evaluate()`; CLI: `cmd`, `eval` |
| `kaggle/kaggle_run.py` + `kernel-metadata.json` | Kaggle script kernel; `RUNS` dict picks GPU + filter per run |
| `colab.ipynb` | interactive notebook, runs baseline and filter one after the other on one GPU |
| `results/<platform>-vN/` | committed stdout logs (`<name>.txt`), `summary.md`, `compare.png` |
| `EXPERIMENTS.md` | log of every attempt — append a section per run |

## Before launching anything

1. Smoke test locally (~1 min; on a Mac it runs on the Apple GPU via MPS, otherwise CPU):
   `python train_gpt2.py --batch_size 4 --sequence_length 256 --num_iterations 10 --grad_accumulation_steps 2 --learning_rate 0.0018 --warmup_iters 5 --smart_filter --filter_warmup 3 --filter_score_len 64`
   With `--smart_filter` off, train losses must match the previous commit bit for bit.
2. Cloud runners `git clone` the **`main` branch from GitHub** — commit and push first, otherwise the
   cloud runs old code.
3. Do not commit `data/`, checkpoints (`*.pt`, ~1.5 GB) or credentials.

## Kaggle (free, 2× T4, 30 GPU-h/week, 12 h per session)

```bash
pip install kaggle                       # token: ~/.kaggle/access_token (or KAGGLE_API_TOKEN)
# edit RUNS / PRESET in kaggle/kaggle_run.py, then:
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status <user>/nocap-smartfilter          # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels logs -f <user>/nocap-smartfilter         # live log (delayed; progress lines every 10 min)
kaggle kernels output <user>/nocap-smartfilter -p results/kaggle-vN
python experiments.py eval results/kaggle-vN/quick
```

- Without `--accelerator NvidiaTeslaT4` or without a phone-verified account the kernel silently runs on
  CPU (`nvidia-smi: not found`).
- The CLI cannot cancel a running version; a new push starts a second one (and uses quota). Cancel in the
  web UI (notebook → Versions).
- Only `quick` fits: `full` needs ~45 h per run on a T4.
- Poll status every ~5 min; don't busy-loop the API.

## Kaggle full run (`kaggle/full/`)

- `full_run.py`: Smart Filtering only, full preset, DDP on both T4s, `--save_every 50`, stops itself after
  11.3 h and writes `full/STATUS` = `CONTINUE` / `DONE` / `FAILED`. Set `POOL` before the first session.
- First session: `kaggle kernels push -p kaggle/full --accelerator NvidiaTeslaT4`.
- When a session is COMPLETE: `kaggle/full/next_session.sh` downloads logs to `results/kaggle-full/`,
  re-uploads `latest.pt` as the private dataset `<user>/nocap-ckpt` and pushes the next session.
- Quota: each session uses ~12 h of the 30 h/week; check the user's remaining quota before pushing.
- Evaluate: `python experiments.py eval-full results/kaggle-full` (steps to 3.3821 / 4768 × step cost).

## After a run

1. Download logs into `results/<platform>-vN/` and run `experiments.py eval`.
2. Append a section to `EXPERIMENTS.md`: commit, what changed vs. the previous attempt, the eval table,
   lessons. Keep wrong conclusions visible with a correction rather than deleting them.
3. Update the status tables in `README.md` and `IDEA.md`.
