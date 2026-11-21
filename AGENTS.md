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
| `modal/modal_run.py` | Modal app: `download` → parallel `train` → `evaluate`, data in volume `nocap` |
| `colab.ipynb`, `modal/modal_notebook.ipynb` | interactive notebooks (same cells), runs sequentially on one GPU |
| `results/<platform>-vN/` | committed stdout logs (`<name>.txt`), `summary.md`, `compare.png` |
| `EXPERIMENTS.md` | log of every attempt — append a section per run |

## Before launching anything

1. Smoke test locally (CPU/MPS, ~1 min):
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

## Modal (paid per second, H100/A100; $30/month free credit)

```bash
pip install modal && modal setup
modal run --detach modal/modal_run.py --preset full --gpu H100 --pool 1.25 --score_len 64
modal app logs nocap-smartfilter                        # follow progress
modal volume ls nocap runs/full
modal volume get nocap runs/full results/modal-full
python experiments.py eval results/modal-full
```

- `--runs baseline,smartfilter` (default) starts one container per run, in parallel, same GPU type.
- A full baseline + filter pair on H100 costs roughly $20; ask the user before launching paid runs.

## After a run

1. Download logs into `results/<platform>-vN/` and run `experiments.py eval`.
2. Append a section to `EXPERIMENTS.md`: commit, what changed vs. the previous attempt, the eval table,
   lessons. Keep wrong conclusions visible with a correction rather than deleting them.
3. Update the status tables in `README.md` and `IDEA.md`.
