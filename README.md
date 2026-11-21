# Smart Filtering for GPT-2

A submission to the [BottleCapAI NoCap-Test benchmark](https://github.com/BottleCapAI/NoCap-Test):
train GPT-2 124M (123.5M parameters) on FineWeb to **val loss ≤ 3.3821** faster than the baseline
(5.401 h on one RTX 4090, 4768 steps, 2.5B tokens; up to 5B tokens allowed).

**Idea:** spend the unused token budget on *reading* more data and train only on the sequences in the
Goldilocks zone. Each micro-batch draws 25% extra candidates, the live model scores them on a 64-token
prefix, and the easiest and noisiest ones are ejected before the forward/backward pass. Reading, scoring
and training run at the same time (background reader thread + side CUDA stream). Details: [IDEA.md](IDEA.md).

```
pool of 20 sequences ──► score (live model, no_grad, first 64 tokens) ──► sort by loss
                                                                            │
          eject 2 easiest ◄─────────────────────────────────────────────────┤
          eject 2 hardest ◄─────────────────────────────────────────────────┤
                                                                            ▼
                                                  train on the middle 16 (as baseline)
```

## Status

| attempt (Kaggle T4, `quick`) | filter val @1200 vs. baseline | steps ratio | step cost | time ratio |
|---|---|---|---|---|
| v2: score 256 tokens | 4.1163 vs. 4.1326 | 0.979 | +10% | 1.077 — loses |
| v3: score 64 tokens | 4.1186 vs. 4.1401 | 0.973 | +3–4% | 1.00–1.015 — tie |

Better per step, not yet faster in time. Full log: [EXPERIMENTS.md](EXPERIMENTS.md).

## Running

Three ready-made setups, all built on [`experiments.py`](experiments.py) (presets, commands, evaluation):

| platform | file | GPUs | good for |
|---|---|---|---|
| Google Colab | [`colab.ipynb`](https://colab.research.google.com/github/vbtronic/BottleCapAI_NoCap-Test/blob/main/colab.ipynb) | free T4; Pro: L4/A100 | `quick` on a T4, `full` with Pro |
| Kaggle | [`kaggle/`](kaggle/) | 2× T4, 30 h/week free | `quick`, two runs at once |
| Modal | [`modal/modal_run.py`](modal/modal_run.py), [`modal/modal_notebook.ipynb`](modal/modal_notebook.ipynb) | H100/A100, pay per second | the `full` benchmark |

```bash
# Kaggle (API token in ~/.kaggle/access_token, phone-verified account)
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels output <user>/nocap-smartfilter -p results/kaggle-vN

# Modal (pip install modal && modal setup)
modal run --detach modal/modal_run.py --preset full --gpu H100
modal volume get nocap runs/full results/modal-full

# compare runs
python experiments.py eval results/kaggle-v3/2
```

How to run it through an AI agent / API: [AGENTS.md](AGENTS.md).

### Local CUDA GPU

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 33   # 33 × 100M tokens (Smart Filtering reads ~3.1B)
./run_baseline.sh                     # upstream baseline, same GPU, for a fair comparison
./run.sh                              # Smart Filtering
```

Both stop at val loss ≤ 3.3821. Extra flags are passed through, e.g. `./run.sh --log_wandb`.
GPUs without bf16 (T4, V100) train in fp16 with a gradient scaler.

### Smoke test (CPU / Apple Silicon)

```bash
python train_gpt2.py --batch_size 4 --sequence_length 256 --num_iterations 20 --grad_accumulation_steps 2 \
  --learning_rate 0.0018 --warmup_iters 5 --smart_filter --filter_warmup 5 --filter_score_len 64
```

## Smart Filtering flags

| flag | default | meaning |
|---|---|---|
| `--smart_filter` | off | enable filtering (off = exact baseline) |
| `--filter_pool` | 1.25 | candidates read per kept sequence |
| `--filter_hard_frac` | 0.5 | share of ejected sequences taken from the hard end |
| `--filter_score_len` | 64 | tokens per candidate used for scoring |
| `--filter_warmup` | 128 | steps without filtering at the start |
| `--no_score_stream` | off | score on the main CUDA stream instead of a side stream |

Other additions over upstream: `--target_val_loss`, `--grad_clip` (off), `--save_every` / `--resume`
(model, optimizer and data position), `--no_compile`, CPU/MPS fallback for smoke tests.

## Files

| file | what |
|---|---|
| `train_gpt2.py` | training script (upstream model, Smart Filtering, pipeline) |
| `experiments.py` | presets, training commands, GPU-independent evaluation |
| `run.sh`, `run_baseline.sh` | full benchmark on a local GPU |
| `colab.ipynb`, `kaggle/`, `modal/` | cloud setups |
| `results/` | logs of past runs |
| `IDEA.md`, `EXPERIMENTS.md`, `AGENTS.md` | method, experiment log, instructions for agents |

---

*Fork of [BottleCapAI/NoCap-Test](https://github.com/BottleCapAI/NoCap-Test), based on
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). MIT license.*
