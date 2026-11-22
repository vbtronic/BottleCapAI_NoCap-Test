# Guide: running v5 and v6 from your own Kaggle account

Thanks for helping! You will run two training jobs of a language model (GPT-2, 124M parameters) on your
Kaggle account. Everything runs in Kaggle's cloud for free; your computer only sends commands and
downloads the results.

| run | what it does | Kaggle GPU quota |
|---|---|---|
| **v5** | short comparison: Smart Filtering (pool 1.5×) vs. the baseline, 1200 steps | ~3.5 h |
| **v6** | full benchmark, Smart Filtering only, until val loss 3.3821 (at most 4768 steps) | ~25–27 h in 3 sessions |

**Who runs what:** v5 is **done** (2026-11-22, pool 1.5 won). You only run **v6**: do the setup (section 1),
then section 3. Section 2 is kept for reference.

**Quota:** Kaggle gives 30 GPU hours per week (resets on Saturday). v5 needs ~3.5 h. v6 needs ~25–27 h, so
whoever runs v6 should start with a clean quota (no other GPU use that week). Remaining quota:
https://www.kaggle.com/settings → **Quotas**.

## 1. Setup (once)

1. **A kaggle.com account with a verified phone number** (Settings → Phone Verification). Without it Kaggle
   gives no GPU and no Internet, and the job silently runs on a CPU.
2. **API token:** https://www.kaggle.com/settings → **API** → **Create New Token**. Save it:
   ```bash
   mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```
   Don't share the token and never commit it.
3. **Tools** (Python 3.10+, git; on Windows use WSL or Git Bash):
   ```bash
   pip install kaggle
   git clone https://github.com/vbtronic/BottleCapAI_NoCap-Test.git
   cd BottleCapAI_NoCap-Test
   kaggle config view          # must show your username
   ```
4. **Set your username** in both kernel configs (replace `YOUR_USERNAME` with your Kaggle username):
   ```bash
   sed -i.bak 's#viktorbrunclk/#YOUR_USERNAME/#' kaggle/kernel-metadata.json kaggle/full/kernel-metadata.json
   ```

## 2. v5 — short comparison (~3.5 h)

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_USERNAME/nocap-smartfilter      # QUEUED → RUNNING → COMPLETE
```

- `--accelerator NvidiaTeslaT4` is required (2× T4); otherwise Kaggle assigns a different GPU or none.
- If the job stays **QUEUED / "Starting…" for a long time** (more than ~30 min), Kaggle's GPUs are full.
  Cancel it on the website (notebook → Versions → ⋯ → Cancel) and **try again tomorrow**. Waiting in the
  queue does not use quota.
- When the status is **COMPLETE**, download and evaluate the results:
  ```bash
  kaggle kernels output YOUR_USERNAME/nocap-smartfilter -p results/kaggle-v5
  python experiments.py eval results/kaggle-v5/quick
  ```
- Send Viktor the `results/kaggle-v5` folder (small: logs and a plot) and the table from the last command.

## 3. v6 — full benchmark (3 sessions, ~25–27 h)

1. **Pool:** already set to `POOL = 1.5` in `kaggle/full/full_run.py` (decided by v5). Don't change it.
2. **First session:**
   ```bash
   kaggle kernels push -p kaggle/full --accelerator NvidiaTeslaT4
   kaggle kernels status YOUR_USERNAME/nocap-full
   ```
   The session stops by itself after ~11.5 h (before Kaggle's 12 h limit) and saves a checkpoint.
3. **Every next session** (when the status is **COMPLETE**):
   ```bash
   ./kaggle/full/next_session.sh
   ```
   The script downloads the logs to `results/kaggle-full/`, uploads the checkpoint (~1.5 GB) as your
   **private** dataset `nocap-ckpt` and starts the next session, which resumes from it. It downloads and
   uploads 1.5 GB, so it takes a while.
4. Repeat step 3 until the script prints **"the run is finished"** (usually after the 3rd session). Then:
   ```bash
   python experiments.py eval-full results/kaggle-full
   ```
   and send Viktor the output and the `results/kaggle-full` folder (without `.pt` files; the script deletes them).

## Troubleshooting

| problem | fix |
|---|---|
| `nvidia-smi: not found` in the log | the account has no verified phone, or `--accelerator NvidiaTeslaT4` is missing |
| `401` / `Unauthorized` | wrong or missing token (step 1.2) |
| `Permission denied` for the script | `chmod +x kaggle/full/next_session.sh` |
| the job failed (`ERROR`) | `kaggle kernels output … -p error` and send Viktor the contents of the `error` folder |
| quota ran out | wait until Saturday; the next v6 session resumes from the last checkpoint |
| stuck in QUEUED / "Starting…" | Kaggle's GPUs are full: cancel on the website and try again tomorrow |

---

*Temporary document: only for the v5 and v6 runs on other people's Kaggle accounts. It will be removed
once those runs are finished and their results are in [EXPERIMENTS.md](../EXPERIMENTS.md).*
