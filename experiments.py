"""Shared experiment helpers for Colab, Kaggle and Modal: presets, training commands, evaluation.

    python experiments.py cmd  --preset quick --name smartfilter --pool 1.5   # print a training command
    python experiments.py eval results/kaggle-v3/2                             # compare runs in a folder

Evaluation never compares wall-clock time across two different GPUs (two T4s in one Kaggle session
differed by up to 22%, see EXPERIMENTS.md). Instead:

    steps ratio   = steps the filter needs to reach the target / steps the baseline needs  (GPU independent)
    step cost     = filtered steps vs. the same run's own unfiltered warm-up steps          (same GPU, same process)
    time ratio    = steps ratio x step cost   (< 1.0 means Smart Filtering is faster)
"""
import argparse
import glob
import math
import os
import re
import sys

TARGET = 3.3821
SEQ = 1024
TOKENS_PER_SHARD = 100_000_000

PRESETS = {
    # 157M baseline tokens, fits a free T4 (~2.5-3 h per run); does not reach 3.3821
    "quick": dict(tokens_per_step=131072, num_iterations=1200, warmup_iters=100, warmdown_iters=300,
                  val_loss_every=100),
    # the official benchmark: 4768 steps, 2.5B tokens, stops at val <= 3.3821
    "full": dict(tokens_per_step=524288, num_iterations=4768, warmup_iters=256, warmdown_iters=1024,
                 val_loss_every=128),
}

FILTER_DEFAULTS = dict(pool=1.25, hard_frac=0.5, score_len=64, warmup=128)


def shards_needed(preset, pool=None, warmup=128):
    """Training shards to download: the filter reads `pool` x more data after its warm-up."""
    p = PRESETS[preset]
    steps = p["num_iterations"]
    tokens = p["tokens_per_step"] * (min(warmup, steps) + (steps - min(warmup, steps)) * (pool or 1.0))
    return min(50, math.ceil(tokens / TOKENS_PER_SHARD) + 1)


def train_cmd(preset, out_dir, ckpt_dir, batch_size=16, pool=None, score_len=None, hard_frac=None,
              warmup=None, extra="", python=sys.executable):
    """Command line for one run. pool=None -> baseline."""
    p = PRESETS[preset]
    assert p["tokens_per_step"] % (batch_size * SEQ) == 0
    accum = p["tokens_per_step"] // (batch_size * SEQ)
    cmd = (f"{python} train_gpt2.py --model d12 --batch_size {batch_size} --grad_accumulation_steps {accum} "
           f"--sequence_length {SEQ} --val_batch_size 16 --num_iterations {p['num_iterations']} "
           f"--learning_rate 0.0018 --weight_decay 0.1 --warmup_iters {p['warmup_iters']} "
           f"--warmdown_iters {p['warmdown_iters']} --val_loss_every {p['val_loss_every']} "
           f"--target_val_loss {TARGET} --output_dir {out_dir} --ckpt_dir {ckpt_dir}")
    if pool is not None:
        f = dict(FILTER_DEFAULTS, pool=pool)
        if score_len is not None:
            f["score_len"] = score_len
        if hard_frac is not None:
            f["hard_frac"] = hard_frac
        if warmup is not None:
            f["warmup"] = warmup
        cmd += (f" --smart_filter --filter_pool {f['pool']} --filter_hard_frac {f['hard_frac']} "
                f"--filter_score_len {f['score_len']} --filter_warmup {f['warmup']}")
    return f"{cmd} {extra}".strip()


# ── evaluation ────────────────────────────────────────────────────────────────

def parse_run(path):
    """Parse the stdout log (<name>.txt) of train_gpt2.py."""
    run = dict(name=os.path.splitext(os.path.basename(path))[0], val=[], steps={}, warmup=None,
               filter=False, device="")
    for line in open(path, errors="replace"):
        if line.startswith("using device"):
            run["device"] = line.split(":", 1)[1].strip()
        m = re.match(r"\[filter\] pool=(\d+) keep=(\d+).*warmup=(\d+)", line)
        if m:
            run.update(filter=True, pool=int(m[1]) / int(m[2]), warmup=int(m[3]))
        m = re.match(r"step:(\d+)/\d+ \| val loss ([\d.na]+) \| train_time:([\d.]+)s", line)
        if m:
            run["val"].append((int(m[1]), float(m[2]), float(m[3])))
        m = re.match(r"step:(\d+)/\d+ \| loss [^|]+\| train_time:([\d.]+)s", line)
        if m:
            run["steps"][int(m[1])] = float(m[2])
    return run


def steps_to(val, target):
    """First (interpolated) step where val loss <= target, or None."""
    for (s0, v0, _), (s1, v1, _) in zip(val, val[1:]):
        if v1 <= target < v0:
            return s0 + (s1 - s0) * (v0 - target) / (v0 - v1)
    return None


def time_to(val, target):
    for (s0, v0, t0), (s1, v1, t1) in zip(val, val[1:]):
        if v1 <= target < v0:
            return t0 + (t1 - t0) * (v0 - target) / (v0 - v1)
    return None


def sec_per_step(steps, a, b):
    if a in steps and b in steps and b > a:
        return (steps[b] - steps[a]) / (b - a)
    return None


def step_cost(run, base):
    """Filtered vs. unfiltered step time inside the filter run, corrected by the baseline's drift."""
    w = run["warmup"]
    last = min(max(run["steps"]), max(base["steps"]))
    a0, a1 = 10, w - 2               # unfiltered (step w-1 already compiles the scoring graph)
    b0, b1 = min(w + 72, last - 10), last
    f_un, f_fi = sec_per_step(run["steps"], a0, a1), sec_per_step(run["steps"], b0, b1)
    g_un, g_fi = sec_per_step(base["steps"], a0, a1), sec_per_step(base["steps"], b0, b1)
    if None in (f_un, f_fi):
        return None, None
    raw = f_fi / f_un
    drift = (g_fi / g_un) if None not in (g_un, g_fi) else 1.0
    return raw / drift, raw


def evaluate(paths, baseline="baseline", plot=None):
    runs = [parse_run(p) for p in paths]
    base = next((r for r in runs if r["name"] == baseline), None) or next(r for r in runs if not r["filter"])
    reached = base["val"] and base["val"][-1][1] <= TARGET
    target = TARGET if reached else base["val"][-1][1]
    s_base = steps_to(base["val"], target) or base["val"][-1][0]
    out = [f"target val loss: {target:.4f}" + (" (benchmark target)" if reached else " (baseline's final val)"),
           "",
           "| run | GPU | final step | final val | steps to target | steps ratio | step cost | time ratio |",
           "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s, v, _ = r["val"][-1]
        st = steps_to(r["val"], target) if r is not base else s_base
        ratio = st / s_base if st else None
        cost, raw = step_cost(r, base) if r["filter"] else (1.0, 1.0)
        tr = ratio * cost if ratio and cost else None
        fmt = lambda x, f: (f % x) if x is not None else "–"
        out.append(f"| {r['name']} | {r['device'].replace('cuda ', '')} | {s} | {v:.4f} | {fmt(st, '%.0f')} | "
                   f"{fmt(ratio, '%.3f')} | {fmt(None if cost is None else (cost - 1) * 100, '%+.1f%%')} | "
                   f"**{fmt(tr, '%.3f')}** |")
    out += ["", "time ratio < 1.0 = Smart Filtering reaches the target faster than the baseline "
            "(steps ratio x step cost; wall-clock across different GPUs is not compared)."]
    text = "\n".join(out)
    print(text)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in runs:
            s, v, _ = zip(*r["val"])
            ax.plot(s, v, marker=".", label=r["name"])
        ax.axhline(target, color="gray", ls="--", lw=1)
        ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.set_ylim(top=min(6, ax.get_ylim()[1]))
        ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.savefig(plot, dpi=120)
        print(f"plot -> {plot}")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cmd")
    c.add_argument("--preset", default="quick", choices=PRESETS)
    c.add_argument("--name", default="run")
    c.add_argument("--out", default="runs")
    c.add_argument("--pool", type=float, default=None, help="filter pool (omit for baseline)")
    c.add_argument("--score_len", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("dir", help="folder with <name>.txt stdout logs")
    e.add_argument("--baseline", default="baseline")
    e.add_argument("--plot", default="")
    a = ap.parse_args()
    if a.cmd == "cmd":
        print(train_cmd(a.preset, f"{a.out}/{a.name}", f"{a.out}/ckpt/{a.name}", pool=a.pool, score_len=a.score_len))
    else:
        evaluate(sorted(glob.glob(os.path.join(a.dir, "*.txt"))), a.baseline, a.plot or None)
