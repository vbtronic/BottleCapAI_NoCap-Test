# Kaggle runner (2x T4), all measured inside one session so both GPUs are known:
#   phase 1a  baseline on GPU0 + baseline on GPU1            -> speed difference of the two GPUs
#   phase 1b  filter pipeline, never filtering, on GPU0 + baseline on GPU1 -> overhead of the pipeline itself
#   phase 2   baseline + Smart Filtering (score_len 64), quick preset, full length, configured from phase 1:
#             - pipeline overhead > 2%  -> filter scores sequentially (--no_score_stream)
#             - one GPU > 2% faster     -> the filter runs on the slower one (its time is corrected anyway,
#                                          and a "win" can't come from a faster card)
# Push with:  kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4   (GPU T4 x2, Internet on)
# Results (logs, plot, summary.txt) land in /kaggle/working -> the notebook's Output tab.
import os
import re
import glob
import subprocess
import sys
import time

REPO = "https://github.com/vbtronic/bottlecap-nocaptest.git"
BRANCH = os.environ.get("NOCAP_BRANCH", "pipeline-kaggle")
WORK = "/kaggle/working"
SRC = "/tmp/nocap"  # outside /kaggle/working: data and checkpoints stay out of the output
DIAG_STEPS = 60
SCORE_LEN = 64

summary = []


def say(msg):
    print(msg, flush=True)
    summary.append(msg)
    with open(f"{WORK}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,power.limit --format=csv")
if not os.path.exists(SRC):
    sh(f"git clone -q -b {BRANCH} {REPO} {SRC}")
os.chdir(SRC)
sh("git log --oneline -1")
sh(f"{sys.executable} -m pip install -q huggingface_hub")
sh(f"{sys.executable} data/cached_fineweb10B.py 3")

B, T, GRAD_ACCUM = 16, 1024, 8  # 131k tokens per step (quick preset)
COMMON = (f"--model d12 --batch_size {B} --grad_accumulation_steps {GRAD_ACCUM} --sequence_length {T} "
          f"--val_batch_size 16 --learning_rate 0.0018 --weight_decay 0.1 --target_val_loss 3.3821")
QUICK = "--num_iterations 1200 --warmup_iters 100 --warmdown_iters 300 --val_loss_every 100"
DIAG = f"--num_iterations {DIAG_STEPS} --warmup_iters 10 --warmdown_iters 0 --val_loss_every 0"
FILTER = f"--smart_filter --filter_pool 1.25 --filter_hard_frac 0.5 --filter_score_len {SCORE_LEN} --filter_warmup 128"
PIPE_ONLY = "--smart_filter --filter_warmup 1000000"  # side stream + prefetch, never scores


def run_pair(phase, jobs):
    """jobs: {name: (gpu, args)}; runs them at the same time, returns {name: log path}."""
    procs, logs = {}, {}
    for name, (gpu, extra) in jobs.items():
        out = f"{WORK}/{phase}/{name}"
        os.makedirs(out, exist_ok=True)
        cmd = (f"{sys.executable} train_gpt2.py {COMMON} {extra} --output_dir {out} "
               f"--ckpt_dir /tmp/ckpt/{phase}/{name}")
        logs[name] = f"{WORK}/{phase}/{name}.txt"
        print(f"[{phase}/{name}] GPU {gpu}: {cmd}", flush=True)
        procs[name] = subprocess.Popen(cmd, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                       stdout=open(logs[name], "w"), stderr=subprocess.STDOUT)
    while any(p.poll() is None for p in procs.values()):
        time.sleep(60 if phase.startswith("1") else 600)
        for name in procs:
            lines = open(logs[name]).read().strip().splitlines()
            print(f"[{phase}/{name}] {lines[-1] if lines else '...'}", flush=True)
    for name, p in procs.items():
        if p.returncode:
            print("\n".join(open(logs[name]).read().splitlines()[-30:]))
            raise SystemExit(f"{phase}/{name} failed with exit code {p.returncode}")
    return logs


def train_times(log):
    """{step: cumulative train time in s} from the per-step lines."""
    t = {}
    for line in open(log):
        m = re.match(r"step:(\d+)/\d+ \| loss .*train_time:([\d.]+)s", line)
        if m:
            t[int(m[1])] = float(m[2])
    return t


def sec_per_step(log, a, b):
    t = train_times(log)
    return (t[b] - t[a]) / (b - a)


# ── phase 1: diagnostics (steps 10..59, after compilation) ──────────────────
a, b = 10, DIAG_STEPS - 1
l1a = run_pair("1a", {"base_gpu0": ("0", DIAG), "base_gpu1": ("1", DIAG)})
g0, g1 = sec_per_step(l1a["base_gpu0"], a, b), sec_per_step(l1a["base_gpu1"], a, b)
gpu_ratio = g1 / g0
say(f"1a baseline  GPU0 {g0:.3f} s/step | GPU1 {g1:.3f} s/step | GPU1/GPU0 = {gpu_ratio:.3f}")

l1b = run_pair("1b", {"pipe_gpu0": ("0", f"{DIAG} {PIPE_ONLY}"), "base_gpu1": ("1", DIAG)})
p0, b1 = sec_per_step(l1b["pipe_gpu0"], a, b), sec_per_step(l1b["base_gpu1"], a, b)
say(f"1b pipeline  GPU0 {p0:.3f} s/step (baseline GPU0 in 1a: {g0:.3f}) -> pipeline overhead "
    f"{100 * (p0 / g0 - 1):+.1f}% | GPU1 baseline {b1:.3f} s/step")

# ── phase 2: configured from phase 1 ────────────────────────────────────────
overhead = p0 / g0 - 1
filter_args = f"{QUICK} {FILTER}"
if overhead > 0.02:
    filter_args += " --no_score_stream"
    say(f"2 decision: side stream costs {100 * overhead:+.1f}% -> filter scores sequentially (--no_score_stream)")
else:
    say(f"2 decision: side stream costs {100 * overhead:+.1f}% -> keep the concurrent side stream")
# GPU speed: average both baseline measurements of GPU1
gpu_ratio = (g1 + b1) / 2 / g0
if gpu_ratio < 0.98:  # GPU1 faster -> baseline on GPU1, filter on GPU0
    gpu_base, gpu_filter, filter_scale = "1", "0", 1 / gpu_ratio
else:
    gpu_base, gpu_filter, filter_scale = "0", "1", gpu_ratio
say(f"2 decision: GPU1/GPU0 = {gpu_ratio:.3f} -> baseline on GPU{gpu_base}, filter on GPU{gpu_filter}, "
    f"filter time divided by {filter_scale:.3f}")
l2 = run_pair("2", {"baseline": (gpu_base, QUICK), "smartfilter": (gpu_filter, filter_args)})


def val_curve(phase, name):
    logs = sorted(glob.glob(f"{WORK}/{phase}/{name}/*.log"), key=os.path.getmtime)
    pts = [re.match(r"s:(\d+) val:([\d.]+) time:([\d.]+)", l) for l in open(logs[-1])]
    return [(int(m[1]), float(m[2]), float(m[3])) for m in pts if m]


curves = {n: val_curve("2", n) for n in ("baseline", "smartfilter")}
for n, c in curves.items():
    s, v, t = c[-1]
    say(f"2 {n:12s} step {s} val {v:.4f} time {t / 3600:.3f} h "
        f"({sec_per_step(l2[n], 200, 1199):.3f} s/step at steps 200-1199)")
fs, fv, ft = curves["smartfilter"][-1]
say(f"2 smartfilter time corrected to the baseline GPU's speed: {ft / filter_scale / 3600:.3f} h")


def time_to(curve, target, scale=1.0):
    for (s0, v0, t0), (s1, v1, t1) in zip(curve, curve[1:]):
        if v1 <= target < v0:
            return (t0 + (t1 - t0) * (v0 - target) / (v0 - v1)) / scale
    return None


target = curves["baseline"][-1][1]
tb, tf = time_to(curves["baseline"], target), time_to(curves["smartfilter"], target, filter_scale)
if tb and tf:
    say(f"2 time to val {target:.4f}: baseline {tb / 3600:.3f} h, smartfilter (GPU-corrected) {tf / 3600:.3f} h "
        f"-> {100 * (tf / tb - 1):+.1f}%")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for n, c in curves.items():
    scale = filter_scale if n == "smartfilter" else 1.0
    s, v, t = zip(*c)
    ax1.plot([x / scale / 3600 for x in t], v, marker=".", label=n + (" (GPU-corrected)" if scale != 1 else ""))
    ax2.plot(s, v, marker=".", label=n)
for ax, xl in [(ax1, "training time [h]"), (ax2, "step")]:
    ax.set_xlabel(xl); ax.set_ylabel("val loss"); ax.set_ylim(top=6); ax.grid(alpha=.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{WORK}/compare.png", dpi=120)
say(f"plot -> {WORK}/compare.png")
