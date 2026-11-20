# Kaggle runner: baseline on GPU 0 and Smart Filtering on GPU 1 at the same time (2x T4).
# Push with:  kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4   (GPU T4 x2, Internet on)
# Results (logs, plot) land in /kaggle/working and appear under the notebook's Output tab.
import os
import re
import glob
import subprocess
import sys
import time

REPO = "https://github.com/vbtronic/bottlecap-nocaptest.git"
BRANCH = os.environ.get("NOCAP_BRANCH", "pipeline-kaggle")
PRESET = os.environ.get("NOCAP_PRESET", "quick")  # quick | full (full does not fit Kaggle's 12 h limit on T4)
WORK = "/kaggle/working"
SRC = "/tmp/nocap"  # outside /kaggle/working: data and checkpoints stay out of the output
OUT = f"{WORK}/{PRESET}"


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
if not os.path.exists(SRC):
    sh(f"git clone -q -b {BRANCH} {REPO} {SRC}")
os.chdir(SRC)
sh("git log --oneline -1")
sh(f"{sys.executable} -m pip install -q huggingface_hub")

if PRESET == "full":
    cfg = dict(tokens_per_step=524288, num_iterations=4768, warmup_iters=256, warmdown_iters=1024,
               val_loss_every=128, shards=33)
else:
    cfg = dict(tokens_per_step=131072, num_iterations=1200, warmup_iters=100, warmdown_iters=300,
               val_loss_every=100, shards=3)
B, T = 16, 1024
grad_accum = cfg["tokens_per_step"] // (B * T)
sh(f"{sys.executable} data/cached_fineweb10B.py {cfg['shards']}")

FILTER = "--smart_filter --filter_pool 1.25 --filter_hard_frac 0.5 --filter_score_len 256 --filter_warmup 128"
runs = {"baseline": ("0", ""), "smartfilter": ("1", FILTER)}
n_gpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout)

procs = {}
for name, (gpu, extra) in runs.items():
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    cmd = (f"{sys.executable} train_gpt2.py --model d12 --batch_size {B} --grad_accumulation_steps {grad_accum} "
           f"--sequence_length {T} --val_batch_size 16 --num_iterations {cfg['num_iterations']} "
           f"--learning_rate 0.0018 --weight_decay 0.1 --warmup_iters {cfg['warmup_iters']} "
           f"--warmdown_iters {cfg['warmdown_iters']} --val_loss_every {cfg['val_loss_every']} "
           f"--target_val_loss 3.3821 --output_dir {OUT}/{name} --ckpt_dir /tmp/ckpt/{name} {extra}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu if n_gpu > 1 else "0")
    print(f"[{name}] GPU {env['CUDA_VISIBLE_DEVICES']}: {cmd}", flush=True)
    procs[name] = subprocess.Popen(cmd, shell=True, env=env, stdout=open(f"{OUT}/{name}.txt", "w"),
                                   stderr=subprocess.STDOUT)
    if n_gpu < 2:
        procs[name].wait()  # one GPU: run them one after another

# progress: print the last line of each log every 10 minutes
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    for name in procs:
        lines = open(f"{OUT}/{name}.txt").read().strip().splitlines()
        print(f"[{name}] {lines[-1] if lines else '...'}", flush=True)

for name, p in procs.items():
    print(f"\n===== {name} (exit {p.returncode}) =====")
    print("\n".join(open(f"{OUT}/{name}.txt").read().splitlines()[-15:]))

# compare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for name in runs:
    logs = sorted(glob.glob(f"{OUT}/{name}/*.log"), key=os.path.getmtime)
    if not logs:
        continue
    pts = [re.match(r"s:(\d+) val:([\d.]+) time:([\d.]+)", l) for l in open(logs[-1])]
    pts = [(int(m[1]), float(m[2]), float(m[3]) / 3600) for m in pts if m]
    if not pts:
        continue
    s, v, h = zip(*pts)
    ax1.plot(h, v, marker=".", label=name)
    ax2.plot(s, v, marker=".", label=name)
    print(f"{name:12s} last: step {s[-1]}, val {v[-1]:.4f}, {h[-1]:.2f} h")
for ax, xl in [(ax1, "training time [h]"), (ax2, "step")]:
    ax.axhline(3.3821, color="gray", ls="--", lw=1)
    ax.set_xlabel(xl); ax.set_ylabel("val loss"); ax.set_ylim(top=6); ax.grid(alpha=.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/compare.png", dpi=120)
print(f"plot -> {OUT}/compare.png")
if any(p.returncode for p in procs.values()):
    sys.exit(1)
