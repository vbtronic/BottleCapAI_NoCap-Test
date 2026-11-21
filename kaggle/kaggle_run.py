# Kaggle runner (GPU T4 x2): two runs at the same time, one per GPU, then evaluation.
#
#   kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4      (Internet on, phone-verified account)
#
# Edit RUNS below to choose what runs where. Results (stdout logs, summary.md, compare.png) are written
# to /kaggle/working and appear under the notebook's Output tab:
#   kaggle kernels output <user>/nocap-smartfilter -p results/kaggle-vN
import os
import subprocess
import sys
import time

REPO = "https://github.com/vbtronic/bottlecap-nocaptest.git"
BRANCH = "main"
PRESET = "quick"  # "full" needs ~45 h per run on a T4 -> does not fit Kaggle's 12 h limit; use Modal
WORK = "/kaggle/working"
SRC = "/tmp/nocap"  # outside /kaggle/working: data and checkpoints stay out of the output

# name: (GPU index, filter pool or None for the baseline, score_len)
RUNS = {
    "smartfilter_p15": ("0", 1.5, 64),
    "baseline": ("1", None, None),
}


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
if not os.path.exists(SRC):
    sh(f"git clone -q -b {BRANCH} {REPO} {SRC}")
os.chdir(SRC)
sys.path.insert(0, SRC)
import experiments as ex  # noqa: E402

sh("git log --oneline -1")
sh(f"{sys.executable} -m pip install -q huggingface_hub")
max_pool = max((pool or 1.0) for _, pool, _ in RUNS.values())
sh(f"{sys.executable} data/cached_fineweb10B.py {ex.shards_needed(PRESET, max_pool)}")

out = f"{WORK}/{PRESET}"
os.makedirs(out, exist_ok=True)
n_gpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout)
procs = {}
for name, (gpu, pool, score_len) in RUNS.items():
    cmd = ex.train_cmd(PRESET, f"{out}/{name}", f"/tmp/ckpt/{name}", pool=pool, score_len=score_len)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu if n_gpu > 1 else "0")
    print(f"[{name}] GPU {env['CUDA_VISIBLE_DEVICES']}: {cmd}", flush=True)
    procs[name] = subprocess.Popen(cmd, shell=True, env=env, stdout=open(f"{out}/{name}.txt", "w"),
                                   stderr=subprocess.STDOUT)
    if n_gpu < 2:
        procs[name].wait()  # a single GPU: one run after the other

while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    for name in procs:
        lines = open(f"{out}/{name}.txt").read().strip().splitlines()
        print(f"[{name}] {lines[-1] if lines else '...'}", flush=True)

failed = [n for n, p in procs.items() if p.returncode]
for name in failed:
    print(f"===== {name} failed =====\n" + "\n".join(open(f"{out}/{name}.txt").read().splitlines()[-30:]))

text = ex.evaluate([f"{out}/{n}.txt" for n in RUNS if n not in failed], plot=f"{WORK}/compare.png")
open(f"{WORK}/summary.md", "w").write(text + "\n")
if failed:
    sys.exit(1)
