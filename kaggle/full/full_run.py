# Full benchmark on Kaggle, Smart Filtering only, spread over several 12 h sessions.
#
# One run (4768 steps, stops at val <= 3.3821) uses both T4s together (DDP, ~22 h of training), so it
# needs ~2-3 sessions. Every session resumes from the checkpoint of the previous one, which
# kaggle/full/next_session.sh downloads and re-uploads as the private dataset <user>/nocap-ckpt.
#
#   first session:  kaggle kernels push -p kaggle/full --accelerator NvidiaTeslaT4
#   next sessions:  kaggle/full/next_session.sh
#
# The baseline is not re-run: by definition the official baseline reaches 3.3821 at step 4768, so
#   steps ratio = filter steps to 3.3821 / 4768, step cost = filtered vs. unfiltered steps of session 1.
import glob
import os
import signal
import subprocess
import sys
import time

START = time.time()
REPO = "https://github.com/vbtronic/NoCap-Test.git"
BRANCH = "main"
POOL = 1.5         # chosen from Kaggle v5 (time ratio 0.984 vs. 1.015 for pool 1.25 in v3)
SCORE_LEN = 64
SAVE_EVERY = 50    # ~15 min of training on 2x T4; at most that much is lost per session
STOP_AFTER_H = 11.3  # stop training before Kaggle's 12 h limit kills the session (outputs would be lost)
WORK = "/kaggle/working"
SRC = "/tmp/nocap"


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
sh(f"{sys.executable} data/cached_fineweb10B.py {ex.shards_needed('full', POOL)}")

ckpts = sorted(glob.glob("/kaggle/input/**/latest.pt", recursive=True), key=os.path.getmtime)
resume = f"--resume {ckpts[-1]}" if ckpts else ""
print("resume from:", ckpts[-1] if ckpts else "nothing (first session)", flush=True)

n_gpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout)
launcher = f"torchrun --standalone --nproc_per_node={n_gpu}" if n_gpu > 1 else sys.executable
cmd = ex.train_cmd("full", f"{WORK}/full", f"{WORK}/ckpt", pool=POOL, score_len=SCORE_LEN,
                   extra=f"--save_every {SAVE_EVERY} {resume}", python=launcher)
os.makedirs(f"{WORK}/full", exist_ok=True)
log_path = f"{WORK}/full/session_{time.strftime('%Y%m%d_%H%M')}.txt"
print(cmd, flush=True)
proc = subprocess.Popen(cmd, shell=True, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                        start_new_session=True)

stopped = False
while proc.poll() is None:
    time.sleep(60)
    hours = (time.time() - START) / 3600
    if int(hours * 60) % 15 == 0:
        lines = open(log_path).read().strip().splitlines()
        print(f"[{hours:.2f} h] {lines[-1] if lines else '...'}", flush=True)
    if hours > STOP_AFTER_H:
        print(f"[{hours:.2f} h] time limit: stopping, the next session resumes from the last checkpoint", flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        stopped = True
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        break

text = open(log_path).read()
done = "target val loss" in text or (proc.returncode == 0 and not stopped)
status = "DONE" if done else ("CONTINUE" if stopped else f"FAILED (exit {proc.returncode})")
print("\n".join(text.strip().splitlines()[-15:]))
print(f"\nSESSION STATUS: {status}", flush=True)
open(f"{WORK}/full/STATUS", "w").write(status + "\n")
for p in glob.glob(f"{WORK}/ckpt/**/final.pt", recursive=True) + glob.glob(f"{WORK}/ckpt/**/*.tmp", recursive=True):
    os.remove(p)  # latest.pt is enough; keeps the output (and the download) at ~1.5 GB
if status.startswith("FAILED"):
    sys.exit(1)
