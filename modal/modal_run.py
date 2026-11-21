"""Modal runner: the full benchmark (4768 steps, stop at val <= 3.3821) on rented GPUs.

    pip install modal && modal setup                       # once: log in
    modal run modal/modal_run.py                           # full preset, baseline + filter, H100, in parallel
    modal run modal/modal_run.py --preset quick --gpu A100 # cheaper smoke run
    modal run modal/modal_run.py --runs smartfilter --pool 1.5
    modal volume get nocap runs/full results/modal-full    # download logs afterwards

Each run gets its own container with the same GPU type; they run in parallel. The data is downloaded
once into the `nocap` volume and shared. Results (stdout logs, summary.md, compare.png) are stored in
the volume under runs/<preset>/. `--detach` keeps the runs going if your laptop goes to sleep.

Cost (Nov 2026 list prices, check modal.com/pricing): H100 ~$4/h, A100-80GB ~$2.5/h. A full run takes
roughly 2-2.5 h on an H100, so baseline + filter is about $20; new accounts get $30/month free credit.
"""
import os
import subprocess
import sys

import modal

REPO = "https://github.com/vbtronic/BottleCapAI_NoCap-Test.git"
BRANCH = "main"
VOL = "/vol"

app = modal.App("nocap-smartfilter")
volume = modal.Volume.from_name("nocap", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", "numpy", "huggingface_hub", "matplotlib")
)


def _checkout():
    src = "/root/nocap"
    if not os.path.exists(src):
        subprocess.run(f"git clone -q -b {BRANCH} {REPO} {src}", shell=True, check=True)
    os.chdir(src)
    sys.path.insert(0, src)
    # data lives in the volume, the code expects it in data/fineweb10B
    os.makedirs(f"{VOL}/fineweb10B", exist_ok=True)
    if not os.path.islink("data/fineweb10B"):
        subprocess.run("rm -rf data/fineweb10B", shell=True, check=True)
        os.symlink(f"{VOL}/fineweb10B", "data/fineweb10B")
    return src


@app.function(image=image, volumes={VOL: volume}, timeout=3 * 3600, cpu=4)
def download(shards: int):
    _checkout()
    subprocess.run(f"python data/cached_fineweb10B.py {shards}", shell=True, check=True)
    volume.commit()


@app.function(image=image, volumes={VOL: volume}, timeout=24 * 3600, gpu="H100", cpu=8)
def train(name: str, preset: str, pool: float | None, score_len: int | None):
    _checkout()
    import experiments as ex

    out = f"{VOL}/runs/{preset}"
    os.makedirs(out, exist_ok=True)
    cmd = ex.train_cmd(preset, f"{out}/{name}", f"/tmp/ckpt/{name}", pool=pool, score_len=score_len)
    print(cmd, flush=True)
    with open(f"{out}/{name}.txt", "w") as log:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for i, line in enumerate(p.stdout):
            log.write(line)
            print(f"[{name}] {line}", end="", flush=True)
            if i % 200 == 0:
                log.flush()
                volume.commit()  # partial logs survive a crash
    volume.commit()
    if p.wait():
        raise RuntimeError(f"{name} failed with exit code {p.returncode}")
    return f"{out}/{name}.txt"


@app.function(image=image, volumes={VOL: volume}, timeout=600)
def evaluate(preset: str, names: list[str]):
    _checkout()
    import experiments as ex

    volume.reload()
    out = f"{VOL}/runs/{preset}"
    text = ex.evaluate([f"{out}/{n}.txt" for n in names], plot=f"{out}/compare.png")
    open(f"{out}/summary.md", "w").write(text + "\n")
    volume.commit()
    return text


@app.local_entrypoint()
def main(preset: str = "full", gpu: str = "H100", runs: str = "baseline,smartfilter",
         pool: float = 1.25, score_len: int = 64):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import experiments as ex  # local copy, only for shards_needed

    names = [r.strip() for r in runs.split(",") if r.strip()]
    max_pool = pool if any(n != "baseline" for n in names) else None
    download.remote(ex.shards_needed(preset, max_pool))
    trainer = train.with_options(gpu=gpu)
    calls = [trainer.spawn(n, preset, None if n == "baseline" else pool, None if n == "baseline" else score_len)
             for n in names]
    for c in calls:
        print("finished:", c.get())
    if "baseline" in names and len(names) > 1:
        print(evaluate.remote(preset, names))
