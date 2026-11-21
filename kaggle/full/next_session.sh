#!/usr/bin/env bash
# Start the next session of the full Kaggle run:
#   1. download the finished session's output (logs + latest.pt) into results/kaggle-full/
#   2. upload latest.pt as a new version of the private dataset <user>/nocap-ckpt
#   3. push the kernel again with that dataset attached; it resumes from the checkpoint
# Usage: kaggle/full/next_session.sh            (run from the repo root, needs the kaggle CLI + token)
set -euo pipefail
KAGGLE=${KAGGLE:-kaggle}
command -v "$KAGGLE" >/dev/null || KAGGLE=.venv/bin/kaggle
USER_NAME=$(python3 -c "import json;print(json.load(open('kaggle/full/kernel-metadata.json'))['id'].split('/')[0])")
KERNEL="$USER_NAME/nocap-full"
DATASET="$USER_NAME/nocap-ckpt"

status=$($KAGGLE kernels status "$KERNEL" | tail -1)
echo "$status"
echo "$status" | grep -qiE "complete|error" || { echo "the session is not finished yet"; exit 1; }

OUT="results/kaggle-full/session-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
$KAGGLE kernels output "$KERNEL" -p "$OUT"
cat "$OUT"/full/STATUS 2>/dev/null || true
if grep -q DONE "$OUT"/full/STATUS 2>/dev/null; then
  echo "the run is finished — evaluate with: python experiments.py eval-full results/kaggle-full"
  exit 0
fi

CKPT=$(find "$OUT" -name latest.pt | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint in the output"; exit 1; }
DS_DIR=$(mktemp -d)
mv "$CKPT" "$DS_DIR/latest.pt"          # checkpoints stay out of git (*.pt is ignored anyway)
find "$OUT" -name "*.pt" -delete
cat > "$DS_DIR/dataset-metadata.json" <<META
{"title": "nocap-ckpt", "id": "$DATASET", "licenses": [{"name": "CC0-1.0"}]}
META
if $KAGGLE datasets status "$DATASET" >/dev/null 2>&1; then
  $KAGGLE datasets version -p "$DS_DIR" -m "checkpoint $(date +%F_%H%M)" -q
else
  $KAGGLE datasets create -p "$DS_DIR" -q            # private by default
fi
rm -rf "$DS_DIR"
until $KAGGLE datasets status "$DATASET" 2>/dev/null | grep -qi ready; do sleep 30; done

python3 - <<PY
import json
p = "kaggle/full/kernel-metadata.json"
m = json.load(open(p))
m["dataset_sources"] = ["$DATASET"]
json.dump(m, open(p, "w"), indent=2)
PY
$KAGGLE kernels push -p kaggle/full --accelerator NvidiaTeslaT4
