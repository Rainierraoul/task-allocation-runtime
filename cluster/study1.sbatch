#!/bin/bash -l
# Study 1 on the Deakin A2I2 cluster: both model servers and the study runner on one node.
#
#   MODE=provenance sbatch --export=ALL,MODE=provenance,CODE=<code dir> --time=00:45:00 cluster/study1.sbatch
#   MODE=verify  sbatch --export=ALL,MODE=verify,CODE=<code dir> cluster/study1.sbatch
#   MODE=collect sbatch --export=ALL,MODE=collect,CODE=<code dir>,WORKERS=16 cluster/study1.sbatch
#
# CODE is a copy of this repository's runtime at a recorded git commit. Output goes to
# /scratch/$USER/dcarr/runs/<MODE>-<job id>. A resubmitted collect job given the same OUT
# resumes from the saved ledger.
#SBATCH --job-name=dcarr-study1
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/dcarr/logs/%x-%j.out
#SBATCH --error=/scratch/%u/dcarr/logs/%x-%j.err
set -euo pipefail
: "${MODE:?MODE must be verify or collect}" "${CODE:?CODE must be the synced code directory}"
export DCARR_ROOT="/scratch/$USER/dcarr"
OUT="${OUT:-$DCARR_ROOT/runs/$MODE-$SLURM_JOB_ID}"
export LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > "$LOGS/gpus.csv"
echo "node $(hostname), job $SLURM_JOB_ID, code $CODE, commit $(cat "$CODE/COMMIT" 2>/dev/null || echo unknown)" | tee "$LOGS/job.txt"

cleanup() { for f in "$LOGS"/t1-server.pid "$LOGS"/t2-server.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true; done; }
trap cleanup EXIT
bash "$CODE/cluster/serve.sh"

cd "$CODE"
export HF_HOME="$DCARR_ROOT/hf" HF_HUB_OFFLINE=1
# Import this copy's runtime and study code, not whichever copy the environment was installed from.
export PYTHONPATH="$CODE/src:$CODE"
PY="$DCARR_ROOT/envs/study/bin/python"
case "$MODE" in
  provenance|verify)
    # Live provenance: constructing LocalBackend once with a fresh ledger records each server's
    # listener, launch arguments, snapshot hashes, CUDA dependencies, GPUs and SLURM facts.
    mkdir -p "$OUT/provenance"
    "$PY" - "$OUT/provenance/provider-calls.jsonl" <<'PROVENANCE' || echo "PROVENANCE CHECK FAILED; see the job's error log" | tee "$OUT/provenance/FAILED"
import json, sys
from pathlib import Path
from study1.backend import LocalBackend
LocalBackend(json.load(open("study1/config.cluster.json")), Path(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[1])]
print("provenance rows:", [r["kind"] for r in rows])
PROVENANCE
    if [ "$MODE" = provenance ]; then [ -f "$OUT/provenance/FAILED" ] && exit 1; exit 0; fi
    "$PY" -m study1.verify_local --config study1/config.cluster.json --out "$OUT/heldout-verification"
    ;;
  collect)
    [ -f "$OUT/materials.json" ] || "$PY" -m study1.instances --out "$OUT/materials.json"
    "$PY" -m study1.run all --materials "$OUT/materials.json" --config study1/config.cluster.json \
      --out "$OUT/study" --workers "${WORKERS:-16}"
    "$PY" -m study1.analyze "$OUT/study"
    ;;
  *) echo "unknown MODE $MODE" >&2; exit 2 ;;
esac
