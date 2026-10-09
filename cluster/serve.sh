#!/bin/bash
# Start both study model servers on this node and wait until they answer.
# T2 (Qwen3.8-27B, bf16) is tensor-parallel on the first two GPUs SLURM allocated; T1
# (Qwen3-4B-Instruct-2507, bf16) on the third. Inside a job, SLURM numbers the allocated GPUs
# 0..n-1 in CUDA_VISIBLE_DEVICES, while their device minor numbers can differ (on l40s-f-02 a job
# saw minor numbers 0, 1 and 3). Each server is therefore given its GPUs by UUID, which vLLM
# resolves through NVML and the provenance inspector checks against nvidia-smi.
# Revisions are pinned and must match study1/config.cluster.json. Logs go to $LOGS.
set -euo pipefail
: "${DCARR_ROOT:?}" "${LOGS:?}"
VLLM="$DCARR_ROOT/envs/vllm/bin/vllm"
export HF_HOME="$DCARR_ROOT/hf" HF_HUB_OFFLINE=1
mkdir -p "$LOGS"
IFS=, read -r -a ALLOCATED <<< "${CUDA_VISIBLE_DEVICES:?SLURM did not set CUDA_VISIBLE_DEVICES}"
if (( ${#ALLOCATED[@]} < 3 )); then echo "need three allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2; exit 1; fi
uuid_of() { nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader | tr -d ' '; }
T2_GPUS="$(uuid_of "${ALLOCATED[0]}"),$(uuid_of "${ALLOCATED[1]}")"
T1_GPU="$(uuid_of "${ALLOCATED[2]}")"
for gpu in "${ALLOCATED[@]:0:3}"; do
  echo "GPU $gpu: $(nvidia-smi -i "$gpu" --query-gpu=name,uuid,pci.bus_id --format=csv,noheader)"
done | tee "$LOGS/gpu-assignment.txt"
echo "T2 on $T2_GPUS, T1 on $T1_GPU" | tee -a "$LOGS/gpu-assignment.txt"

CUDA_VISIBLE_DEVICES="$T2_GPUS" "$VLLM" serve Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --host 127.0.0.1 --port 8001 --tensor-parallel-size 2 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 --enable-prefix-caching \
  > "$LOGS/t2-server.log" 2>&1 &
echo $! > "$LOGS/t2-server.pid"

CUDA_VISIBLE_DEVICES="$T1_GPU" "$VLLM" serve Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 --enable-prefix-caching \
  > "$LOGS/t1-server.log" 2>&1 &
echo $! > "$LOGS/t1-server.pid"

deadline=$((SECONDS + 1800))
for port in 8000 8001; do
  until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
    if (( SECONDS > deadline )); then echo "server on port $port did not start within 30 minutes" >&2; exit 1; fi
    for pidfile in "$LOGS"/t1-server.pid "$LOGS"/t2-server.pid; do
      kill -0 "$(cat "$pidfile")" 2>/dev/null || { echo "a server exited; see $LOGS" >&2; exit 1; }
    done
    sleep 10
  done
  echo "server on port $port is up"
done
