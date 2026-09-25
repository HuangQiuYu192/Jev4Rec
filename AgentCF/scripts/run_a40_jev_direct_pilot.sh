#!/usr/bin/env bash
set -euo pipefail

# This is a Qwen Jev-style pilot, not an official Jev run.
export CUDA_VISIBLE_DEVICES=0
export JEV_DEVICE=cuda:0
export JEV_QWEN_MODEL=/home/hqy/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/$(ls /home/hqy/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots | head -n 1)
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/../agentcf"
mkdir -p ../../experiments/jev/outputs
python -m uvicorn jev.server:app --host 127.0.0.1 --port 8010 > ../../experiments/jev/outputs/qwen_jev_server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl --silent --fail http://127.0.0.1:8010/health >/dev/null; then break; fi
  sleep 2
done
curl --silent --fail http://127.0.0.1:8010/health >/dev/null

python run_jev_direct_rerank.py \
  --dataset-dir dataset/CDs-100-user-dense \
  --endpoint http://127.0.0.1:8010/v1/systemone \
  --users 20 \
  --seed 2026 \
  --recall-budget 10 \
  --fix-pos -1 \
  --output ../../experiments/jev/outputs/cds_dense_qwen_jev_direct_pilot20_seed2026.jsonl
