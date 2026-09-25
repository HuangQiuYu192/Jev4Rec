#!/usr/bin/env bash
set -euo pipefail

# A small validation step before enabling Jev in AgentCF's stateful updates.
export CUDA_VISIBLE_DEVICES=0
export JEV_DEVICE=cuda:0
export JEV_QWEN_MODEL=/home/hqy/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/$(ls /home/hqy/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots | head -n 1)
export TOKENIZERS_PARALLELISM=false

source "${CONDA_HOME:-/home/hqy/miniconda3}/etc/profile.d/conda.sh"
conda activate "${JEV_CONDA_ENV:-jev4rec}"

cd "$(dirname "$0")/../agentcf"
mkdir -p ../../experiments/jev/outputs
python -m uvicorn jev.server:app --host 127.0.0.1 --port 8010 > ../../experiments/jev/outputs/qwen_jev_gate_server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl --silent --fail http://127.0.0.1:8010/health >/dev/null; then break; fi
  sleep 2
done
curl --silent --fail http://127.0.0.1:8010/health >/dev/null

python run_jev_pairwise_gate_probe.py \
  --dataset-dir dataset/CDs-100-user-dense \
  --endpoint http://127.0.0.1:8010/v1/systemone \
  --users "${JEV_GATE_USERS:-1}" \
  --seed 2026 \
  --history-length 8 \
  --recall-budget 10 \
  --use-pretrained-descriptions \
  --output ../../experiments/jev/outputs/cds_dense_qwen_jev_pairwise_gate_probe.jsonl
