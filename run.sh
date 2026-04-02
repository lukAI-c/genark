#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# Optional: activate your conda env if needed
# source "/data1/home/clk/anaconda3/etc/profile.d/conda.sh"
# conda activate genesis

cd /data1/home/clk/project/genark

python orchestrator.py \
  --model_gpu 2 \
  --worker_gpus 3 4 5 \
  --max_parallel 300 \
  --max_concurrent 3 \
  --output metrics_final.json \
  --port 8102