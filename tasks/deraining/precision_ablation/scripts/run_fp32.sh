#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"
python tasks/train.py \
  --output_root outputs/derainh \
  --exp_name "${EXP_NAME:-precision_fp32}" \
  --sde_type goub \
  --schedule linear \
  --precision fp32 \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT" "$@"
