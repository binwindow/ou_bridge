#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"
python tasks/train.py \
  --output_root outputs/derainh \
  --exp_name goub_linear_ema099_fp32 \
  --sde_type goub \
  --schedule linear \
  --ema_beta 0.99 \
  --precision fp32 \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT" "$@"
