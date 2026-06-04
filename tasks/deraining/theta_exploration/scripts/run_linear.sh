#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"
python tasks/train.py \
  --output_root outputs/derainh \
  --exp_name goub_linear \
  --sde_type goub \
  --schedule linear \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT" "$@"
