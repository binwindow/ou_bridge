#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"
python tasks/deraining/train.py \
  --exp_name "${EXP_NAME:-precision_fp16_bs16}" \
  --sde_type goub \
  --schedule linear \
  --precision fp16 \
  --batch_size 16 \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT" "$@"
