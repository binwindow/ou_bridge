#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"
python tasks/deraining/train.py \
  --exp_name cudnn_deterministic \
  --sde_type goub \
  --schedule linear \
  --ema_beta 0.999 \
  --no_amp \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT" \
  --cudnn_deterministic "$@"
