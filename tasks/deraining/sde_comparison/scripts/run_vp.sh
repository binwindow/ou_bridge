#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"
python tasks/deraining/train.py \
  --exp_name sde_vp \
  --sde_type vp \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --num_steps_sampling 40 \
  --data_root "$DATA_ROOT" "$@"
