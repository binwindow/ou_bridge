#!/bin/bash
# GOUB Deraining: Linear theta schedule
set -e

DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"

python tasks/deraining/train.py \
  --exp_name derain_linear \
  --schedule linear \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT"
