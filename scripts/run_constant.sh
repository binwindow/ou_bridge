#!/bin/bash
# GOUB Deraining: Constant theta schedule
set -e

DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"

python tasks/deraining/train.py \
  --exp_name derain_constant \
  --schedule constant \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT"
