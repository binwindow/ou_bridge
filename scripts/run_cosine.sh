#!/bin/bash
# GOUB Deraining: Cosine theta schedule
set -e

DATA_ROOT="${DATA_ROOT:-source/DeRain-H}"

python tasks/deraining/train.py \
  --exp_name derain_cosine \
  --schedule cosine \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT"
