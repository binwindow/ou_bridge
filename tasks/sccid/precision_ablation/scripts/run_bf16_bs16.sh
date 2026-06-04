#!/bin/bash
set -e
DATA_ROOT="${DATA_ROOT:-/home/lsc/project/dataset/S-CCID}"
python tasks/train.py \
  --output_root outputs/sccid \
  --exp_name "${EXP_NAME:-precision_bf16_bs16}" \
  --dataset sccid \
  --sde_type goub \
  --schedule linear \
  --precision bf16 \
  --batch_size 16 \
  --gpu "${GPU:-0}" \
  --total_iterations 200000 \
  --data_root "$DATA_ROOT" "$@"
