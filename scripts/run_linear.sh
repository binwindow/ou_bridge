#!/bin/bash
# GOUB Deraining: Linear theta schedule
set -e

DATAROOT_GT="${DATAROOT_GT:-/path/to/rain100h/gt}"
DATAROOT_LQ="${DATAROOT_LQ:-/path/to/rain100h/lq}"

python tasks/deraining/train.py \
  --exp_name derain_linear \
  --schedule linear \
  --total_iterations 200000 \
  --dataroot_GT "$DATAROOT_GT" \
  --dataroot_LQ "$DATAROOT_LQ"
