#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afan/G4RegFormer

CUDA_VISIBLE_DEVICES=0 \
/home/afan/anaconda3/envs/single_cell/bin/python pretrain/main_pretrain.py \
  --model mae_bin_large \
  --batch-size 64 \
  --epochs 20 \
  --mask-ratio 0.4 \
  --valid-ratio 0.03 \
  --peak-quantile 0.995 \
  --base-lr 1e-4 \
  --num-workers 4

# Two GPUs:
# CUDA_VISIBLE_DEVICES=0,1 /home/afan/anaconda3/envs/single_cell/bin/torchrun \
#   --standalone --nproc_per_node=2 pretrain/main_pretrain.py \
#   --model mae_bin_large --batch-size 64 --epochs 20 --mask-ratio 0.4
