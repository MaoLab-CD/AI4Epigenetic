#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afan/G4RegFormer

CUDA_VISIBLE_DEVICES=1 \
/home/afan/anaconda3/envs/single_cell/bin/python finetune/main_finetune.py \
  --model polii_peak_large \
  --finetune pretrain/output/20260730-234159-mae_bin_large/checkpoint-19.pth \
  --batch_size 64 \
  --eval_batch_size 128 \
  --epochs 30 \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --blr 1e-3 \
  --encoder_lr_scale 0.1 \
  --save_metric loss \
  --early_stop 10 \
  --num_workers 4

# Scratch baseline using the identical split and normalization:
# CUDA_VISIBLE_DEVICES=0 \
# /home/afan/anaconda3/envs/single_cell/bin/python finetune/main_finetune.py \
#   --model polii_peak_large --scratch --batch_size 64 --epochs 30 --seed 0
