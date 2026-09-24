#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afan/G4RegFormer
/home/afan/anaconda3/envs/single_cell/bin/python -m downstream.CRERegGene.main \
  --pretrain-checkpoint pretrain/output/20260823-172131-mae_bin_large/checkpoint-19.pth
