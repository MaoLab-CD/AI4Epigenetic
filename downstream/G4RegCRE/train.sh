#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afan/G4RegFormer
/home/afan/anaconda3/envs/single_cell/bin/python -m downstream.G4RegCRE.main \
  --data data/preprocessed/downstream_data/G4RegCRE/G4_1000bp_Reg_CRE_ATACPeak/training_data/g4_reg_cre_dataset.h5 \
  --pretrain-checkpoint pretrain/output/20260823-172131-mae_bin_large/checkpoint-19.pth \
  --batch-size 8 \
  --max-g4-per-cre 256
