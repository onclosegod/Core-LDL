#!/usr/bin/env bash
# Four-GPU RAF-ML example. Batch size is per process (global batch size: 512).
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  train_torch.py \
  --multi-gpu \
  --dataset RAF_ML \
  --dataset-dir ./dataset \
  --missing-rates 0.1 0.5 0.9 \
  --num-bins 100 \
  --lr 5e-3 \
  --batch-size 128 \
  --n-outer-repeats 5 \
  --n-splits 5 \
  --random-seed 42 \
  --output-dir ./outputs/main
