#!/bin/bash

python main.py \
    --dataset-path /media/pollen/T7/cornell \
    --dataset cornell \
    --use-depth 0 \
    --use-rgb 1 \
    --split 0.9 \
    --ds-rotate 0.1 \
    --num-workers 8 \
    --batch-size 32 \
    --vis False \
    --epochs 250 \
    --batches-per-epoch 200\
    --val-batches 32 \
    --description "Learning on the cornell dataset with 10% of the dataset removed for cross validation, only rgb" \
    --outdir output/cornell_rgb_only \
    --logdir cornell_rgb_only
    