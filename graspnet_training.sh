#!/bin/bash

python main_grasp_1b.py \
    --dataset-path /media/pollen/T7/graspnet \
    --dataset graspnet1b \
    --use-depth 1 \
    --use-rgb 1 \
    --split 0.9 \
    --ds-rotate 0.1 \
    --num-workers 8 \
    --batch-size 32 \
    --vis False \
    --epochs 1000 \
    --batches-per-epoch 220\
    --val-batches 32 \
    --description "Learning on the graspnet dataset, 10%cross-validation, embed_dim=96, heads=[3,6,12,24]" \
    --outdir output/graspnet2 \
    --logdir graspnet2_logs
 
