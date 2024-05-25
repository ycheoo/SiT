#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,3,4,5

nproc_per_node=$1
master_port=$(($RANDOM + 11451))

echo $1 $2 $3

torchrun \
    --master_port=$master_port \
    --nproc_per_node=$nproc_per_node \
    ./src/pretrain/main_mae.py --dataset $2 --config $3