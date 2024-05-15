#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

nproc_per_node=6
master_port=$(($RANDOM + 11451))
echo $1 $2
torchrun \
--master_port=$master_port \
--nproc_per_node=$nproc_per_node \
./src/pretrain/main_finetune.py --dataset $1 --config $2