#!/bin/bash
source ./functions.sh

EXPERIMENTS=(
    "r0605u0508_b20_display r0605u0508_modified 904785 checkpoints/pretrain/finetune/finetune_radio_0531-0603/checkpoint-199.pth"
    # "r0606u0818_b5 r0606u0818_modified checkpoints/pretrain/finetune/finetune_radio_0531-0603/checkpoint-199.pth"
    # "r0607u0822_b5 r0607u0822_modified checkpoints/pretrain/finetune/finetune_radio_0531-0603/checkpoint-199.pth"
)
read -p "Are you sure you want to proceed? (y/n) " confirm
if [ "$confirm" = "y" ]; then
    (
        for EXP in "${EXPERIMENTS[@]}"; do
            nproc_per_node=1
            master_port=$(s2p "$EXP")

            result=($(split "$EXP"))
            config="${result[0]}"
            domains="${result[1]}"
            seed="${result[2]}"
            checkpoint="${result[3]}"

            echo \"$(date)\" >> ./exp.log

            torchrun \
            --master_port=$master_port \
            --nproc_per_node=$nproc_per_node \
            ./src/incsr/main_incsr.py \
            --dataset incsr16 \
            --domains $domains \
            --train_batch_size 40 \
            --model_backbone_loc $checkpoint \
            --log_dir ./logs/incsr/$config \
            --result_dir ./results/incsr/$config \

            wait
        done
    ) &
else
    echo "Task cancelled."
fi
