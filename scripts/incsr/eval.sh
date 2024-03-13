#!/bin/bash
source ./functions.sh

EXPERIMENTS=(
    "debug_eval checkpoints/pretrain/finetune/finetune_radio_0531/checkpoint-49.pth checkpoints/incsr/debug_clsinc/pools.pth"
)

read -p "Are you sure you want to proceed? (y/n) " confirm
if [ "$confirm" = "y" ]; then
    (
        for EXP in "${EXPERIMENTS[@]}"; do
            nproc_per_node=2
            master_port=$(s2p "$EXP")

            result=($(split "$EXP"))
            config="${result[0]}"
            checkpoint_backbone="${result[1]}"
            checkpoint_pools="${result[2]}"

            echo \"$(date)\" >> ./exp.log

            torchrun \
            --master_port=$master_port \
            --nproc_per_node=$nproc_per_node \
            ./src/incsr/main_incsr.py \
            --method eval \
            --dataset radio88 \
            --domains 0531 \
            --limit_inst 0 \
            --init_inst 0 \
            --inc_inst 0 \
            --init_cls 8 \
            --inc_cls 0 \
            --test_batch_size 100 \
            --model_backbone_loc $checkpoint_backbone \
            --model_pools_loc $checkpoint_pools \
            --log_dir ./logs/incsr/$config \
            --result_dir ./results/incsr/$config \

            wait
        done
    ) &
else
    echo "Task cancelled."
fi
