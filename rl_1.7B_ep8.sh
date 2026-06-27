#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

EXP_NUM="8"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    accelerate launch \
            --config_file ./config/zero2_opt.yaml \
            --num_processes 4 --main_process_port 29510 \
            rl.py \
        --model_path output_dir/Qwen3-1.7B_sft_ep1/checkpoint-375 \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 4 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ranking \
        --num_generations 16 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 0.2 \
        --dapo False \
        --output_dir output_dir/Qwen3-1.7B_RL/ep${EXP_NUM} \
        --wandb_project MiniOneRec_Qwen3-1.7B \
        --wandb_run_name rl_4xA6000_ep${EXP_NUM} \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
