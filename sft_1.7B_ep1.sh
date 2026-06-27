export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon23/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon23/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon23/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon23/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 1 \
            sft.py \
            --base_model Qwen/Qwen3-1.7B \
            --batch_size 1024 \
            --micro_batch_size 8 \
            --learning_rate 1e-4 \
            --num_epochs 10 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/m1Rec_Qwen3-1.7B_sft_ep1 \
            --wandb_project SpecRec_m1Rec_Qwen3-1.7B \
            --wandb_run_name ep1_lr_1e-4 \
            --wandb_group_name m1Rec_SFT_1.7B \
            --category ${category} \
            --train_from_scratch False \
            --seed 22 \
            --sid_index_path ./data/Amazon23/Industrial_and_Scientific/Industrial_and_Scientific.opq.index.json \
            --item_meta_path ./data/Amazon23/Industrial_and_Scientific/Industrial_and_Scientific.item.json \
            --freeze_LLM False
done