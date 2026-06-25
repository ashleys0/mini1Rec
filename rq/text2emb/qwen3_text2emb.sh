accelerate launch --num_processes 1 qwen3_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon23/Industrial_and_Scientific \
    --plm_name qwen \
    --plm_checkpoint Qwen/Qwen3-Embedding-8B \
    --features title,description \
    --max_sent_len 2048 \
    --batch_size 8
