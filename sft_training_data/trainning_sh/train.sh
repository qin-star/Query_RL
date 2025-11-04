#!/bin/bash

# ==============================
# Qwen3-8B LoRA SFT 训练脚本（Swift 2.x 官方风格）
# 适配本地模型 + 小数据集（467条）
# ==============================

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="/home/jovyan2/query_rl/model/Qwen3-8B"
DATASET_PATH="/home/jovyan2/query_rl/sft_training_data/data/sft/chengla_v2/dataset_latest_new.json"
OUTPUT_DIR="/home/jovyan2/query_rl/output/qwen3-8b-lora-sft"

# 超参数（参考官方 + 小数据优化）
swift sft \
    --model "$MODEL_PATH" \
    --train_type lora \
    --dataset "$DATASET_PATH" \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --loss_scale ignore_empty_think \
    --overwrite_output_dir