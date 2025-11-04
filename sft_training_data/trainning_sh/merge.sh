CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters /home/jovyan2/query_rl/output/qwen3-8b-lora-sft/v3-20251031-111238/checkpoint-159 \
    --merge_lora true

