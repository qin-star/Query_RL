CUDA_VISIBLE_DEVICES=1 \
swift app \
    --adapters /home/jovyan2/query_rl/output/v19-20251028-165612/checkpoint-120 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --vllm_max_model_len 8192 \
    --response_prefix '<think>\n\n</think>\n\n'