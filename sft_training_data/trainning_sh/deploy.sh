# 部署后，使用/home/jovyan2/query_rl/fill_GPU_Qwen3.py 这个程序，提升GPU利用率

CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model /home/jovyan2/query_rl/model/Qwen3-8B \
    --adapters /home/jovyan2/query_rl/output/v19-20251028-165612/checkpoint-120 \
    --infer_backend vllm \
    --temperature 0 \
    --host 0.0.0.0 \
    --port 8007 \
    --api-key "sk-xxxx" \
    --max_new_tokens 2048

