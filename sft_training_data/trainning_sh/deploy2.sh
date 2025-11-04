#!/bin/bash

# === 配置区 ===
MODEL_PATH="/home/jovyan2/query_rl/output/qwen3-8b-lora-sft/v3-20251031-111238/checkpoint-159-merged"  # 合并后的完整模型路径
MODEL_NAME="Qwen3-8B-SFT"                                     # 对外暴露的模型名称（OpenAI API 中使用）
PORT=8008
HOST="0.0.0.0"
API_KEY="sk-xxxx"

# GPU 设置
CUDA_VISIBLE_DEVICES=1
GPU_MEMORY_UTILIZATION=0.95    # H100 显存大，可设高些（0.9~0.95）
TENSOR_PARALLEL_SIZE=1         # Qwen3-8B 单卡可跑，若多卡可改为 2/4/8

# 性能调优参数
MAX_MODEL_LEN=32768            # Qwen3 支持长上下文，按需调整（8192 / 16384 / 32768）
MAX_NUM_SEQS=256               # 最大并发序列数（影响吞吐）
ENABLE_CHUNKED_PREFILL=true    # 启用分块预填充，提升长文本吞吐
MAX_NUM_BATCHED_TOKENS=32768   # 动态批处理最大 token 数（越大吞吐越高，但延迟略增）

# === 启动命令 ===
echo "🚀 正在启动 vLLM 服务：Qwen3-8B (SFT 合并版)"
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype bfloat16 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --enable-chunked-prefill \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --api-key "$API_KEY" \