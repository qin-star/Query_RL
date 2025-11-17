#!/bin/bash

# 持续训练脚本
export CUDA_VISIBLE_DEVICES=0

# 配置
INITIAL_MODEL_PATH="/home/jovyan2/query_rl/output/qwen3-8b-lora-sft/v3-20251031-111238/checkpoint-159-merged"
OUTPUT_DIR="outputs/sales_rag_rl_continuous"
CHECKPOINT_INTERVAL=500  # 每500步保存一次
VLLM_PORT=8007
VLLM_SLEEP_TIME=30  # vllm重启后等待时间（秒）
TRAINING_INTERVAL=1800  # 训练间隔时间（秒），30分钟

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 初始化模型路径
CURRENT_MODEL_PATH=${INITIAL_MODEL_PATH}

# 函数：启动 vLLM 服务
start_vllm() {
    local model_path=$1
    local model_name=$2
    
    echo "[$(date)] 启动 vLLM 服务，模型: ${model_path}"
    
    # 先检查端口是否被占用
    if lsof -i :${VLLM_PORT} >/dev/null 2>&1; then
        echo "[$(date)] 端口 ${VLLM_PORT} 已被占用，尝试杀死相关进程..."
        # 使用更精确的进程匹配方式
        pkill -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}"
        sleep ${VLLM_SLEEP_TIME}
    fi
    
    # 启动 vLLM 服务
    echo "[$(date)] 启动 vLLM 命令: CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model ${model_path} --served-model-name ${model_name} --port ${VLLM_PORT} --max-model-len 4096 --gpu-memory-utilization 0.4 --dtype bfloat16 --enforce-eager"
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
            --model ${model_path} \
            --served-model-name ${model_name} \
            --port ${VLLM_PORT} \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.4 \
            --dtype bfloat16 \
            --enforce-eager > ${OUTPUT_DIR}/vllm.log 2>&1 &
    
    # 记录进程ID
    VLLM_PID=$!
    echo "[$(date)] vLLM 服务已启动，进程ID: ${VLLM_PID}"
    
    # 等待服务启动
    echo "[$(date)] 等待 vLLM 服务启动..."
    sleep ${VLLM_SLEEP_TIME}
    
    # 检查服务是否正常
    echo "[$(date)] 检查 vLLM 服务状态..."
    if ! curl -s http://localhost:${VLLM_PORT}/v1/models >/dev/null 2>&1; then
        echo "[$(date)] 警告：vLLM 服务可能未正常启动"
        echo "[$(date)] 检查 vLLM 进程状态..."
        if ps -p ${VLLM_PID} > /dev/null; then
            echo "[$(date)] vLLM 进程仍在运行，PID: ${VLLM_PID}"
        else
            echo "[$(date)] vLLM 进程已退出"
            echo "[$(date)] vLLM 退出日志："
            cat ${OUTPUT_DIR}/vllm.log
        fi
        echo "[$(date)] 尝试检查端口 ${VLLM_PORT} 状态..."
        netstat -tlnp | grep :${VLLM_PORT}
        return 1
    fi
    
    echo "[$(date)] vLLM 服务启动成功"
    return 0
}

# 函数：停止 vLLM 服务
stop_vllm() {
    echo "[$(date)] 停止 vLLM 服务..."
    
    # 使用更精确的进程匹配方式
    if pkill -f "vllm.entrypoints.openai.api_server.*--port ${VLLM_PORT}"; then
        echo "[$(date)] vLLM 服务已停止"
        sleep ${VLLM_SLEEP_TIME}
    else
        echo "[$(date)] 未找到运行中的 vLLM 服务"
    fi
}

# 函数：查找最新的 checkpoint
find_latest_checkpoint() {
    local checkpoint_dir=$1
    
    if [ ! -d "${checkpoint_dir}" ]; then
        echo ""
        return 1
    fi
    
    # 查找最新的 checkpoint
    local latest=$(ls -t ${checkpoint_dir}/checkpoint-* 2>/dev/null | head -1)
    
    if [ -z "${latest}" ]; then
        echo ""
        return 1
    fi
    
    # 验证 checkpoint 是否有效
    if [ ! -f "${latest}/config.json" ] || [ ! -f "${latest}/pytorch_model.bin" ]; then
        echo "[$(date)] 警告：checkpoint ${latest} 不完整，跳过"
        echo ""
        return 1
    fi
    
    echo "${latest}"
    return 0
}

# 主循环
while true; do
    echo "[$(date)] ==============================================="
    echo "[$(date)] 开始新一轮训练..."
    echo "[$(date)] 当前使用模型: ${CURRENT_MODEL_PATH}"
    
    # 启动 vLLM 服务
    if start_vllm "${CURRENT_MODEL_PATH}" "Qwen3-8B-SFT"; then
        echo "[$(date)] vLLM 服务启动成功，开始训练"
        
        # 执行训练
        PYTHONPATH=/home/jovyan2/query_rl/code:$PYTHONPATH python -m verl.trainer.main_ppo \
            --model_path ${CURRENT_MODEL_PATH} \
            --output_dir ${OUTPUT_DIR} \
            --checkpoint_interval ${CHECKPOINT_INTERVAL}
        
        TRAINING_EXIT_CODE=$?
        
        if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
            echo "[$(date)] 训练异常退出，退出码: ${TRAINING_EXIT_CODE}"
        else
            echo "[$(date)] 训练正常完成"
        fi
        
        # 停止 vLLM 服务
        stop_vllm
        
        # 检查是否有新的 checkpoint
        echo "[$(date)] 检查是否有新的 checkpoint..."
        LATEST_CKPT=$(find_latest_checkpoint "${OUTPUT_DIR}")
        
        if [ ! -z "${LATEST_CKPT}" ]; then
            echo "[$(date)] 发现新 checkpoint: ${LATEST_CKPT}"
            
            # 更新当前模型路径
            CURRENT_MODEL_PATH=${LATEST_CKPT}
            
            # 重启 vLLM 服务加载新模型
            if start_vllm "${CURRENT_MODEL_PATH}" "qwen-query-rewrite-v1"; then
                echo "[$(date)] 新模型已加载并启动成功"
            else
                echo "[$(date)] 新模型启动失败，继续使用之前的模型"
            fi
        else
            echo "[$(date)] 未发现新的 checkpoint，继续使用当前模型"
        fi
    else
        echo "[$(date)] vLLM 服务启动失败，跳过本轮训练"
    fi
    
    echo "[$(date)] 本轮训练完成，等待 ${TRAINING_INTERVAL} 秒后继续..."
    echo "[$(date)] ==============================================="
    sleep ${TRAINING_INTERVAL}
done