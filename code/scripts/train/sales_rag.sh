#!/bin/bash
# Sales-RAG × DeepRetrieval 强化学习训练脚本
# 专门针对销售场景的QueryRewrite训练

set -e

# 配置参数
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 模型和数据配置
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
CONFIG_FILE="config/sales_rag_rl_config.yaml"
DATA_DIR="data/deepretrieval_training"
OUTPUT_DIR="outputs/sales_rag_rl_$(date +%Y%m%d_%H%M%S)"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=================================================="
echo "Sales-RAG DeepRetrieval RL Training"
echo "=================================================="
echo "Base Model: ${BASE_MODEL}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "=================================================="

# 检查必要文件
echo "检查训练环境..."

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "❌ 数据目录不存在: ${DATA_DIR}"
    echo "请先运行数据收集脚本"
    exit 1
fi

# 检查训练数据
TRAIN_FILE="${DATA_DIR}/rl_train_latest.jsonl"
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "❌ 训练数据文件不存在: ${TRAIN_FILE}"
    echo "请先运行用户反馈收集，生成训练数据"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "${TRAIN_FILE}")
echo "✅ 找到训练样本: ${SAMPLE_COUNT} 条"

if [ ${SAMPLE_COUNT} -lt 30 ]; then
    echo "⚠️  训练样本较少 (${SAMPLE_COUNT} < 30)，建议收集更多数据"
    read -p "是否继续训练? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "训练已取消"
        exit 0
    fi
fi

# 检查GPU
echo "检查GPU环境..."
nvidia-smi || {
    echo "❌ 未检测到GPU或nvidia-smi不可用"
    exit 1
}

# 检查Python环境
echo "检查Python环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "❌ PyTorch环境异常"
    exit 1
}

python -c "import verl; print('VERL环境正常')" || {
    echo "❌ VERL环境异常，请检查安装"
    exit 1
}

echo "✅ 环境检查通过"

# 备份配置文件到输出目录
cp ${CONFIG_FILE} ${OUTPUT_DIR}/training_config.yaml

# 生成训练参数
LEARNING_RATE="1e-6"
BATCH_SIZE="4"
MAX_EPOCHS="1"
CLIP_RANGE="0.1"

# 根据数据量调整参数
if [ ${SAMPLE_COUNT} -gt 100 ]; then
    BATCH_SIZE="8"
    MAX_EPOCHS="2"
elif [ ${SAMPLE_COUNT} -lt 20 ]; then
    BATCH_SIZE="2"
    LEARNING_RATE="5e-7"  # 更小的学习率
fi

echo "训练参数:"
echo "  - 学习率: ${LEARNING_RATE}"
echo "  - 批次大小: ${BATCH_SIZE}" 
echo "  - 训练轮数: ${MAX_EPOCHS}"
echo "  - Clip范围: ${CLIP_RANGE}"

# 启动训练
echo ""
echo "开始训练..."
echo "输出将保存到: ${OUTPUT_DIR}/training.log"

LOG_FILE="${OUTPUT_DIR}/training.log"

python -m verl.trainer.main_ppo \
    --config ${CONFIG_FILE} \
    --model.path ${BASE_MODEL} \
    --data.train_path ${TRAIN_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --logging.run_name "sales_rag_rl_$(date +%Y%m%d_%H%M%S)" \
    --ppo.learning_rate ${LEARNING_RATE} \
    --ppo.batch_size ${BATCH_SIZE} \
    --ppo.max_epochs ${MAX_EPOCHS} \
    --ppo.clip_range ${CLIP_RANGE} \
    --reward.type "sales_rag" \
    --distributed.num_gpus 1 \
    --distributed.gradient_checkpointing true \
    --logging.save_steps 50 \
    --evaluation.eval_steps 20 \
    2>&1 | tee ${LOG_FILE}

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "✅ 训练完成成功!"
    
    # 检查checkpoint
    CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-final"
    if [ -d "${CHECKPOINT_DIR}" ]; then
        echo "✅ 找到最终checkpoint: ${CHECKPOINT_DIR}"
        
        # 验证模型文件
        if [ -f "${CHECKPOINT_DIR}/pytorch_model.bin" ] || [ -f "${CHECKPOINT_DIR}/model.safetensors" ]; then
            echo "✅ 模型文件验证通过"
            
            # 保存模型信息
            echo "创建模型信息文件..."
            cat > ${OUTPUT_DIR}/model_info.json << EOF
{
    "model_name": "sales_rag_deepretrieval",
    "base_model": "${BASE_MODEL}",
    "training_date": "$(date -Iseconds)",
    "sample_count": ${SAMPLE_COUNT},
    "training_params": {
        "learning_rate": "${LEARNING_RATE}",
        "batch_size": ${BATCH_SIZE},
        "max_epochs": ${MAX_EPOCHS},
        "clip_range": "${CLIP_RANGE}"
    },
    "checkpoint_path": "${CHECKPOINT_DIR}",
    "config_file": "${CONFIG_FILE}",
    "training_log": "${LOG_FILE}"
}
EOF
            
            echo "✅ 模型训练完成，可以部署使用"
            echo ""
            echo "下一步操作:"
            echo "1. 部署模型: bash scripts/deploy/deploy_model.sh ${CHECKPOINT_DIR}"
            echo "2. 测试模型: python scripts/eval/test_sales_rag.py --model_path ${CHECKPOINT_DIR}"
            echo "3. 启用A/B测试: 在sales-rag配置中设置 DEEPRETRIEVAL_ENABLED=true"
        else
            echo "❌ 模型文件验证失败"
            exit 1
        fi
    else
        echo "❌ 未找到最终checkpoint"
        exit 1
    fi
else
    echo "❌ 训练失败，退出码: ${TRAIN_EXIT_CODE}"
    echo "请查看日志: ${LOG_FILE}"
    exit 1
fi

echo ""
echo "=================================================="
echo "训练日志: ${LOG_FILE}"
echo "模型输出: ${OUTPUT_DIR}"
echo "=================================================="




