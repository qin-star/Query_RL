#!/bin/bash
# SalesRAG GRPO+GPT-5混合训练启动脚本
# 实现完整的流程图训练逻辑

set -e

echo "🚀 启动SalesRAG GRPO+GPT-5混合训练..."
echo "================================================"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
echo "📁 项目根目录: $PROJECT_ROOT"

# 激活虚拟环境（如果需要）
# source /path/to/venv/bin/activate

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl_code:$PROJECT_ROOT/src:$PYTHONPATH"
echo "🐍 PYTHONPATH: $PYTHONPATH"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=4,5,6,7  # 使用4张GPU

# 训练配置
CONFIG_FILE="$PROJECT_ROOT/verl_code/config/sales_rag_grpo_hybrid_config.yaml"
TRAIN_DATA="$PROJECT_ROOT/data/sales_rag/train.jsonl"
VAL_DATA="$PROJECT_ROOT/data/sales_rag/val.jsonl"

echo "📋 配置文件: $CONFIG_FILE"
echo "📊 训练数据: $TRAIN_DATA"
echo "📊 验证数据: $VAL_DATA"

# 检查文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据不存在: $TRAIN_DATA"
    exit 1
fi

echo "================================================"
echo "🔥 训练模式: GRPO + GPT-5混合"
echo "🎯 流程:"
echo "  1. 生成5个候选改写"
echo "  2. GRPO组内打分"
echo "  3. 选择最优候选"
echo "  4. 调用RAG接口（8B和32B）"
echo "  5. GPT-5双模型对比评分"
echo "  6. 奖励融合（GPT-5 85% + GRPO 15%）"
echo "  7. PPO参数更新"
echo "================================================"

# 启动训练
cd "$PROJECT_ROOT/verl_code"

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/verl_code/config" \
    --config-name="sales_rag_grpo_hybrid_config" \
    algorithm.select_best_from_group=true \
    algorithm.hybrid_grpo.enable=true \
    algorithm.hybrid_grpo.gpt5_weight=0.85 \
    algorithm.hybrid_grpo.grpo_weight=0.15 \
    actor_rollout_ref.rollout.n=5 \
    trainer.logger=[console] \
    trainer.project_name=sales_rag_grpo_hybrid \
    trainer.experiment_name="grpo_gpt5_$(date +%Y%m%d_%H%M%S)"

echo "✅ 训练完成！"
