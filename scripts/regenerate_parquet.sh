#!/bin/bash
# 重新生成干净的 Parquet 文件
# 此脚本会自动添加 reward_model.ground_truth 字段

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔄 重新生成 Parquet 文件（包含 ground_truth 字段）..."
echo "================================================"

# 备份旧文件
if [ -f "data/sales_rag/train.parquet" ]; then
    backup_file="data/sales_rag/train.parquet.backup.$(date +%Y%m%d_%H%M%S)"
    mv data/sales_rag/train.parquet "$backup_file"
    echo "✓ 已备份 train.parquet -> $backup_file"
fi

if [ -f "data/sales_rag/val.parquet" ]; then
    backup_file="data/sales_rag/val.parquet.backup.$(date +%Y%m%d_%H%M%S)"
    mv data/sales_rag/val.parquet "$backup_file"
    echo "✓ 已备份 val.parquet -> $backup_file"
fi

echo ""
echo "📝 从 JSONL 生成新的 Parquet 文件..."
echo "================================================"

# 生成训练集
echo ""
echo "🔨 处理训练集..."
python /home/jovyan2/query_rl/query_rl_code/scripts/jsonl_to_parquet_converter.py \
    --input /home/jovyan2/query_rl/query_rl_code/data/sales_rag/train.jsonl \
    --output /home/jovyan2/query_rl/query_rl_code/data/sales_rag/train.parquet \
    --validate

# 生成验证集（如果存在）
if [ -f "data/sales_rag/val.jsonl" ]; then
    echo ""
    echo "🔨 处理验证集..."
    python /home/jovyan2/query_rl/query_rl_code/scripts/jsonl_to_parquet_converter.py \
        --input /home/jovyan2/query_rl/query_rl_code/data/sales_rag/val.jsonl \
        --output /home/jovyan2/query_rl/query_rl_code/data/sales_rag/val.parquet \
        --validate
else
    echo ""
    echo "⚠️  验证集不存在，跳过"
fi

echo ""
echo "================================================"
echo "✅ Parquet 文件重新生成完成！"
echo "================================================"
echo ""
echo "📊 生成的文件："
ls -lh data/sales_rag/*.parquet

echo ""
echo "💡 下一步："
echo "  1. 运行完整验证: python scripts/verify_grpo_hybrid.py"
echo "  2. 开始训练: bash scripts/run_grpo_hybrid.sh"
