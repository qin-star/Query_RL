#!/bin/bash
# 训练启动前检查脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 训练启动前检查..."
echo "================================================"

# 检查 Python 环境
echo "📦 检查 Python 环境..."
python --version
echo "✓ Python 可用"

# 检查必需的 Python 包
echo ""
echo "📦 检查必需的包..."
python -c "
import sys
required_packages = [
    'torch',
    'transformers',
    'datasets',
    'pyarrow',
    'omegaconf',
    'hydra',
    'ray',
    'verl'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        print(f'  ✗ {pkg} (缺失)')
        missing.append(pkg)

if missing:
    print(f'\n❌ 缺少包: {missing}')
    sys.exit(1)
else:
    print('\n✓ 所有必需的包都已安装')
"

# 检查配置文件
echo ""
echo "📋 检查配置文件..."
CONFIG_FILE="verl_code/config/sales_rag_grpo_hybrid_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "✓ 配置文件存在: $CONFIG_FILE"

# 检查数据文件
echo ""
echo "📊 检查数据文件..."
python -c "
import os
import sys

os.chdir('verl_code')

from omegaconf import OmegaConf

cfg = OmegaConf.load('config/sales_rag_grpo_hybrid_config.yaml')

train_files = cfg.data.train_files
val_files = cfg.data.val_files

print(f'训练文件: {train_files}')
print(f'验证文件: {val_files}')

all_ok = True
for files, name in [(train_files, '训练'), (val_files, '验证')]:
    for f in files:
        abs_path = os.path.abspath(f)
        if os.path.exists(abs_path):
            size = os.path.getsize(abs_path) / 1024 / 1024
            print(f'  ✓ {name}文件存在: {abs_path} ({size:.2f} MB)')
        else:
            print(f'  ✗ {name}文件不存在: {abs_path}')
            all_ok = False

if not all_ok:
    print('\n❌ 部分数据文件缺失')
    sys.exit(1)
else:
    print('\n✓ 所有数据文件都存在')
"

# 检查模型路径
echo ""
echo "🤖 检查模型路径..."
python -c "
import os
import sys

os.chdir('verl_code')

from omegaconf import OmegaConf

cfg = OmegaConf.load('config/sales_rag_grpo_hybrid_config.yaml')

model_path = cfg.actor_rollout_ref.model.path
print(f'模型路径: {model_path}')

if os.path.exists(model_path):
    print(f'  ✓ 模型目录存在')
    
    # 检查必需的文件
    required_files = ['config.json', 'tokenizer_config.json']
    for f in required_files:
        fpath = os.path.join(model_path, f)
        if os.path.exists(fpath):
            print(f'  ✓ {f} 存在')
        else:
            print(f'  ⚠ {f} 不存在（可能不影响）')
else:
    print(f'  ✗ 模型目录不存在: {model_path}')
    sys.exit(1)
"

# 检查自定义模块
echo ""
echo "🔧 检查自定义模块..."
python -c "
import sys
import os

os.chdir('verl_code')
sys.path.insert(0, os.getcwd())

# 检查 HybridTrainingManager
try:
    from verl.workers.hybrid_grpo_training_manager import HybridTrainingManager
    print('  ✓ HybridTrainingManager 可导入')
except ImportError as e:
    print(f'  ✗ HybridTrainingManager 导入失败: {e}')
    sys.exit(1)

# 检查自定义奖励函数
try:
    from verl.utils.reward_score.sales_rag import compute_score
    print('  ✓ sales_rag.compute_score 可导入')
except ImportError as e:
    print(f'  ✗ sales_rag.compute_score 导入失败: {e}')
    sys.exit(1)

print('\n✓ 所有自定义模块都可用')
"

# 检查 GPU
echo ""
echo "🎮 检查 GPU..."
python -c "
import torch

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f'  ✓ 检测到 {n_gpus} 个 GPU')
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'    GPU {i}: {name} ({mem:.1f} GB)')
else:
    print('  ⚠ 未检测到 GPU，将使用 CPU')
"

# 检查 Parquet 文件完整性
echo ""
echo "📦 检查 Parquet 文件完整性..."
python scripts/diagnose_parquet_issue.py 2>&1 | grep -E "(✓|✗|❌)" || true

echo ""
echo "================================================"
echo "✅ 所有检查通过！可以开始训练"
echo ""
echo "运行训练："
echo "  bash scripts/run_grpo_hybrid.sh"
echo "================================================"
