#!/bin/bash
# GPU 内存检查脚本

echo "🎮 GPU 内存状态检查"
echo "================================================"

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 不可用"
    exit 1
fi

# 显示 GPU 信息
echo ""
echo "📊 GPU 总览："
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv

# 显示详细的进程信息
echo ""
echo "🔍 GPU 进程详情："
nvidia-smi

# 计算可用内存百分比
echo ""
echo "📈 内存使用分析："
python3 -c "
import subprocess
import re

result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], 
                       capture_output=True, text=True)

for line in result.stdout.strip().split('\n'):
    parts = [x.strip() for x in line.split(',')]
    if len(parts) >= 4:
        idx, total, free, used = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        free_pct = (free / total) * 100
        used_pct = (used / total) * 100
        
        print(f'GPU {idx}:')
        print(f'  总内存: {total/1024:.2f} GB')
        print(f'  已用: {used/1024:.2f} GB ({used_pct:.1f}%)')
        print(f'  可用: {free/1024:.2f} GB ({free_pct:.1f}%)')
        
        # 计算不同 utilization 设置需要的内存
        for util in [0.4, 0.45, 0.5, 0.55, 0.6]:
            required = total * util / 1024
            if free >= total * util:
                status = '✓'
            else:
                status = '✗'
            print(f'  {status} gpu_memory_utilization={util}: 需要 {required:.2f} GB')
        print()
"

# 建议
echo "================================================"
echo "💡 建议："
echo ""
echo "1. 如果有其他进程占用 GPU，考虑："
echo "   - 杀掉不需要的进程"
echo "   - 使用其他空闲的 GPU"
echo "   - 降低 gpu_memory_utilization"
echo ""
echo "2. 当前配置已设置为 0.45，如果还不够："
echo "   - 进一步降低到 0.4 或 0.35"
echo "   - 减小 max_num_batched_tokens"
echo "   - 减小 max_num_seqs"
echo ""
echo "3. 清理 GPU 内存："
echo "   python -c 'import torch; torch.cuda.empty_cache()'"
echo "================================================"
