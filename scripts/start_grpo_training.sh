#!/bin/bash

# SalesRAG GRPO双模型训练启动脚本 v2.0
# 用法: bash scripts/start_grpo_training.sh [mode]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查环境配置..."
    
    # 检查Python版本
    if ! python3 --version &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi 未找到，请确保CUDA环境正确"
    fi
    
    # 检查必要目录
    local required_dirs=("data/sales_rag" "model/Qwen3-8B" "outputs" "cache")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_warn "目录 $dir 不存在，正在创建..."
            mkdir -p "$dir"
        fi
    done
    
    log_info "环境检查完成"
}

# 数据预处理
preprocess_data() {
    log_info "开始数据预处理..."
    
    cd /home/jovyan2/query_rl
    
    # 检查Excel文件是否存在
    if [ ! -f "data/sales_rag/RL_tranning_data/橙啦-query_RL_训练集.xlsx" ]; then
        log_error "Excel训练数据文件不存在: data/sales_rag/RL_tranning_data/橙啦-query_RL_训练集.xlsx"
        exit 1
    fi
    
    # 运行数据预处理
    log_blue "运行Excel数据预处理..."
    python scripts/process_excel_data.py \
        --input data/sales_rag/RL_tranning_data/橙啦-query_RL_训练集.xlsx \
        --output data/sales_rag/train_processed.jsonl \
        --template src/prompts/query_rewrite_prompt.txt
    
    if [ $? -ne 0 ]; then
        log_error "数据预处理失败"
        exit 1
    fi
    
    # 转换数据格式
    log_blue "转换数据格式为parquet..."
    python -c "
import pandas as pd
import json
import os

# 检查处理后的数据文件
processed_file = 'data/sales_rag/train_processed.jsonl'
if not os.path.exists(processed_file):
    print('ERROR: 处理后的数据文件不存在')
    exit(1)

# 读取处理后的数据
with open(processed_file, 'r') as f:
    data = [json.loads(line) for line in f]

if len(data) == 0:
    print('ERROR: 数据为空')
    exit(1)

# 转换为DataFrame
df = pd.DataFrame(data)

# 保存为parquet
df.to_parquet('data/sales_rag/train_dual_model.parquet', index=False)
print(f'成功转换训练数据: {len(df)} 条样本')

# 处理验证集（如果存在）
val_file = 'data/sales_rag/val_processed.jsonl'
if os.path.exists(val_file):
    with open(val_file, 'r') as f:
        val_data = [json.loads(line) for line in f]
    
    if len(val_data) > 0:
        val_df = pd.DataFrame(val_data)
        val_df.to_parquet('data/sales_rag/val_dual_model.parquet', index=False)
        print(f'成功转换验证数据: {len(val_df)} 条样本')
    else:
        print('验证数据为空，跳过')
else:
    print('验证数据文件不存在，跳过')
"
    
    if [ $? -ne 0 ]; then
        log_error "数据格式转换失败"
        exit 1
    fi
    
    log_info "数据预处理完成"
}

# 启动训练
start_training() {
    local mode=$1
    log_info "启动GRPO训练，模式: $mode"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=4
    export PYTHONPATH="${PYTHONPATH}:/home/jovyan2/query_rl"
    
    # 确保在正确的工作目录
    cd /home/jovyan2/query_rl
    
    log_blue "设置环境变量..."
    log_info "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    log_info "PYTHONPATH: $PYTHONPATH"
    
    # 根据模式选择启动方式
    case $mode in
        "single")
            log_info "启动单机单卡训练..."
            cd /home/jovyan2/query_rl/verl_code && \
            /home/jovyan/work/tanzichang/miniconda/envs/Query_RL_310/bin/python -m torch.distributed.run \
                --nproc_per_node=1 \
                --nnodes=1 \
                --node_rank=0 \
                --master_addr=localhost \
                --master_port=29500 \
                verl/trainer/main_ppo.py \
                --config-path=config \
                --config-name=sales_rag_grpo_dual_model_config \
                trainer.experiment_name=sales_rag_grpo_dual_v1 \
                trainer.project_name=sales_rag_grpo_dual
            ;;
        "ray")
            log_info "启动Ray分布式训练..."
            
            # 检查Ray是否安装
            if ! command -v ray &> /dev/null; then
                log_error "Ray未安装，请先安装Ray: pip install ray"
                exit 1
            fi
            
            # 启动Ray集群
            log_blue "启动Ray集群..."
            ray start --head --num-gpus=2
            
            # 运行训练
            cd /home/jovyan2/query_rl/verl_code && \
            /home/jovyan/work/tanzichang/miniconda/envs/Query_RL_310/bin/python verl/trainer/main_ppo.py \
                --config-path=config \
                --config-name=sales_rag_grpo_dual_model_config \
                trainer.experiment_name=sales_rag_grpo_dual_v1 \
                trainer.project_name=sales_rag_grpo_dual \
                trainer.use_ray=true
                
            # 训练结束后停止Ray
            ray stop
            ;;
        *)
            log_error "不支持的训练模式: $mode"
            echo "支持的训练模式: single, ray"
            exit 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        log_info "训练启动成功"
    else
        log_error "训练启动失败"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "SalesRAG GRPO双模型训练启动脚本"
    echo ""
    echo "用法: $0 [mode] [options]"
    echo ""
    echo "训练模式:"
    echo "  single    单机单卡训练 (默认)"
    echo "  ray       Ray分布式训练"
    echo ""
    echo "选项:"
    echo "  --help     显示此帮助信息"
    echo "  --skip-data 跳过数据预处理"
    echo "  --debug     启用调试模式"
    echo ""
    echo "示例:"
    echo "  $0 single                 # 单机单卡训练"
    echo "  $0 ray                    # Ray分布式训练"
    echo "  $0 single --skip-data     # 跳过数据预处理"
    echo "  $0 ray --debug           # 启用调试模式"
}

# 主函数
main() {
    local mode="single"
    local skip_data=false
    local debug_mode=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            single|ray)
                mode=$1
                shift
                ;;
            --skip-data)
                skip_data=true
                shift
                ;;
            --debug)
                debug_mode=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 调试模式设置
    if [ "$debug_mode" = true ]; then
        set -x
        export DEBUG=1
        log_info "启用调试模式"
    fi
    
    log_blue "=========================================="
    log_info "SalesRAG GRPO双模型训练 v2.0"
    log_info "训练模式: $mode"
    log_info "跳过数据预处理: $skip_data"
    log_info "调试模式: $debug_mode"
    log_blue "=========================================="
    
    # 检查环境
    check_environment
    
    # 数据预处理
    if [ "$skip_data" = false ]; then
        preprocess_data
    else
        log_info "跳过数据预处理"
    fi
    
    # 启动训练
    start_training $mode
    
    log_info "脚本执行完成"
}

# 运行主函数
main "$@"