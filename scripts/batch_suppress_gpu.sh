#!/bin/bash
# 批量GPU屏蔽脚本
# 支持多种批量操作模式：单GPU、多GPU、范围GPU、配置文件等

set -e  # 遇到错误立即退出

# ------------------- 配置变量 -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPRESS_GPU_CMD="/home/jovyan/chenbingwei/gpu_monitor/suppress_gpu_detect"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/batch_suppress_gpu_$(date +%Y%m%d_%H%M%S).log"
CONFIG_FILE="${SCRIPT_DIR}/batch_gpu_config.yaml"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# GPU范围配置
GPU_RANGE_START=0
GPU_RANGE_END=7
DEFAULT_MINUTES=10

# ------------------- 工具函数 -------------------

# 日志函数
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
}

# 错误处理
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# 检查suppress_gpu_detect命令是否存在
check_command() {
    if [[ ! -f "$SUPPRESS_GPU_CMD" ]]; then
        error_exit "suppress_gpu_detect 命令不存在: $SUPPRESS_GPU_CMD"
    fi
    if [[ ! -x "$SUPPRESS_GPU_CMD" ]]; then
        error_exit "suppress_gpu_detect 命令不可执行: $SUPPRESS_GPU_CMD"
    fi
}

# 验证GPU ID是否有效
validate_gpu_id() {
    local gpu_id="$1"
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        error_exit "无效的GPU ID: $gpu_id (必须是数字)"
    fi
    if (( gpu_id < GPU_RANGE_START || gpu_id > GPU_RANGE_END )); then
        error_exit "GPU ID $gpu_id 超出范围 (${GPU_RANGE_START}-${GPU_RANGE_END})"
    fi
}

# 验证分钟数是否有效
validate_minutes() {
    local minutes="$1"
    if ! [[ "$minutes" =~ ^[0-9]+$ ]]; then
        error_exit "无效的分钟数: $minutes (必须是正整数)"
    fi
    if (( minutes < 1 || minutes > 60 )); then
        log "WARNING" "分钟数 $minutes 超出建议范围 (1-60)，但仍然执行"
    fi
}

# ------------------- GPU操作函数 -------------------

# 屏蔽单个GPU
suppress_single_gpu() {
    local gpu_id="$1"
    local minutes="$2"
    
    validate_gpu_id "$gpu_id"
    validate_minutes "$minutes"
    
    log "INFO" "正在屏蔽GPU $gpu_id，时长 $minutes 分钟..."
    if "$SUPPRESS_GPU_CMD" "$gpu_id" "$minutes"; then
        log "SUCCESS" "GPU $gpu_id 屏蔽成功"
    else
        log "ERROR" "GPU $gpu_id 屏蔽失败"
        return 1
    fi
}

# 屏蔽多个GPU
suppress_multiple_gpus() {
    local gpu_list="$1"
    local minutes="$2"
    
    validate_minutes "$minutes"
    
    log "INFO" "正在批量屏蔽GPU列表: $gpu_list，时长 $minutes 分钟..."
    if "$SUPPRESS_GPU_CMD" "$gpu_list" "$minutes"; then
        log "SUCCESS" "GPU列表 $gpu_list 屏蔽成功"
    else
        log "ERROR" "GPU列表 $gpu_list 屏蔽失败"
        return 1
    fi
}

# 屏蔽GPU范围
suppress_gpu_range() {
    local start_gpu="$1"
    local end_gpu="$2"
    local minutes="$3"
    
    validate_gpu_id "$start_gpu"
    validate_gpu_id "$end_gpu"
    validate_minutes "$minutes"
    
    if (( start_gpu > end_gpu )); then
        local temp=$start_gpu
        start_gpu=$end_gpu
        end_gpu=$temp
    fi
    
    local gpu_range="${start_gpu}-${end_gpu}"
    log "INFO" "正在屏蔽GPU范围: $gpu_range，时长 $minutes 分钟..."
    if "$SUPPRESS_GPU_CMD" "$gpu_range" "$minutes"; then
        log "SUCCESS" "GPU范围 $gpu_range 屏蔽成功"
    else
        log "ERROR" "GPU范围 $gpu_range 屏蔽失败"
        return 1
    fi
}

# 解除GPU屏蔽
clear_gpu_suppression() {
    local gpu_spec="$1"
    
    log "INFO" "正在解除GPU屏蔽: $gpu_spec..."
    if "$SUPPRESS_GPU_CMD" --clear "$gpu_spec"; then
        log "SUCCESS" "GPU $gpu_spec 屏蔽解除成功"
    else
        log "ERROR" "GPU $gpu_spec 屏蔽解除失败"
        return 1
    fi
}

# 显示当前屏蔽状态
show_suppression_status() {
    log "INFO" "正在查询GPU屏蔽状态..."
    "$SUPPRESS_GPU_CMD" --list
}

# ------------------- 配置文件处理 -------------------

# 创建示例配置文件
create_sample_config() {
    cat > "$CONFIG_FILE" << EOF
# 批量GPU屏蔽配置文件
# 支持多种操作模式

# 模式1: 批量操作列表
batch_operations:
  - operation: suppress
    gpu_list: "0,1,2,3"
    minutes: 10
    description: "屏蔽前4个GPU 10分钟"
  
  - operation: suppress
    gpu_range: "4-7"
    minutes: 15
    description: "屏蔽后4个GPU 15分钟"
  
  - operation: clear
    gpu_list: "0,1"
    description: "解除GPU 0,1的屏蔽"

# 模式2: 分组配置
gpu_groups:
  group_a:
    gpus: "0,1,2,3"
    default_minutes: 10
    
  group_b:
    gpus: "4,5,6,7"
    default_minutes: 15

# 模式3: 预设方案
presets:
  training:
    suppress_gpus: "0-3"
    minutes: 30
    
  inference:
    suppress_gpus: "4-7"
    minutes: 60
    
  maintenance:
    suppress_gpus: "0-7"
    minutes: 120
EOF

    log "INFO" "示例配置文件已创建: $CONFIG_FILE"
}

# 从配置文件执行批量操作
execute_from_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        error_exit "配置文件不存在: $config_file"
    fi
    
    log "INFO" "正在从配置文件执行批量操作: $config_file"
    
    # 检查是否有yaml解析工具
    if ! command -v yq &> /dev/null && ! command -v python3 &> /dev/null; then
        error_exit "需要 yq 或 python3 来解析配置文件"
    fi
    
    # 使用python3解析yaml（如果没有yq）
    if command -v python3 &> /dev/null; then
        python3 << 'EOF'
import yaml
import sys
import os

config_file = sys.argv[1] if len(sys.argv) > 1 else 'batch_gpu_config.yaml'

try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'batch_operations' in config:
        print("batch_operations:")
        for i, op in enumerate(config['batch_operations']):
            print(f"  op_{i}:")
            for key, value in op.items():
                print(f"    {key}: {value}")
    
except Exception as e:
    print(f"Error parsing config file: {e}")
    sys.exit(1)
EOF
    fi
    
    log "WARNING" "配置文件解析功能需要进一步实现，目前仅显示配置结构"
}

# ------------------- 主函数 -------------------

print_help() {
    local script_name=$(basename "$0")
    cat << EOF
用法:
  $script_name <command> [options]

命令:
  single <gpu_id> <minutes>           屏蔽单个GPU
  multi <gpu_list> <minutes>          屏蔽多个GPU (如: "0,1,2,4")
  range <start_gpu> <end_gpu> <minutes> 屏蔽GPU范围
  clear <gpu_spec>                   解除GPU屏蔽
  status                             显示当前屏蔽状态
  config <config_file>               从配置文件执行批量操作
  create-config                      创建示例配置文件
  help                               显示此帮助信息

示例:
  $script_name single 4 10              # 屏蔽GPU 4，10分钟
  $script_name multi "4,5,6,7" 10      # 屏蔽GPU 0,1,2,4，10分钟
  $script_name range 0 3 15            # 屏蔽GPU 0-3，15分钟
  $script_name clear "0,1,2,4"        # 解除GPU 0,1,2,4的屏蔽
  $script_name status                  # 查看当前屏蔽状态
  $script_name config batch_gpu_config.yaml  # 从配置文件执行
  $script_name create-config           # 创建示例配置文件

配置文件格式:
  支持YAML格式，包含批量操作列表、分组配置和预设方案
EOF
}

# ------------------- 主逻辑 -------------------

main() {
    check_command
    
    if [[ $# -eq 0 ]]; then
        print_help
        exit 0
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        single)
            if [[ $# -ne 2 ]]; then
                error_exit "用法: $0 single <gpu_id> <minutes>"
            fi
            suppress_single_gpu "$1" "$2"
            ;;
        multi)
            if [[ $# -ne 2 ]]; then
                error_exit "用法: $0 multi <gpu_list> <minutes>"
            fi
            suppress_multiple_gpus "$1" "$2"
            ;;
        range)
            if [[ $# -ne 3 ]]; then
                error_exit "用法: $0 range <start_gpu> <end_gpu> <minutes>"
            fi
            suppress_gpu_range "$1" "$2" "$3"
            ;;
        clear)
            if [[ $# -ne 1 ]]; then
                error_exit "用法: $0 clear <gpu_spec>"
            fi
            clear_gpu_suppression "$1"
            ;;
        status)
            show_suppression_status
            ;;
        config)
            if [[ $# -eq 0 ]]; then
                execute_from_config "$CONFIG_FILE"
            else
                execute_from_config "$1"
            fi
            ;;
        create-config)
            create_sample_config
            ;;
        help|--help|-h)
            print_help
            ;;
        *)
            error_exit "未知命令: $command. 使用 '$0 help' 查看帮助"
            ;;
    esac
}

# 执行主函数
main "$@"