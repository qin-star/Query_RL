"""
GRPO训练环境验证脚本
检查所有必要的文件和配置是否正确
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    """验证GRPO训练环境"""
    print("\n" + "="*60)
    print("GRPO训练环境验证")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    checks = []
    
    # 核心文件
    print("\n【核心文件】")
    checks.append(check_file_exists(
        os.path.join(base_dir, "verl_code/verl/trainer/main_ppo.py"),
        "verl主训练入口"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "scripts/run_grpo_standard.sh"),
        "GRPO训练脚本"
    ))
    
    # 数据目录
    print("\n【数据目录】")
    data_dir = os.path.join(base_dir, "data/sales_rag")
    if check_directory_exists(data_dir, "数据目录"):
        checks.append(check_file_exists(
            os.path.join(data_dir, "train.parquet"),
            "训练数据"
        ))
        checks.append(check_file_exists(
            os.path.join(data_dir, "val.parquet"),
            "验证数据"
        ))
    else:
        print(f"⚠ 数据目录不存在，请创建: {data_dir}")
        checks.append(False)
        checks.append(False)
    
    # 模型路径（仅检查是否配置）
    print("\n【模型配置】")
    print("ℹ 请确认模型路径已在启动脚本中正确配置")
    print("  默认路径: /home/jovyan2/query_rl/model/Qwen3-8B")
    
    # 输出目录
    print("\n【输出目录】")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"ℹ 创建检查点目录: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
    check_directory_exists(checkpoint_dir, "检查点目录")
    
    # 文档
    print("\n【文档】")
    checks.append(check_file_exists(
        os.path.join(base_dir, "README_GRPO_TRAINING.md"),
        "训练指南"
    ))
    
    # 检查已删除的混合架构文件
    print("\n【已清理的混合架构文件】")
    removed_files = [
        "verl_code/verl/workers/grpo_selector.py",
        "verl_code/verl/workers/rag_adapter.py",
        "verl_code/verl/workers/hybrid_reward_combiner.py",
        "scripts/train_hybrid_grpo_gpt5.py",
        "scripts/test_hybrid_components.py",
        "README_HYBRID_GRPO.md",
        "INTEGRATION_COMPLETE.md",
        "VERL_INTEGRATION_STATUS.md",
        "FINAL_SUMMARY.md",
    ]
    
    all_cleaned = True
    for removed_file in removed_files:
        filepath = os.path.join(base_dir, removed_file)
        if not os.path.exists(filepath):
            print(f"✓ 已删除: {removed_file}")
        else:
            print(f"⚠ 仍存在: {removed_file}")
            all_cleaned = False
    
    # 汇总
    print("\n" + "="*60)
    print("检查结果汇总")
    print("="*60)
    
    total = len(checks)
    passed = sum(checks)
    
    print(f"核心文件检查: {passed}/{total} 通过")
    print(f"混合架构清理: {'✓ 完成' if all_cleaned else '⚠ 未完成'}")
    
    if passed == total and all_cleaned:
        print("\n🎉 环境验证通过！可以开始训练")
        print("\n下一步:")
        print("1. 确认数据已准备: data/sales_rag/train.parquet")
        print("2. 确认模型路径: 编辑 scripts/run_grpo_standard.sh")
        print("3. 启动训练: bash scripts/run_grpo_standard.sh")
        print("4. 查看文档: README_GRPO_TRAINING.md")
        return 0
    else:
        print("\n⚠ 环境验证未通过，请检查上述问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
