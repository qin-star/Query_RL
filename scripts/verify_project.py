"""
项目完整性验证脚本
检查所有必要的文件和组件是否存在
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def main():
    """验证项目完整性"""
    print("\n" + "="*60)
    print("GRPO+GPT-5混合架构项目完整性检查")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    checks = []
    
    # 核心组件
    print("\n【核心组件】")
    checks.append(check_file_exists(
        os.path.join(base_dir, "verl_code/verl/workers/grpo_selector.py"),
        "GRPO选择器"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "verl_code/verl/workers/rag_adapter.py"),
        "RAG适配器"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "verl_code/verl/workers/gpt5_dual_model_rater.py"),
        "GPT-5评估器"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "verl_code/verl/workers/hybrid_reward_combiner.py"),
        "混合奖励合成器"
    ))
    
    # 基础设施
    print("\n【基础设施】")
    checks.append(check_file_exists(
        os.path.join(base_dir, "src/core/rag_chater.py"),
        "RAG调用类"
    ))
    
    # 脚本文件
    print("\n【脚本文件】")
    checks.append(check_file_exists(
        os.path.join(base_dir, "scripts/test_hybrid_components.py"),
        "组件测试脚本"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "scripts/train_hybrid_grpo_gpt5.py"),
        "训练集成脚本"
    ))
    
    # 文档
    print("\n【文档】")
    checks.append(check_file_exists(
        os.path.join(base_dir, "README_HYBRID_GRPO.md"),
        "使用指南"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "Plan_md/GRPO_RAG_Query_Rewrite_Architecture_v3.md"),
        "设计文档"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "Plan_md/代码修改完成总结.md"),
        "修改总结"
    ))
    
    # 检查已删除的旧文件
    print("\n【已清理的旧文件】")
    old_files = [
        "verl_code/verl/workers/hybrid_grpo_reward_calculator.py",
        "verl_code/verl/workers/hybrid_grpo_training_manager.py",
        "verl_code/verl/workers/grpo_group_generator.py",
        "verl_code/verl/workers/actor_model_processor.py",
        "verl_code/verl/workers/actor_model_processor_v2.py",
        "verl_code/verl/workers/reference_model_processor.py",
    ]
    
    for old_file in old_files:
        filepath = os.path.join(base_dir, old_file)
        if not os.path.exists(filepath):
            print(f"✓ 已删除: {old_file}")
        else:
            print(f"⚠ 仍存在: {old_file}")
    
    # 汇总
    print("\n" + "="*60)
    print("检查结果汇总")
    print("="*60)
    
    total = len(checks)
    passed = sum(checks)
    
    print(f"总计: {total} 项")
    print(f"通过: {passed} 项")
    print(f"失败: {total - passed} 项")
    
    if passed == total:
        print("\n🎉 项目完整性检查通过！")
        print("\n下一步:")
        print("1. 运行组件测试: python scripts/test_hybrid_components.py")
        print("2. 运行训练示例: python scripts/train_hybrid_grpo_gpt5.py")
        print("3. 查看使用指南: README_HYBRID_GRPO.md")
        return 0
    else:
        print("\n⚠ 项目不完整，请检查缺失的文件")
        return 1

if __name__ == "__main__":
    sys.exit(main())
