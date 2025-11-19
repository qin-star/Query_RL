#!/usr/bin/env python3
"""
配置文件验证脚本
检查常见的配置错误
"""

import sys
from pathlib import Path
from omegaconf import OmegaConf

def validate_batch_size_config(cfg, path_prefix=""):
    """验证批次大小配置"""
    errors = []
    
    # 检查 actor 配置
    if hasattr(cfg, 'actor_rollout_ref') and hasattr(cfg.actor_rollout_ref, 'actor'):
        actor = cfg.actor_rollout_ref.actor
        if actor.get('ppo_micro_batch_size') is not None and \
           actor.get('ppo_micro_batch_size_per_gpu') is not None:
            errors.append(
                f"{path_prefix}actor_rollout_ref.actor: "
                "不能同时设置 ppo_micro_batch_size 和 ppo_micro_batch_size_per_gpu"
            )
    
    # 检查 critic 配置
    if hasattr(cfg, 'critic'):
        critic = cfg.critic
        if critic.get('ppo_micro_batch_size') is not None and \
           critic.get('ppo_micro_batch_size_per_gpu') is not None:
            errors.append(
                f"{path_prefix}critic: "
                "不能同时设置 ppo_micro_batch_size 和 ppo_micro_batch_size_per_gpu"
            )
    
    return errors

def validate_file_paths(cfg):
    """验证文件路径"""
    errors = []
    warnings = []
    
    # 检查数据文件
    if hasattr(cfg, 'data'):
        for file_type in ['train_files', 'val_files']:
            if hasattr(cfg.data, file_type):
                files = getattr(cfg.data, file_type)
                for f in files:
                    # 相对路径会在运行时解析，这里只检查格式
                    if not f.endswith(('.parquet', '.jsonl', '.json')):
                        warnings.append(
                            f"data.{file_type}: 文件 {f} 格式可能不支持"
                        )
    
    # 检查模型路径（只警告，不报错）
    if hasattr(cfg, 'actor_rollout_ref') and hasattr(cfg.actor_rollout_ref, 'model'):
        model_path = cfg.actor_rollout_ref.model.get('path')
        if model_path and not Path(model_path).exists():
            warnings.append(
                f"actor_rollout_ref.model.path: 路径 {model_path} 不存在（可能在服务器上）"
            )
    
    return errors, warnings

def validate_algorithm_config(cfg):
    """验证算法配置"""
    errors = []
    warnings = []
    
    if hasattr(cfg, 'algorithm'):
        algo = cfg.algorithm
        
        # 检查 hybrid_grpo 配置
        if hasattr(algo, 'hybrid_grpo') and algo.hybrid_grpo.get('enable'):
            hg = algo.hybrid_grpo
            
            # 检查权重和
            gpt5_weight = hg.get('gpt5_weight', 0)
            grpo_weight = hg.get('grpo_weight', 0)
            
            if abs(gpt5_weight + grpo_weight - 1.0) > 0.01:
                warnings.append(
                    f"algorithm.hybrid_grpo: "
                    f"权重和不为1 (gpt5={gpt5_weight}, grpo={grpo_weight})"
                )
            
            # 检查组大小
            group_size = hg.get('group_size', 5)
            if hasattr(cfg, 'actor_rollout_ref') and \
               hasattr(cfg.actor_rollout_ref, 'rollout'):
                rollout_n = cfg.actor_rollout_ref.rollout.get('n', 5)
                if group_size != rollout_n:
                    warnings.append(
                        f"algorithm.hybrid_grpo.group_size ({group_size}) != "
                        f"actor_rollout_ref.rollout.n ({rollout_n})"
                    )
    
    return errors, warnings

def validate_custom_modules(cfg):
    """验证自定义模块配置"""
    errors = []
    warnings = []
    
    # 检查自定义奖励函数
    if hasattr(cfg, 'custom_reward_function'):
        crf = cfg.custom_reward_function
        if crf.get('path'):
            # 检查路径格式
            path = crf.get('path')
            if not path.endswith('.py'):
                errors.append(
                    f"custom_reward_function.path: "
                    f"路径 {path} 不是 Python 文件"
                )
    
    return errors, warnings

def main():
    # 加载配置
    config_path = Path(__file__).parent.parent / "verl_code/config/sales_rag_grpo_hybrid_config.yaml"
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    print(f"🔍 验证配置文件: {config_path}")
    print("=" * 60)
    
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return 1
    
    all_errors = []
    all_warnings = []
    
    # 运行各项检查
    print("\n📋 检查批次大小配置...")
    errors = validate_batch_size_config(cfg)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("  ✓ 批次大小配置正确")
    
    print("\n📁 检查文件路径...")
    errors, warnings = validate_file_paths(cfg)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    if not errors and not warnings:
        print("  ✓ 文件路径配置正确")
    
    print("\n🎯 检查算法配置...")
    errors, warnings = validate_algorithm_config(cfg)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    if not errors and not warnings:
        print("  ✓ 算法配置正确")
    
    print("\n🔧 检查自定义模块...")
    errors, warnings = validate_custom_modules(cfg)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    if not errors and not warnings:
        print("  ✓ 自定义模块配置正确")
    
    # 总结
    print("\n" + "=" * 60)
    if all_errors:
        print(f"❌ 发现 {len(all_errors)} 个错误")
        print("\n请修复以上错误后再启动训练")
        return 1
    elif all_warnings:
        print(f"⚠ 发现 {len(all_warnings)} 个警告")
        print("\n警告不会阻止训练，但建议检查")
        return 0
    else:
        print("✅ 配置文件验证通过！")
        return 0

if __name__ == "__main__":
    sys.exit(main())
