#!/usr/bin/env python3
"""
检查配置文件中缺失的必需字段
"""

import sys
from pathlib import Path
from omegaconf import OmegaConf

# 必需的 trainer 字段
REQUIRED_TRAINER_FIELDS = [
    'balance_batch',
    'total_epochs',
    'total_training_steps',
    'project_name',
    'experiment_name',
    'logger',
    'log_val_generations',
    'rollout_data_dir',
    'validation_data_dir',
    'nnodes',
    'n_gpus_per_node',
    'save_freq',
    'test_freq',
    'critic_warmup',
    'default_hdfs_dir',
    'del_local_ckpt_after_load',
    'default_local_dir',
    'max_actor_ckpt_to_keep',
    'max_critic_ckpt_to_keep',
    'ray_wait_register_center_timeout',
    'device',
    'use_legacy_worker_impl',
    'resume_mode',
    'resume_from_path',
    'val_only',
    'val_before_train',
    'esi_redundant_time',
]

def check_fields(cfg, required_fields, section_name):
    """检查必需字段"""
    missing = []
    
    if not hasattr(cfg, section_name):
        print(f"❌ 缺少整个 {section_name} 部分")
        return required_fields
    
    section = getattr(cfg, section_name)
    
    for field in required_fields:
        if not hasattr(section, field):
            missing.append(field)
    
    return missing

def main():
    config_path = Path(__file__).parent.parent / "verl_code/config/sales_rag_grpo_hybrid_config.yaml"
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    print(f"🔍 检查配置文件: {config_path.name}")
    print("=" * 60)
    
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return 1
    
    # 检查 trainer 字段
    print("\n📋 检查 trainer 字段...")
    missing = check_fields(cfg, REQUIRED_TRAINER_FIELDS, 'trainer')
    
    if missing:
        print(f"❌ 缺少 {len(missing)} 个字段:")
        for field in missing:
            print(f"  - trainer.{field}")
        print("\n建议添加这些字段到配置文件")
        return 1
    else:
        print("✓ 所有 trainer 字段都存在")
    
    # 显示当前配置
    print("\n📊 当前 trainer 配置:")
    for field in REQUIRED_TRAINER_FIELDS:
        value = getattr(cfg.trainer, field)
        print(f"  {field}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ 配置检查通过！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
