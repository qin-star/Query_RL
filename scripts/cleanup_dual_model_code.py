#!/usr/bin/env python3
"""
清理双模型对比代码脚本
安全备份并移除过时的双模型对比实现，保留修正版混合GRPO架构
"""

import os
import shutil
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 备份目录
BACKUP_DIR = "/home/jovyan2/query_rl/deprecated/dual_model_backup"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 需要清理的文件列表
FILES_TO_CLEANUP = [
    "/home/jovyan2/query_rl/verl_code/verl/workers/dual_model_reward_calculator.py",
    "/home/jovyan2/query_rl/verl_code/verl/workers/gpt5_dual_model_rater.py",
]

# 需要更新的配置文件
CONFIG_FILES_TO_UPDATE = [
    "/home/jovyan2/query_rl/verl_code/verl/trainer/config/sales_rag_grpo_dual_model_config.yaml",
    "/home/jovyan2/query_rl/verl_code/config/sales_rag_grpo_dual_model_config.yaml",
]

def backup_file(file_path):
    """备份文件"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在，跳过备份: {file_path}")
            return False
        
        # 创建备份文件名
        filename = os.path.basename(file_path)
        backup_filename = f"{TIMESTAMP}_{filename}"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        # 复制文件
        shutil.copy2(file_path, backup_path)
        logger.info(f"✅ 备份成功: {file_path} -> {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 备份失败: {file_path}, 错误: {e}")
        return False

def safe_remove_file(file_path):
    """安全移除文件"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在，跳过移除: {file_path}")
            return True
        
        # 先备份
        if backup_file(file_path):
            # 然后移除
            os.remove(file_path)
            logger.info(f"✅ 移除成功: {file_path}")
            return True
        else:
            logger.error(f"❌ 由于备份失败，跳移除: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 移除失败: {file_path}, 错误: {e}")
        return False

def update_config_file(config_path):
    """更新配置文件，添加废弃标记"""
    try:
        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}")
            return False
        
        # 读取原文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 添加废弃标记
        deprecated_header = f"""# ⚠️  已废弃 - 双模型对比配置
# 此配置文件基于错误的双模型对比架构，已被混合GRPO架构替代
# 请使用新的混合GRPO配置：algorithm.hybrid_grpo
# 备份时间: {TIMESTAMP}
# 替代文件: verl_code/verl/trainer/config/ppo_trainer.yaml (algorithm.hybrid_grpo部分)

"""
        
        new_content = deprecated_header + content
        
        # 先备份
        if backup_file(config_path):
            # 然后更新
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info(f"✅ 更新配置文件成功: {config_path}")
            return True
        else:
            logger.error(f"❌ 由于备份失败，跳过更新: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 更新配置文件失败: {config_path}, 错误: {e}")
        return False

def create_migration_guide():
    """创建迁移指南"""
    try:
        guide_content = f"""# 双模型对比架构迁移指南

## 🚫 已废弃的架构
基于双模型对比的GRPO实现已被废弃，原因：
- ❌ 误解了GRPO核心原理（组内相对优化 vs 跨模型对比）
- ❌ 试图用绝对质量评估替代相对优势计算
- ❌ 破坏了GRPO的零均值特性

## ✅ 新的混合GRPO架构
已迁移到修正版混合GRPO架构：

### 核心改进
1. **保持GRPO核心**：仍然使用官方组内相对优化算法
2. **辅助信号增强**：GPT-5提供质量信号，但不破坏相对性
3. **零均值保持**：所有奖励都经过组内中心化
4. **官方兼容性**：完全兼容verl官方GRPO实现

### 关键文件变更
```
废弃文件（已备份到 {BACKUP_DIR}）：
- dual_model_reward_calculator.py → 混合奖励计算器
- gpt5_dual_model_rater.py → 组内中心化评估

新增文件：
- hybrid_grpo_reward_calculator.py → 修正版混合奖励计算
- grpo_group_generator.py → GRPO组生成器
- actor_model_processor_v2.py → 组内多样本生成
```

### 配置变更
```yaml
# 旧配置（已废弃）
algorithm.hybrid_training:
  enable: true
  auxiliary_weight: 0.3
  # ...其他双模型对比参数

# 新配置（推荐使用）
algorithm.hybrid_grpo:
  enable: true
  grpo_weight: 0.7           # GRPO主权重
  auxiliary_weight: 0.3      # GPT-5辅助权重
  auxiliary_centralization: true  # 关键：组内中心化
  auxiliary_normalization: std    # 标准化方式
```

### 训练脚本变更
```bash
# 旧脚本（已废弃）
scripts/run_grpo_query_RL.sh

# 新脚本（推荐使用）
scripts/run_hybrid_grpo_official_format.sh
```

## 🔧 迁移步骤
1. 备份现有配置和代码
2. 使用新的混合GRPO训练脚本
3. 更新配置文件为algorithm.hybrid_grpo格式
4. 验证训练流程正确性

## 📞 技术支持
如遇到问题，请参考：
- 修正版架构文档：GRPO_RAG_Query_Rewrite_Architecture_v3.md
- 官方GRPO实现：verl_code/verl/trainer/ppo/core_algos.py
- 混合奖励计算器：hybrid_grpo_reward_calculator.py

备份时间: {TIMESTAMP}
"""
        
        guide_path = os.path.join(BACKUP_DIR, f"{TIMESTAMP}_MIGRATION_GUIDE.md")
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"✅ 创建迁移指南成功: {guide_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建迁移指南失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始清理双模型对比代码")
    logger.info(f"备份目录: {BACKUP_DIR}")
    logger.info(f"时间戳: {TIMESTAMP}")
    
    # 创建备份目录
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # 1. 清理核心代码文件
    logger.info("📁 清理核心代码文件...")
    for file_path in FILES_TO_CLEANUP:
        safe_remove_file(file_path)
    
    # 2. 更新配置文件
    logger.info("⚙️  更新配置文件...")
    for config_path in CONFIG_FILES_TO_UPDATE:
        update_config_file(config_path)
    
    # 3. 创建迁移指南
    logger.info("📖 创建迁移指南...")
    create_migration_guide()
    
    logger.info("✅ 清理完成！")
    logger.info(f"📁 所有备份文件保存在: {BACKUP_DIR}")
    logger.info("🔄 请使用新的混合GRPO架构继续开发")

if __name__ == "__main__":
    main()