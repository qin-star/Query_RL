#!/usr/bin/env python3
"""
验证GRPO+GPT-5混合训练实现
检查所有关键组件是否正确集成
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "verl_code"))
sys.path.insert(0, str(project_root / "src"))

def check_imports():
    """检查关键模块是否可以导入"""
    print("=" * 60)
    print("🔍 检查模块导入...")
    print("=" * 60)
    
    checks = []
    
    # 1. 检查ray_trainer修改
    try:
        from verl.trainer.ppo.ray_trainer import select_best_from_groups, RayPPOTrainer
        print("✅ select_best_from_groups 函数导入成功")
        checks.append(True)
        
        # 检查是否有新方法
        if hasattr(RayPPOTrainer, '_call_rag_and_compute_gpt5_rewards'):
            print("✅ RayPPOTrainer._call_rag_and_compute_gpt5_rewards 方法存在")
            checks.append(True)
        else:
            print("❌ RayPPOTrainer._call_rag_and_compute_gpt5_rewards 方法不存在")
            checks.append(False)
    except ImportError as e:
        print(f"❌ ray_trainer 导入失败: {e}")
        checks.append(False)
    
    # 2. 检查GPT-5评分器
    try:
        from verl.workers.gpt5_dual_model_rater import GPT5DualModelRater
        print("✅ GPT5DualModelRater 导入成功")
        checks.append(True)
    except ImportError as e:
        print(f"❌ GPT5DualModelRater 导入失败: {e}")
        checks.append(False)
    
    # 3. 检查RAG接口
    try:
        from src.core.rag_chater import RagChater
        print("✅ RagChater 导入成功")
        
        # 检查方法
        if hasattr(RagChater, 'chat_8b') and hasattr(RagChater, 'chat'):
            print("✅ RagChater.chat_8b 和 RagChater.chat 方法存在")
            checks.append(True)
        else:
            print("❌ RagChater 缺少必要方法")
            checks.append(False)
    except ImportError as e:
        print(f"❌ RagChater 导入失败: {e}")
        checks.append(False)
    
    # 4. 检查GRPO算法
    try:
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        print("✅ compute_grpo_outcome_advantage 导入成功")
        checks.append(True)
    except ImportError as e:
        print(f"❌ GRPO算法导入失败: {e}")
        checks.append(False)
    
    return all(checks)


def check_config_files():
    """检查配置文件是否存在"""
    print("\n" + "=" * 60)
    print("📋 检查配置文件...")
    print("=" * 60)
    
    config_file = project_root / "verl_code" / "config" / "sales_rag_grpo_hybrid_config.yaml"
    
    if config_file.exists():
        print(f"✅ 配置文件存在: {config_file}")
        
        # 读取并检查关键配置
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            checks = [
                ('data_source: sales_rag_hybrid', 'data_source配置'),
                ('select_best_from_group: true', '组内选择配置'),
                ('hybrid_grpo:', '混合训练配置'),
                ('gpt5_weight:', 'GPT-5权重配置'),
                ('grpo_weight:', 'GRPO权重配置'),
                ('n: 5', '候选数量配置'),
            ]
            
            all_ok = True
            for pattern, desc in checks:
                if pattern in content:
                    print(f"  ✅ {desc}")
                else:
                    print(f"  ❌ 缺少{desc}")
                    all_ok = False
            
            return all_ok
    else:
        print(f"❌ 配置文件不存在: {config_file}")
        return False


def check_data_files():
    """检查数据文件是否存在并验证格式"""
    print("\n" + "=" * 60)
    print("📊 检查数据文件...")
    print("=" * 60)
    
    train_file_parquet = project_root / "data" / "sales_rag" / "train.parquet"
    train_file_jsonl = project_root / "data" / "sales_rag" / "train.jsonl"
    val_file_parquet = project_root / "data" / "sales_rag" / "val.parquet"
    val_file_jsonl = project_root / "data" / "sales_rag" / "val.jsonl"
    
    checks = []
    
    # 检查训练数据
    train_file = None
    if train_file_parquet.exists():
        train_file = train_file_parquet
        print(f"✅ 训练数据存在: {train_file}")
        print(f"   大小: {train_file.stat().st_size / 1024:.2f} KB")
        checks.append(True)
    elif train_file_jsonl.exists():
        train_file = train_file_jsonl
        print(f"✅ 训练数据存在: {train_file}")
        print(f"   大小: {train_file.stat().st_size / 1024:.2f} KB")
        checks.append(True)
    else:
        print(f"❌ 训练数据不存在: {train_file_parquet} 或 {train_file_jsonl}")
        checks.append(False)
    
    # 验证数据格式（检查 reward_model.ground_truth）
    if train_file and train_file.suffix == '.parquet':
        try:
            import pandas as pd
            df = pd.read_parquet(train_file)
            
            if len(df) > 0:
                first_row = df.iloc[0]
                
                # 检查必要字段
                if 'reward_model' in first_row:
                    reward_model = first_row['reward_model']
                    
                    if isinstance(reward_model, dict):
                        if 'ground_truth' in reward_model:
                            print("  ✅ reward_model.ground_truth 字段存在")
                            ground_truth = reward_model['ground_truth']
                            
                            # 检查 ground_truth 的内容
                            if isinstance(ground_truth, dict):
                                required_keys = ['context', 'user_profile', 'history_summary']
                                missing_keys = [k for k in required_keys if k not in ground_truth]
                                
                                if not missing_keys:
                                    print("  ✅ ground_truth 包含所有必要字段")
                                    checks.append(True)
                                else:
                                    print(f"  ⚠️  ground_truth 缺少字段: {missing_keys}")
                                    checks.append(True)  # 警告但不阻止
                            else:
                                print(f"  ❌ ground_truth 不是字典类型: {type(ground_truth)}")
                                checks.append(False)
                        else:
                            print("  ❌ reward_model 缺少 ground_truth 字段")
                            print("  💡 运行修复脚本: python scripts/fix_reward_model_field.py --input data/sales_rag/train.parquet --backup")
                            checks.append(False)
                    else:
                        print(f"  ❌ reward_model 不是字典类型: {type(reward_model)}")
                        checks.append(False)
                else:
                    print("  ❌ 数据缺少 reward_model 字段")
                    checks.append(False)
        except Exception as e:
            print(f"  ⚠️  数据格式验证失败: {e}")
            checks.append(True)  # 不阻止，但给出警告
    
    # 检查验证数据
    if val_file_parquet.exists():
        print(f"✅ 验证数据存在: {val_file_parquet}")
        print(f"   大小: {val_file_parquet.stat().st_size / 1024:.2f} KB")
        checks.append(True)
    elif val_file_jsonl.exists():
        print(f"✅ 验证数据存在: {val_file_jsonl}")
        print(f"   大小: {val_file_jsonl.stat().st_size / 1024:.2f} KB")
        checks.append(True)
    else:
        print(f"⚠️  验证数据不存在 (可选)")
        checks.append(True)  # 验证数据是可选的
    
    return all(checks)


def check_scripts():
    """检查启动脚本是否存在"""
    print("\n" + "=" * 60)
    print("🚀 检查启动脚本...")
    print("=" * 60)
    
    script_file = project_root / "scripts" / "run_grpo_hybrid.sh"
    
    if script_file.exists():
        print(f"✅ 启动脚本存在: {script_file}")
        return True
    else:
        print(f"❌ 启动脚本不存在: {script_file}")
        return False


def print_summary(results):
    """打印验证总结"""
    print("\n" + "=" * 60)
    print("📝 验证总结")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results.values())
    
    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
    
    print("\n" + "-" * 60)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有检查通过！可以开始训练。")
        print("\n启动命令:")
        print("  cd scripts")
        print("  ./run_grpo_hybrid.sh")
    else:
        print("\n⚠️  部分检查失败，请修复后再试。")
    
    return passed == total


def main():
    """主函数"""
    print("🔍 GRPO+GPT-5混合训练实现验证")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print("=" * 60)
    
    results = {
        "模块导入": check_imports(),
        "配置文件": check_config_files(),
        "数据文件": check_data_files(),
        "启动脚本": check_scripts(),
    }
    
    success = print_summary(results)
    
    if success:
        print("\n" + "=" * 60)
        print("📖 快速开始指南")
        print("=" * 60)
        print("""
1. 确保RAG服务正在运行:
   - 8B接口: http://localhost:8000/chat_8b
   - 32B接口: http://localhost:8000/chat

2. 配置GPT-5 API密钥（如果需要）

3. 启动训练:
   cd scripts
   chmod +x run_grpo_hybrid.sh
   ./run_grpo_hybrid.sh

4. 监控训练（如果使用wandb）:
   访问 https://wandb.ai/your-project

5. 查看详细文档:
   cat GRPO_HYBRID_IMPLEMENTATION.md
        """)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
