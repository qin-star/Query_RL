#!/usr/bin/env python3
"""
快速检查数据集的 reward_model 字段
"""

import pandas as pd
import json
import sys
from pathlib import Path


def check_dataset(file_path):
    """检查数据集格式"""
    print(f"📖 检查数据集: {file_path}")
    print("=" * 60)
    
    try:
        # 读取数据
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            print(f"❌ 不支持的文件格式: {file_path}")
            return False
        
        print(f"✅ 数据集大小: {len(df)} 条")
        print(f"✅ 字段列表: {list(df.columns)}")
        
        if len(df) == 0:
            print("⚠️  数据集为空")
            return False
        
        # 检查第一条数据
        first_row = df.iloc[0]
        print("\n" + "=" * 60)
        print("🔍 检查第一条数据...")
        print("=" * 60)
        
        # 检查 reward_model 字段
        if 'reward_model' not in first_row:
            print("❌ 缺少 'reward_model' 字段")
            print("\n可用字段:", list(first_row.keys()))
            return False
        
        reward_model = first_row['reward_model']
        print(f"✅ reward_model 字段存在")
        print(f"   类型: {type(reward_model)}")
        
        if not isinstance(reward_model, dict):
            print(f"❌ reward_model 不是字典类型")
            print(f"   内容: {reward_model}")
            return False
        
        print(f"   键: {list(reward_model.keys())}")
        
        # 检查 ground_truth
        if 'ground_truth' not in reward_model:
            print("\n❌ reward_model 缺少 'ground_truth' 字段")
            print("\n📋 当前 reward_model 结构:")
            print(json.dumps(reward_model, indent=2, ensure_ascii=False))
            print("\n" + "=" * 60)
            print("💡 修复建议:")
            print("=" * 60)
            print("运行以下命令修复数据集:")
            print(f"  python scripts/fix_reward_model_field.py --input {file_path} --backup")
            return False
        
        ground_truth = reward_model['ground_truth']
        print(f"\n✅ ground_truth 字段存在")
        print(f"   类型: {type(ground_truth)}")
        
        if isinstance(ground_truth, dict):
            print(f"   键: {list(ground_truth.keys())}")
            print("\n📋 ground_truth 内容:")
            print(json.dumps(ground_truth, indent=2, ensure_ascii=False))
            
            # 检查必要字段
            required_keys = ['context', 'user_profile', 'history_summary']
            missing_keys = [k for k in required_keys if k not in ground_truth]
            
            if missing_keys:
                print(f"\n⚠️  ground_truth 缺少推荐字段: {missing_keys}")
                print("   (这些字段对于 RAG 评估很重要)")
            else:
                print("\n✅ ground_truth 包含所有推荐字段")
        else:
            print(f"   内容: {ground_truth}")
        
        print("\n" + "=" * 60)
        print("✅ 数据集格式检查通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python quick_check_dataset.py <数据集文件路径>")
        print("\n示例:")
        print("  python scripts/quick_check_dataset.py data/sales_rag/train.parquet")
        print("  python scripts/quick_check_dataset.py data/sales_rag/val.parquet")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    
    success = check_dataset(file_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
