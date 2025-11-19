#!/usr/bin/env python3
"""
修复 SalesRAG 数据集中的 reward_model 字段
添加 ground_truth 包含必要的上下文信息
"""

import pandas as pd
import json
from pathlib import Path


def fix_reward_model_field(input_file: str, output_file: str = None):
    """
    修复数据集中的 reward_model 字段，添加 ground_truth
    
    Args:
        input_file: 输入的 parquet 文件路径
        output_file: 输出的 parquet 文件路径（如果为 None，则覆盖原文件）
    """
    print(f"📖 读取数据集: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"✅ 数据集大小: {len(df)} 条")
    
    # 检查第一条数据的结构
    if len(df) > 0:
        first_row = df.iloc[0]
        print(f"\n📋 第一条数据的字段: {list(first_row.keys())}")
        
        if 'reward_model' in first_row:
            print(f"📦 reward_model 类型: {type(first_row['reward_model'])}")
            print(f"📦 reward_model 内容: {first_row['reward_model']}")
    
    # 修复每一行的 reward_model 字段
    fixed_count = 0
    for idx, row in df.iterrows():
        reward_model = row.get('reward_model', {})
        
        # 如果 reward_model 不是字典，初始化为字典
        if not isinstance(reward_model, dict):
            reward_model = {}
        
        # 检查是否已有 ground_truth
        if 'ground_truth' not in reward_model:
            # 从 parsed_data 中提取必要信息
            parsed_data = row.get('parsed_data', {})
            if isinstance(parsed_data, str):
                try:
                    parsed_data = json.loads(parsed_data)
                except:
                    parsed_data = {}
            
            # 构建 ground_truth，包含 RAG 需要的上下文信息
            ground_truth = {
                'context': parsed_data.get('context', ''),
                'user_profile': parsed_data.get('user_profile', ''),
                'history_summary': parsed_data.get('history_summary', ''),
                'original_query': parsed_data.get('current_query', ''),
            }
            
            # 更新 reward_model
            reward_model['ground_truth'] = ground_truth
            reward_model['style'] = reward_model.get('style', 'rule')  # 默认为 rule-based
            
            df.at[idx, 'reward_model'] = reward_model
            fixed_count += 1
    
    print(f"\n✅ 修复了 {fixed_count} 条数据")
    
    # 保存修复后的数据
    if output_file is None:
        output_file = input_file
    
    print(f"💾 保存到: {output_file}")
    df.to_parquet(output_file, index=False)
    
    print("✅ 完成！")
    
    # 验证修复结果
    print("\n🔍 验证修复结果...")
    df_verify = pd.read_parquet(output_file)
    first_reward_model = df_verify.iloc[0]['reward_model']
    print(f"📦 修复后的 reward_model 结构:")
    print(json.dumps(first_reward_model, indent=2, ensure_ascii=False))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='修复 SalesRAG 数据集的 reward_model 字段')
    parser.add_argument('--input', '-i', required=True, help='输入的 parquet 文件路径')
    parser.add_argument('--output', '-o', help='输出的 parquet 文件路径（默认覆盖原文件）')
    parser.add_argument('--backup', '-b', action='store_true', help='是否备份原文件')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 备份原文件
    if args.backup:
        backup_path = input_path.with_suffix('.parquet.bak')
        print(f"📦 备份原文件到: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)
    
    # 修复数据
    fix_reward_model_field(args.input, args.output)


if __name__ == '__main__':
    main()
