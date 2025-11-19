#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用verl官方库函数将JSONL转换为Parquet格式
复用verl现有的数据处理逻辑
使用方法：logger
python scripts/jsonl_to_parquet_converter.py \
    --input "/home/jovyan2/query_rl/data/sales_rag/train_val.jsonl" \
    --output "/home/jovyan2/query_rl/data/sales_rag/val.parquet" \
    --validate
"""

import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSONL文件
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        List[Dict]: 数据列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
        
        logger.info(f"成功读取 {len(data)} 条记录从 {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
        raise


def convert_jsonl_to_parquet(jsonl_path: str, parquet_path: str) -> None:
    """
    使用pandas直接转换JSONL到Parquet格式
    这是verl官方推荐的标准方式
    
    Args:
        jsonl_path: 输入JSONL文件路径
        parquet_path: 输出Parquet文件路径
    """
    try:
        # 读取JSONL数据
        data = read_jsonl_file(jsonl_path)
        
        if not data:
            logger.warning(f"没有数据可以转换: {jsonl_path}")
            return
        
        # 转换为DataFrame - 这是verl标准格式
        df = pd.DataFrame(data)
        
        # 处理reward_model字段，确保包含ground_truth
        if 'reward_model' in df.columns:
            fixed_count = 0
            
            def fix_reward_model(row):
                """修复reward_model字段，添加ground_truth"""
                reward_model = row.get('reward_model', {})
                parsed_data = row.get('parsed_data', {})
                
                # 如果reward_model不是字典，初始化为字典
                if not isinstance(reward_model, dict):
                    reward_model = {}
                
                # 如果缺少ground_truth，从parsed_data中提取
                if 'ground_truth' not in reward_model or reward_model.get('ground_truth') == {}:
                    # 解析parsed_data（可能是字符串或字典）
                    if isinstance(parsed_data, str):
                        try:
                            parsed_data = json.loads(parsed_data)
                        except:
                            parsed_data = {}
                    
                    # 构建ground_truth
                    ground_truth = {
                        'context': parsed_data.get('context', ''),
                        'user_profile': parsed_data.get('user_profile', ''),
                        'history_summary': parsed_data.get('history_summary', ''),
                        'original_query': parsed_data.get('current_query', ''),
                    }
                    
                    reward_model['ground_truth'] = ground_truth
                    reward_model['style'] = reward_model.get('style', 'rule')
                    
                    nonlocal fixed_count
                    fixed_count += 1
                
                return reward_model
            
            df['reward_model'] = df.apply(fix_reward_model, axis=1)
            
            if fixed_count > 0:
                logger.info(f"✓ 为 {fixed_count} 条记录添加了 ground_truth 字段")
        
        # 确保输出目录存在
        output_path = Path(parquet_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为Parquet格式 - 使用verl兼容的设置
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='snappy',
            index=False  # verl不需要索引
        )
        
        logger.info(f"成功转换 {len(data)} 条记录到 {parquet_path}")
        logger.info(f"Parquet文件信息: 行数={len(df)}, 列数={len(df.columns)}")
        
        # 显示数据预览
        if len(df) > 0:
            logger.info("数据预览:")
            logger.info(f"列名: {list(df.columns)}")
            if 'prompt' in df.columns:
                logger.info(f"第一个prompt示例: {str(df['prompt'].iloc[0])[:100]}...")
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        raise


def validate_parquet_file(parquet_path: str) -> bool:
    """
    验证Parquet文件是否符合verl要求
    
    Args:
        parquet_path: Parquet文件路径
        
    Returns:
        bool: 验证结果
    """
    try:
        # 读取Parquet文件
        df = pd.read_parquet(parquet_path)
        
        logger.info(f"验证Parquet文件: {parquet_path}")
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"列名: {list(df.columns)}")
        
        # 检查verl必需的字段
        required_fields = ['prompt', 'data_source']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            logger.warning(f"缺少verl必需字段: {missing_fields}")
        else:
            logger.info("✓ 所有verl必需字段都存在")
        
        # 检查数据完整性
        if 'prompt' in df.columns:
            empty_prompts = df['prompt'].isna().sum()
            if empty_prompts > 0:
                logger.warning(f"发现 {empty_prompts} 个空prompt")
            else:
                logger.info("✓ 所有prompt字段都有效")
        
        # 检查reward_model字段（GRPO需要）
        if 'reward_model' in df.columns:
            logger.info("✓ 检测到reward_model字段，符合GRPO要求")
            
            # 检查ground_truth字段
            first_reward_model = df['reward_model'].iloc[0]
            if isinstance(first_reward_model, dict):
                if 'ground_truth' in first_reward_model:
                    logger.info("✓ reward_model包含ground_truth字段")
                    ground_truth = first_reward_model['ground_truth']
                    if isinstance(ground_truth, dict):
                        logger.info(f"  - ground_truth字段: {list(ground_truth.keys())}")
                        required_keys = ['context', 'user_profile', 'history_summary']
                        missing_keys = [k for k in required_keys if k not in ground_truth]
                        if not missing_keys:
                            logger.info("✓ ground_truth包含所有必要字段")
                        else:
                            logger.warning(f"⚠ ground_truth缺少推荐字段: {missing_keys}")
                    else:
                        logger.warning(f"⚠ ground_truth不是字典类型: {type(ground_truth)}")
                else:
                    logger.error("✗ reward_model缺少ground_truth字段")
                    return False
            else:
                logger.warning(f"⚠ reward_model不是字典类型: {type(first_reward_model)}")
        
        return True
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='JSONL转Parquet格式转换器 - 兼容verl框架',
        epilog='示例: python jsonl_to_parquet_converter.py --input train.jsonl --output train.parquet'
    )
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, 
                       help='输出Parquet文件路径')
    parser.add_argument('--validate', '-v', action='store_true', 
                       help='转换后验证文件是否符合verl要求')
    
    args = parser.parse_args()
    
    try:
        logger.info("开始JSONL到Parquet格式转换...")
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")
        
        # 执行转换
        convert_jsonl_to_parquet(args.input, args.output)
        
        # 验证（如果指定）
        if args.validate:
            logger.info("\n开始验证Parquet文件...")
            is_valid = validate_parquet_file(args.output)
            if is_valid:
                logger.info("✓ Parquet文件验证通过，符合verl要求")
            else:
                logger.warning("✗ Parquet文件验证失败")
                return 1
        
        logger.info("\n转换完成！🎉")
        logger.info("生成的Parquet文件可以直接用于verl训练框架")
        
    except Exception as e:
        logger.error(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())