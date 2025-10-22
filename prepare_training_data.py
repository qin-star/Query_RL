# -*- coding: utf-8 -*-
"""
DeepRetrieval训练数据准备脚本
将女博士FAQ数据转换为DeepRetrieval训练格式
"""

import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_deepretrieval_data():
    """准备DeepRetrieval训练数据"""
    
    # 1. 读取训练数据
    csv_file = Path('code/data/five_deal_answer_res.csv')
    logger.info(f"读取数据文件: {csv_file}")
    
    df = pd.read_csv(csv_file, encoding='utf-8')
    logger.info(f"数据加载完成,共 {len(df)} 条记录")
    
    # 2. 创建输出目录
    output_dir = Path('data/wuboshi_faq/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 3. 数据集划分 (80% train, 20% dev)
    train_size = int(len(df) * 0.8)
    df_train = df[:train_size]
    df_dev = df[train_size:]
    
    # 4. 转换训练集
    logger.info("转换训练集...")
    train_data = []
    for idx, row in df_train.iterrows():
        train_data.append({
            "query_id": f"q{idx}",
            "query": row['query'],
            "rewritten_query": row['res_queries'],  # 作为监督信号
            "answer": row['answer'],
            "corpus_ids": [f"doc{idx}"]
        })
    
    with open(output_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"训练集保存完成: {len(train_data)} 条")
    
    # 5. 转换验证集
    logger.info("转换验证集...")
    dev_data = []
    for idx, row in df_dev.iterrows():
        dev_data.append({
            "query_id": f"q{idx}",
            "query": row['query'],
            "rewritten_query": row['res_queries'],
            "answer": row['answer'],
            "corpus_ids": [f"doc{idx}"]
        })
    
    with open(output_dir / 'dev.jsonl', 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"验证集保存完成: {len(dev_data)} 条")
    
    # 6. 构建文档库 (corpus)
    logger.info("构建文档库...")
    corpus_data = []
    for idx, row in df.iterrows():
        corpus_data.append({
            "doc_id": f"doc{idx}",
            "title": row['query'],
            "text": row['answer']
        })
    
    with open(output_dir / 'corpus.jsonl', 'w', encoding='utf-8') as f:
        for item in corpus_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"文档库保存完成: {len(corpus_data)} 条")
    
    # 7. 生成统计信息
    stats = {
        "total_samples": len(df),
        "train_samples": len(train_data),
        "dev_samples": len(dev_data),
        "corpus_size": len(corpus_data),
        "avg_query_length": df['query'].str.len().mean(),
        "avg_answer_length": df['answer'].str.len().mean(),
        "categories": {
            "胶原蛋白相关": df['query'].str.contains('胶原蛋白').sum(),
            "备孕相关": df['query'].str.contains('备孕').sum(),
            "服用方法相关": df['query'].str.contains('服用|饮用|使用').sum(),
            "效果相关": df['query'].str.contains('效果|多久').sum(),
        }
    }
    
    with open(output_dir / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 8. 输出摘要
    logger.info("\n" + "="*60)
    logger.info("数据准备完成!")
    logger.info("="*60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"训练样本: {len(train_data)}")
    logger.info(f"验证样本: {len(dev_data)}")
    logger.info(f"文档总数: {len(corpus_data)}")
    logger.info(f"平均问题长度: {stats['avg_query_length']:.1f} 字符")
    logger.info(f"平均答案长度: {stats['avg_answer_length']:.1f} 字符")
    logger.info("\n数据分类统计:")
    for category, count in stats['categories'].items():
        logger.info(f"  {category}: {count} 条")
    logger.info("="*60)

if __name__ == "__main__":
    prepare_deepretrieval_data()

