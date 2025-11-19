#!/usr/bin/env python3
"""验证并修复Parquet文件"""
import pyarrow.parquet as pq
import pandas as pd
import sys

def verify_and_fix_parquet(file_path):
    """验证并尝试修复Parquet文件"""
    print(f"\n检查文件: {file_path}")
    
    try:
        # 方法1: 使用pyarrow直接读取
        print("方法1: 使用pyarrow读取...")
        table = pq.read_table(file_path)
        print(f"✓ PyArrow读取成功: {len(table)} 行")
        
        # 方法2: 使用pandas读取
        print("方法2: 使用pandas读取...")
        df = pd.read_parquet(file_path)
        print(f"✓ Pandas读取成功: {df.shape}")
        
        # 方法3: 使用datasets库读取（verl使用的方式）
        print("方法3: 使用datasets库读取...")
        import datasets
        dataset = datasets.load_dataset("parquet", data_files=file_path)["train"]
        print(f"✓ Datasets读取成功: {len(dataset)} 行")
        
        print(f"\n✅ 文件 {file_path} 完全正常！")
        return True
        
    except Exception as e:
        print(f"\n❌ 文件损坏: {e}")
        print("\n尝试修复...")
        
        try:
            # 读取并重新保存
            df = pd.read_parquet(file_path, engine='fastparquet')
            backup_path = file_path + ".backup"
            fixed_path = file_path + ".fixed"
            
            # 备份原文件
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"✓ 已备份到: {backup_path}")
            
            # 重新保存
            df.to_parquet(fixed_path, engine='pyarrow', compression='snappy', index=False)
            print(f"✓ 已修复到: {fixed_path}")
            print(f"\n请手动替换: mv {fixed_path} {file_path}")
            
        except Exception as fix_error:
            print(f"❌ 修复失败: {fix_error}")
            return False
    
    return False

if __name__ == "__main__":
    files = [
        "/home/jovyan2/query_rl/query_rl_code/data/sales_rag/train.parquet",
        "/home/jovyan2/query_rl/query_rl_code/data/sales_rag/val.parquet"
    ]
    
    all_ok = True
    for f in files:
        if not verify_and_fix_parquet(f):
            all_ok = False
    
    sys.exit(0 if all_ok else 1)
