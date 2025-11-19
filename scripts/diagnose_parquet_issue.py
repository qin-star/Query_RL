#!/usr/bin/env python3
"""
诊断 Parquet 文件加载问题
"""

import os
import sys
from pathlib import Path

def check_file(filepath):
    """检查单个文件"""
    print(f"\n{'='*60}")
    print(f"检查文件: {filepath}")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    # 检查文件大小
    size = os.path.getsize(filepath)
    print(f"✓ 文件存在，大小: {size:,} bytes ({size/1024/1024:.2f} MB)")
    
    # 检查文件权限
    readable = os.access(filepath, os.R_OK)
    print(f"✓ 可读权限: {readable}")
    
    # 检查文件头（magic bytes）
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            print(f"✓ 文件头 (前4字节): {header.hex()}")
            
            # Parquet 文件应该以 'PAR1' 开头
            if header == b'PAR1':
                print("✓ 文件头正确 (PAR1)")
            else:
                print(f"❌ 文件头错误，应该是 'PAR1' (50415231)，实际是: {header}")
                return False
            
            # 检查文件尾
            f.seek(-8, 2)  # 从文件末尾往前8字节
            footer = f.read(8)
            print(f"✓ 文件尾 (后8字节): {footer.hex()}")
            
            # Parquet 文件尾应该以 'PAR1' 结尾
            if footer[-4:] == b'PAR1':
                print("✓ 文件尾正确 (PAR1)")
            else:
                print(f"❌ 文件尾错误，应该以 'PAR1' 结尾，实际是: {footer[-4:]}")
                return False
                
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    # 尝试用 pyarrow 读取
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(filepath)
        print(f"✓ PyArrow 读取成功: {len(table)} 行, {len(table.schema)} 列")
        print(f"  列名: {table.schema.names}")
    except Exception as e:
        print(f"❌ PyArrow 读取失败: {e}")
        return False
    
    # 尝试用 datasets 读取
    try:
        import datasets
        ds = datasets.load_dataset('parquet', data_files=filepath, split='train')
        print(f"✓ Datasets 读取成功: {len(ds)} 行")
    except Exception as e:
        print(f"❌ Datasets 读取失败: {e}")
        return False
    
    print(f"✅ 文件 {filepath} 完全正常！")
    return True


def main():
    project_root = Path(__file__).parent.parent
    
    print("🔍 Parquet 文件诊断工具")
    print(f"项目根目录: {project_root}")
    
    # 检查所有可能的 Parquet 文件路径
    possible_paths = [
        # 绝对路径
        project_root / "data/sales_rag/train.parquet",
        project_root / "data/sales_rag/val.parquet",
        
        # 相对路径（从 verl_code 目录）
        project_root / "verl_code/../data/sales_rag/train.parquet",
        project_root / "verl_code/../data/sales_rag/val.parquet",
        
        # 其他可能的位置
        Path("/home/jovyan2/query_rl/data/sales_rag/train.parquet"),
        Path("/home/jovyan2/query_rl/data/sales_rag/val.parquet"),
        Path("/home/jovyan2/query_rl/query_rl_code/data/sales_rag/train.parquet"),
        Path("/home/jovyan2/query_rl/query_rl_code/data/sales_rag/val.parquet"),
    ]
    
    # 去重并检查
    checked_paths = set()
    all_ok = True
    
    for path in possible_paths:
        # 解析为绝对路径
        abs_path = path.resolve()
        
        # 跳过已检查的路径
        if abs_path in checked_paths:
            continue
        checked_paths.add(abs_path)
        
        # 检查文件
        if abs_path.exists():
            result = check_file(str(abs_path))
            if not result:
                all_ok = False
    
    print(f"\n{'='*60}")
    if all_ok:
        print("✅ 所有 Parquet 文件都正常！")
        print("\n建议：")
        print("1. 检查 Hydra 配置是否正确加载了文件路径")
        print("2. 设置 HYDRA_FULL_ERROR=1 查看完整错误堆栈")
        print("3. 检查是否有其他地方硬编码了错误的路径")
    else:
        print("❌ 发现问题！请根据上面的错误信息修复文件")
        print("\n建议：")
        print("1. 重新生成 Parquet 文件")
        print("2. 检查磁盘空间和文件权限")
        print("3. 如果使用网络文件系统，尝试复制到本地")
    print(f"{'='*60}\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
