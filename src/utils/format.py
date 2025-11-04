#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL格式化工具
将紧凑的单行JSON转换为多行缩进格式，便于阅读
"""

import json
import argparse
from pathlib import Path
import asyncio

from src.utils.log import logger


def format_jsonl_file(input_file, output_file=None, indent=2):
    """
    格式化JSONL文件
    
    Args:
        input_file (str): 输入的JSONL文件路径
        output_file (str): 输出文件路径，如果为None则在原文件名后加_formatted
        indent (int): 缩进空格数
    """
    # 确定输出文件路径
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_formatted{input_path.suffix}")
    
    # 读取并格式化
    formatted_lines = []
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        content = None
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    content = f.readlines()
                logger.info(f"使用编码: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            raise ValueError("无法读取文件，尝试了多种编码方式")
        
        for line_num, line in enumerate(content, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON
                json_obj = json.loads(line)
                # 格式化为多行
                formatted_json = json.dumps(json_obj, ensure_ascii=False, indent=indent)
                formatted_lines.append(formatted_json)
                logger.info(f"处理第 {line_num} 行完成")
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行JSON格式错误: {e}")
                continue
        
        # 写入格式化后的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, formatted_line in enumerate(formatted_lines):
                f.write(formatted_line)
                # 如果不是最后一行，添加分隔符
                if i < len(formatted_lines) - 1:
                    f.write('\n' + '='*50 + '\n')
                f.write('\n')
        
        logger.info(f"\n格式化完成!")
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"处理了 {len(formatted_lines)} 条记录")
        
    except FileNotFoundError:
        logger.error(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")


async def preview_jsonl_file(input_file, lines=None):
    """
    预览JSONL文件的前几行（格式化显示）
    
    Args:
        input_file (str): 输入的JSONL文件路径
        lines (int): 预览行数，None表示显示全部
    """
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        content = None
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    content = f.readlines()
                logger.info(f"使用编码: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            raise ValueError("无法读取文件，尝试了多种编码方式")
            
        if lines is None:
            logger.info(f"\n预览文件: {input_file} (全部内容)\n")
        else:
            logger.info(f"\n预览文件: {input_file} (前 {lines} 行)\n")
        
        count = 0
        for line_num, line in enumerate(content, 1):
            if lines is not None and count >= lines:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                json_obj = json.loads(line)
                formatted_json = json.dumps(json_obj, ensure_ascii=False, indent=2)
                logger.info(f"第 {line_num} 行:")
                logger.info(formatted_json)
                logger.info("-" * 50)
                count += 1
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行JSON格式错误: {e}")
                    
    except FileNotFoundError:
        logger.error(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        logger.error(f"预览过程中出错: {e}")


async def main():
    parser = argparse.ArgumentParser(description="JSONL文件格式化工具")
    parser.add_argument("input_file", help="输入的JSONL文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-i", "--indent", type=int, default=2, help="缩进空格数 (默认: 2)")
    parser.add_argument("-p", "--preview", action="store_true", help="只预览文件，不生成输出文件")
    parser.add_argument("-l", "--lines", type=int, help="预览行数 (不指定则显示全部)")
    
    args = parser.parse_args()
    
    if args.preview:
        await preview_jsonl_file(args.input_file, args.lines)
    else:
        await format_jsonl_file(args.input_file, args.output, args.indent)


if __name__ == "__main__":
    asyncio.run(main())
