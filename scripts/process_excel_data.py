#!/usr/bin/env python3

"""
Excel训练数据处理脚本
用于处理橙啦-query_RL_训练集.xlsx文件，生成训练数据
"""

import pandas as pd
import re
import json
import logging
import argparse
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExcelDataProcessor:
    """Excel数据处理器"""
    
    def __init__(self, excel_path: str):
        """
        初始化Excel数据处理器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = Path(excel_path)
        self.data = None
        
        # 验证文件是否存在
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel文件不存在: {excel_path}")
        
        logger.info(f"初始化Excel数据处理器，文件路径: {excel_path}")
    
    def read_excel(self) -> pd.DataFrame:
        """
        读取Excel文件
        
        Returns:
            pd.DataFrame: 读取的数据框
        """
        try:
            logger.info(f"开始读取Excel文件: {self.excel_path}")
            
            # 读取Excel文件
            self.data = pd.read_excel(self.excel_path)
            
            logger.info(f"Excel文件读取成功，共 {len(self.data)} 行数据")
            logger.info(f"列名: {list(self.data.columns)}")
            
            # 检查是否包含必需的列（支持多种列名）
            dialogue_column = None
            possible_column_names = ['历史传参上下文', '最终传参上下文', '对话历史', '历史对话']
            
            for col_name in possible_column_names:
                if col_name in self.data.columns:
                    dialogue_column = col_name
                    break
            
            if dialogue_column is None:
                raise ValueError(f"Excel文件中缺少必需的对话数据列，期望的列名: {possible_column_names}，实际列名: {list(self.data.columns)}")
            
            # 重命名列到标准名称
            self.data = self.data.rename(columns={dialogue_column: '历史传参上下文'})
            logger.info(f"使用列 '{dialogue_column}' 作为对话数据列")
            
            return self.data
            
        except Exception as e:
            logger.error(f"读取Excel文件失败: {e}")
            raise
    
    def extract_query_and_history(self, dialogue_text: str) -> Dict[str, str]:
        """
        从对话文本中提取用户查询和历史对话
        保留[客户][时间戳]:前缀格式
        
        Args:
            dialogue_text: 完整的对话历史文本
        
        Returns:
            dict: 包含history_chat和query的字典
        """
        try:
            # 预处理对话文本
            dialogue_text = self.preprocess_dialogue(dialogue_text)
            
            # 按行分割对话
            lines = dialogue_text.strip().split('\n')
            
            # 找到最后一个客户发言
            last_customer_line = None
            history_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 解析发言者
                if line.startswith('[客户]'):
                    last_customer_line = line
                    history_lines.append(line)
                elif line.startswith('[销售]'):
                    history_lines.append(line)
                else:
                    # 处理可能的格式问题
                    history_lines.append(line)
            
            # 提取查询（最后一个客户的发言内容，保留完整前缀）
            if last_customer_line:
                query_content = last_customer_line  # 保留完整格式，包括[客户][时间戳]:前缀
            else:
                query_content = ""
            
            # 构建历史对话（除最后一个客户发言外的所有内容）
            if last_customer_line and last_customer_line in history_lines:
                history_lines.remove(last_customer_line)
            
            history_chat = '\n'.join(history_lines)
            
            result = {
                "history_chat": history_chat,
                "query": query_content
            }
            
            logger.debug(f"提取结果 - history_chat长度: {len(history_chat)}, query: {query_content}")
            
            return result
            
        except Exception as e:
            logger.error(f"提取查询和历史对话失败: {e}")
            return {
                "history_chat": "",
                "query": ""
            }
    
    def preprocess_dialogue(self, dialogue_text: str) -> str:
        """
        预处理对话文本
        
        Args:
            dialogue_text: 原始对话文本
        
        Returns:
            str: 预处理后的对话文本
        """
        try:
            # 标准化空白字符，但保留换行符
            lines = dialogue_text.strip().split('\n')
            processed_lines = []
            
            for line in lines:
                # 移除行首行尾的空白字符，但保留内容
                processed_line = line.strip()
                if processed_line:  # 只保留非空行
                    processed_lines.append(processed_line)
            
            # 重新组合
            processed_text = '\n'.join(processed_lines)
            
            return processed_text
            
        except Exception as e:
            logger.warning(f"预处理对话文本失败: {e}")
            return dialogue_text
    
    def load_prompt_template(self, template_path: str) -> str:
        """
        加载prompt模板
        
        Args:
            template_path: 模板文件路径
        
        Returns:
            str: 模板内容
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            logger.info(f"成功加载prompt模板: {template_path}")
            return template_content
            
        except Exception as e:
            logger.error(f"加载prompt模板失败: {e}")
            raise
    
    def combine_prompt(self, template: str, data: Dict[str, str]) -> str:
        """
        组合prompt模板和数据
        
        Args:
            template: prompt模板
            data: 包含history_chat, query, user_profile, thought的数据
        
        Returns:
            str: 组合后的完整prompt
        """
        try:
            # 准备模板参数
            template_data = {
                "history_chat": data.get("history_chat", ""),
                "query": data.get("query", ""),
                "thought": data.get("thought", "")
            }
            
            # 使用简单的字符串替换（避免Jinja2依赖）
            combined_prompt = template
            
            # 替换占位符
            combined_prompt = combined_prompt.replace("{{history_chat}}", template_data["history_chat"])
            combined_prompt = combined_prompt.replace("{{query}}", template_data["query"])
            combined_prompt = combined_prompt.replace("{{thought}}", template_data["thought"])
            
            # 处理条件语句（简单实现）
            # 移除没有thought的条件块
            if not template_data["thought"]:
                # 移除 {% if thought %} ... {% endif %} 块
                import re
                pattern = r'\{% if thought %\}(.*?)\{% endif %\}'
                combined_prompt = re.sub(pattern, '', combined_prompt, flags=re.DOTALL)
            
            logger.debug(f"Prompt组合完成，长度: {len(combined_prompt)}")
            return combined_prompt
            
        except Exception as e:
            logger.error(f"Prompt组合失败: {e}")
            raise
    
    def generate_training_samples(self, prompt_template_path: str) -> List[Dict[str, Any]]:
        """
        生成训练样本 v2.0 - 支持双模型GRPO架构
        
        Args:
            prompt_template_path: prompt模板文件路径
        
        Returns:
            List[Dict]: 训练样本列表
        """
        if self.data is None:
            self.read_excel()
        
        # 加载prompt模板
        template = self.load_prompt_template(prompt_template_path)
        
        training_samples = []
        
        for idx, row in self.data.iterrows():
            try:
                dialogue_text = row['历史传参上下文']
                
                if pd.isna(dialogue_text) or not str(dialogue_text).strip():
                    logger.warning(f"第 {idx} 行数据为空，跳过")
                    continue
                
                # 提取查询和历史对话
                parsed_data = self.extract_query_and_history(str(dialogue_text))
                
                # 验证提取结果
                if not parsed_data["query"]:
                    logger.warning(f"第 {idx} 行未能提取到用户查询，跳过")
                    continue
                
                # 添加可选字段（Excel中没有，设为空）
                parsed_data["user_profile"] = ""
                parsed_data["thought"] = ""
                
                # 组合prompt
                complete_prompt = self.combine_prompt(template, parsed_data)
                
                # 🔥 新增：构建符合GRPO架构的训练样本格式
                training_sample = {
                    "prompt_id": f"train_{idx:06d}",
                    "original_dialogue": str(dialogue_text),
                    "prompt": complete_prompt,
                    "parsed_data": parsed_data,
                    
                    # 🔥 新增：模型配置
                    "model_configs": {
                        "actor_model": {
                            "model_name": "Qwen3-8B-Instruct",
                            "rag_endpoint": "/chat_8b",
                            "input_format": "structured_json",
                            "expected_output": ["user_profile", "rewritten_query", "history_summary"]
                        },
                        "reference_model": {
                            "model_name": "Qwen3-32B-Instruct",
                            "rag_endpoint": "/chat",
                            "input_format": "raw_prompt",
                            "expected_output": "full_response"
                        }
                    },
                    
                    # 🔥 新增：评分配置
                    "reward_config": {
                        "scoring_model": "GPT-5",
                        "scoring_dimensions": ["质量提升度", "相关性准确性", "信息完整性", "检索有效性"],
                        "comparison_mode": "dual_model"
                    },
                    
                    # 🔥 保留原有字段用于向后兼容
                    "data_source": "sales_rag_rl",
                    "actor_input": {
                        "history_chat": parsed_data["history_chat"],
                        "query": parsed_data["query"],
                        "user_profile": parsed_data["user_profile"],
                        "thought": parsed_data["thought"],
                        "original_dialogue": str(dialogue_text)
                    },
                    "reward_model": {
                        "type": "gpt5_comparison",
                        "baseline_model": "Qwen3-32B-Instruct",
                        "scoring_dimensions": ["质量提升度", "相关性准确性", "信息完整性", "检索有效性"]
                    },
                    "expected_output_format": {
                        "user_profile": "string",
                        "rewritten_query": "string",
                        "history_summary": "string"
                    },
                    "metadata": {
                        "source_file": str(self.excel_path),
                        "row_index": idx,
                        "processing_timestamp": pd.Timestamp.now().isoformat(),
                        "architecture_version": "v2.0"
                    }
                }
                
                training_samples.append(training_sample)
                
            except Exception as e:
                logger.error(f"处理第 {idx} 行数据时发生错误: {e}")
                continue
        
        logger.info(f"成功生成 {len(training_samples)} 个训练样本（v2.0 GRPO架构）")
        return training_samples
    
    def save_training_samples(self, output_path: str, samples: List[Dict[str, Any]]) -> None:
        """
        保存训练样本到文件
        
        Args:
            output_path: 输出文件路径
            samples: 训练样本列表
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据文件扩展名选择保存格式
            if output_path.suffix.lower() == '.jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            elif output_path.suffix.lower() == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的输出格式: {output_path.suffix}")
            
            logger.info(f"训练样本已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存训练样本失败: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            Dict: 统计信息
        """
        if self.data is None:
            self.read_excel()
        
        # 基本统计
        total_rows = len(self.data)
        non_empty_rows = len(self.data.dropna(subset=['历史传参上下文']))
        
        # 对话长度统计
        dialogue_lengths = []
        query_lengths = []
        
        for idx, row in self.data.iterrows():
            dialogue_text = row['历史传参上下文']
            if pd.isna(dialogue_text):
                continue
            
            dialogue_text = str(dialogue_text)
            dialogue_lengths.append(len(dialogue_text))
            
            # 提取查询长度
            parsed_data = self.extract_query_and_history(dialogue_text)
            query_lengths.append(len(parsed_data["query"]))
        
        statistics = {
            "total_rows": total_rows,
            "non_empty_rows": non_empty_rows,
            "empty_rows": total_rows - non_empty_rows,
            "avg_dialogue_length": sum(dialogue_lengths) / len(dialogue_lengths) if dialogue_lengths else 0,
            "max_dialogue_length": max(dialogue_lengths) if dialogue_lengths else 0,
            "min_dialogue_length": min(dialogue_lengths) if dialogue_lengths else 0,
            "avg_query_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
            "max_query_length": max(query_lengths) if query_lengths else 0,
            "min_query_length": min(query_lengths) if query_lengths else 0,
        }
        
        return statistics
    
    def build_validation_dataset(self, sample_size: int = 100, method: str = "random",
                               seed: int = 42) -> List[Dict[str, Any]]:
        """
        构建验证数据集 - 从原始数据中随机采样
        
        Args:
            sample_size: 采样大小
            method: 采样方法
            seed: 随机种子
            
        Returns:
            List[Dict]: 验证样本列表
        """
        if self.data is None:
            self.read_excel()
        
        # 设置随机种子
        random.seed(seed)
        
        # 获取所有对话数据
        all_dialogues = []
        for idx, row in self.data.iterrows():
            dialogue_text = row['历史传参上下文']
            if pd.notna(dialogue_text) and str(dialogue_text).strip():
                all_dialogues.append(str(dialogue_text).strip())
        
        if not all_dialogues:
            logger.warning("没有可用的对话数据用于构建验证数据集")
            return []
        
        # 根据采样大小调整
        if sample_size > len(all_dialogues):
            logger.warning(f"采样大小 {sample_size} 大于可用数据量 {len(all_dialogues)}，使用全部数据")
            sample_size = len(all_dialogues)
        
        # 执行采样
        if method == "random":
            sampled_dialogues = random.sample(all_dialogues, sample_size)
        elif method == "stratified":
            # 分层采样：根据对话长度分层
            dialogues_by_length = {}
            for dialogue in all_dialogues:
                length_category = len(dialogue) // 200  # 每200字符为一个层级
                if length_category not in dialogues_by_length:
                    dialogues_by_length[length_category] = []
                dialogues_by_length[length_category].append(dialogue)
            
            sampled_dialogues = []
            samples_per_layer = sample_size // len(dialogues_by_length)
            remainder = sample_size % len(dialogues_by_length)
            
            for i, (layer, layer_data) in enumerate(dialogues_by_length.items()):
                layer_sample_size = samples_per_layer + (1 if i < remainder else 0)
                if layer_sample_size > len(layer_data):
                    layer_sample_size = len(layer_data)
                sampled_dialogues.extend(random.sample(layer_data, layer_sample_size))
        else:
            # 默认使用随机采样
            sampled_dialogues = random.sample(all_dialogues, sample_size)
        
        # 转换为验证数据集格式
        validation_samples = []
        for idx, dialogue_text in enumerate(sampled_dialogues):
            validation_sample = {
                "prompt": dialogue_text,
                "data_source": "sales_rag",
                "reward_model": {"ground_truth": {}},
                "metadata": {
                    "source_file": str(self.excel_path),
                    "sample_index": idx,
                    "sampling_method": method,
                    "sampling_timestamp": pd.Timestamp.now().isoformat()
                }
            }
            validation_samples.append(validation_sample)
        
        logger.info(f"成功构建 {len(validation_samples)} 个验证样本 (方法: {method})")
        return validation_samples
    
    def split_train_val(self, train_ratio: float = 0.8, val_ratio: float = 0.2,
                       seed: int = 42) -> tuple:
        """
        将数据集分割为训练集和验证集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            
        Returns:
            tuple: (训练样本列表, 验证样本列表)
        """
        if self.data is None:
            self.read_excel()
        
        # 设置随机种子
        random.seed(seed)
        
        # 获取所有有效数据索引
        valid_indices = []
        for idx, row in self.data.iterrows():
            dialogue_text = row['历史传参上下文']
            if pd.notna(dialogue_text) and str(dialogue_text).strip():
                valid_indices.append(idx)
        
        if not valid_indices:
            logger.warning("没有可用的数据")
            return [], []
        
        # 随机打乱索引
        random.shuffle(valid_indices)
        
        # 计算分割点
        total_size = len(valid_indices)
        train_size = int(total_size * train_ratio)
        
        # 分割数据
        train_indices = valid_indices[:train_size]
        val_indices = valid_indices[train_size:]
        
        logger.info(f"数据集分割完成 - 训练集: {len(train_indices)}, 验证集: {len(val_indices)}")
        return train_indices, val_indices


def main():
    """主函数 - 默认同时生成训练集和验证集"""
    parser = argparse.ArgumentParser(description='Excel训练数据处理 - 一键生成训练数据和验证数据集')
    parser.add_argument('--input', '-i', type=str, required=True, help='Excel文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出文件路径（训练数据）')
    parser.add_argument('--template', '-t', type=str, required=True, help='Prompt模板文件路径')
    
    # 验证数据集参数（现在有了合理的默认值）
    parser.add_argument('--val-output', '-vo', type=str, help='验证数据集输出文件路径（默认: 自动生成）')
    parser.add_argument('--val-size', '-vs', type=int, default=100, help='验证集大小 (默认: 100)')
    parser.add_argument('--val-method', '-vm', type=str, default='random',
                       choices=['random', 'stratified'], help='验证集采样方法 (默认: random)')
    parser.add_argument('--split-ratio', '-sr', type=float, nargs=2,
                       help='训练集和验证集分割比例，例如: 0.8 0.2')
    parser.add_argument('--no-val', action='store_true', help='不生成验证数据集')
    
    parser.add_argument('--statistics', '-s', action='store_true', help='显示统计信息')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建数据处理器
        processor = ExcelDataProcessor(args.input)
        
        # 显示统计信息
        if args.statistics:
            stats = processor.get_statistics()
            print("数据统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        
        # 生成训练样本
        samples = processor.generate_training_samples(args.template)
        
        # 保存训练数据
        processor.save_training_samples(args.output, samples)
        print(f"训练样本已保存到: {args.output}")
        print(f"成功处理 {len(samples)} 个训练样本")
        
        # 默认生成验证数据集（除非明确指定 --no-val）
        if not args.no_val:
            print("\n" + "="*50)
            print("开始自动构建验证数据集...")
            
            # 自动生成验证集输出路径
            if not args.val_output:
                if args.output.endswith('.jsonl'):
                    args.val_output = args.output.replace('.jsonl', '_val.jsonl')
                elif args.output.endswith('.json'):
                    args.val_output = args.output.replace('.json', '_val.json')
                else:
                    args.val_output = args.output + '_val.jsonl'
            
            if args.split_ratio:
                # 使用分割比例方式
                if len(args.split_ratio) != 2:
                    raise ValueError("请提供两个比例值: 训练集比例 验证集比例")
                
                train_ratio, val_ratio = args.split_ratio
                if abs(train_ratio + val_ratio - 1.0) > 0.01:
                    logger.warning(f"比例之和不等于1: {train_ratio} + {val_ratio} = {train_ratio + val_ratio}")
                
                train_indices, val_indices = processor.split_train_val(train_ratio, val_ratio, args.seed)
                
                # 构建验证样本
                val_samples = []
                for idx in val_indices[:min(len(val_indices), args.val_size)]:
                    row = processor.data.iloc[idx]
                    dialogue_text = row['历史传参上下文']
                    if pd.notna(dialogue_text) and str(dialogue_text).strip():
                        val_sample = {
                            "prompt": str(dialogue_text).strip(),
                            "data_source": "sales_rag",
                            "reward_model": {"ground_truth": {}},
                            "metadata": {
                                "source_file": str(processor.excel_path),
                                "sample_index": idx,
                                "sampling_method": "split_ratio",
                                "split_ratio": f"{train_ratio}:{val_ratio}"
                            }
                        }
                        val_samples.append(val_sample)
                
                if val_samples:
                    processor.save_training_samples(args.val_output, val_samples)
                    print(f"验证数据集已保存到: {args.val_output}")
                    print(f"验证样本数量: {len(val_samples)}")
                
            else:
                # 默认使用随机采样方式
                val_samples = processor.build_validation_dataset(
                    sample_size=args.val_size,
                    method=args.val_method,
                    seed=args.seed
                )
                
                if val_samples:
                    processor.save_training_samples(args.val_output, val_samples)
                    print(f"验证数据集已保存到: {args.val_output}")
                    print(f"验证样本数量: {len(val_samples)}")
                    print(f"采样方法: {args.val_method}")
        
        print("\n" + "="*50)
        print("数据处理完成！")
        print(f"  输入文件: {args.input}")
        print(f"  训练数据: {args.output} ({len(samples)} 样本)")
        if not args.no_val and 'val_samples' in locals() and val_samples:
            print(f"  验证数据: {args.val_output} ({len(val_samples)} 样本)")
        elif args.no_val:
            print("  验证数据: 未生成 (使用 --no-val 参数)")
        elif args.split_ratio and 'val_samples' in locals():
            val_output_path = args.val_output or args.output.replace('.jsonl', '_val.jsonl').replace('.json', '_val.json')
            print(f"  验证数据: {val_output_path} ({len(val_samples)} 样本)")
            
    except Exception as e:
        logger.error(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())