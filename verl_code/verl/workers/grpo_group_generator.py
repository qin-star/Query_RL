"""
GRPO组生成器 - 确保正确的组内相对优化
基于官方GRPO实现，生成用于组内比较的多个响应
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GRPOGroup:
    """GRPO组数据结构"""
    group_id: str
    prompt: str
    original_query: str
    history_chat: List[Dict[str, str]]
    samples: List[Dict[str, Any]]  # 组内多个响应样本
    metadata: Dict[str, Any]


class GRPOGroupGenerator:
    """GRPO组生成器 - 生成用于组内相对优化的样本组"""
    
    def __init__(self, group_size: int = 5, temperature_range: tuple = (0.6, 0.9)):
        """
        初始化GRPO组生成器
        
        Args:
            group_size: 每组生成的响应数量
            temperature_range: 温度采样范围，用于生成多样化的响应
        """
        self.group_size = group_size
        self.temperature_range = temperature_range
        self._group_counter = 0
    
    def generate_groups(self, batch_data: List[Dict[str, Any]]) -> List[GRPOGroup]:
        """
        从批次数据中生成GRPO组
        
        Args:
            batch_data: 批次数据，每个元素包含查询信息
            
        Returns:
            List[GRPOGroup]: GRPO组列表
        """
        groups = []
        
        for item in batch_data:
            try:
                group = self._create_single_group(item)
                groups.append(group)
            except Exception as e:
                logger.error(f"生成GRPO组失败: {e}, 数据: {item}")
                continue
        
        logger.info(f"成功生成 {len(groups)} 个GRPO组，每组 {self.group_size} 个样本")
        return groups
    
    def _create_single_group(self, item: Dict[str, Any]) -> GRPOGroup:
        """创建单个GRPO组"""
        
        # 提取基础信息
        original_query = item.get("original_query", "")
        history_chat = item.get("history_chat", [])
        
        # 构建统一的prompt（用于生成多样化响应）
        prompt = self._build_group_prompt(original_query, history_chat)
        
        # 生成组ID
        group_id = f"group_{self._group_counter}"
        self._group_counter += 1
        
        # 创建组结构
        group = GRPOGroup(
            group_id=group_id,
            prompt=prompt,
            original_query=original_query,
            history_chat=history_chat,
            samples=[],
            metadata={
                "source_file": item.get("source_file", ""),
                "timestamp": item.get("timestamp", ""),
                "user_id": item.get("user_id", ""),
                "product_info": item.get("product_info", {})
            }
        )
        
        # 为组内每个样本生成不同的生成参数（确保多样性）
        for sample_idx in range(self.group_size):
            sample_params = self._generate_sample_parameters(sample_idx)
            
            sample = {
                "sample_id": f"{group_id}_sample_{sample_idx}",
                "group_id": group_id,
                "prompt": prompt,
                "original_query": original_query,
                "history_chat": history_chat,
                "generation_params": sample_params,
                "metadata": {
                    "temperature": sample_params["temperature"],
                    "top_p": sample_params["top_p"],
                    "sample_index": sample_idx
                }
            }
            
            group.samples.append(sample)
        
        return group
    
    def _build_group_prompt(self, original_query: str, history_chat: List[Dict[str, str]]) -> str:
        """
        构建用于生成多样化响应的统一prompt
        
        Args:
            original_query: 原始查询
            history_chat: 历史对话
            
        Returns:
            str: 统一prompt
        """
        # 构建包含历史对话的完整上下文
        context_parts = []
        
        # 添加历史对话
        if history_chat:
            for turn in history_chat[-3:]:  # 只保留最近3轮对话
                if isinstance(turn, dict):
                    user_msg = turn.get("user", "")
                    assistant_msg = turn.get("assistant", "")
                    if user_msg:
                        context_parts.append(f"用户: {user_msg}")
                    if assistant_msg:
                        context_parts.append(f"助手: {assistant_msg}")
        
        # 添加当前查询
        context_parts.append(f"用户: {original_query}")
        context_parts.append("助手: 我来帮您优化查询并检索相关信息。")
        
        return "\n".join(context_parts)
    
    def _generate_sample_parameters(self, sample_idx: int) -> Dict[str, float]:
        """
        为组内不同样本生成不同的采样参数（确保多样性）
        
        Args:
            sample_idx: 样本索引
            
        Returns:
            Dict[str, float]: 生成参数
        """
        # 在温度范围内线性插值
        temp_min, temp_max = self.temperature_range
        temperature = temp_min + (temp_max - temp_min) * (sample_idx / max(1, self.group_size - 1))
        
        # 为每个样本生成略有不同的top_p值
        base_top_p = 0.9
        top_p_variation = 0.05 * (sample_idx - self.group_size // 2) / max(1, self.group_size // 2)
        top_p = max(0.7, min(1.0, base_top_p + top_p_variation))
        
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "do_sample": True
        }
    
    def validate_groups(self, groups: List[GRPOGroup]) -> List[GRPOGroup]:
        """
        验证GRPO组的有效性
        
        Args:
            groups: GRPO组列表
            
        Returns:
            List[GRPOGroup]: 有效的组列表
        """
        valid_groups = []
        
        for group in groups:
            try:
                # 检查基本字段
                if not group.prompt or not group.original_query:
                    logger.warning(f"组 {group.group_id} 缺少必要字段")
                    continue
                
                # 检查样本数量
                if len(group.samples) != self.group_size:
                    logger.warning(f"组 {group.group_id} 样本数量不匹配: {len(group.samples)} != {self.group_size}")
                    continue
                
                # 检查样本多样性（确保生成参数不同）
                temperatures = [sample["generation_params"]["temperature"] for sample in group.samples]
                if max(temperatures) - min(temperatures) < 0.05:
                    logger.warning(f"组 {group.group_id} 温度参数差异过小，可能影响多样性")
                
                valid_groups.append(group)
                
            except Exception as e:
                logger.error(f"验证组 {group.group_id} 失败: {e}")
                continue
        
        logger.info(f"验证通过 {len(valid_groups)}/{len(groups)} 个GRPO组")
        return valid_groups


# 工具函数：将GRPO组转换为训练批次格式
def groups_to_training_batch(groups: List[GRPOGroup]) -> List[Dict[str, Any]]:
    """
    将GRPO组转换为训练批次格式
    
    Args:
        groups: GRPO组列表
        
    Returns:
        List[Dict[str, Any]]: 训练批次数据
    """
    batch_data = []
    
    for group in groups:
        for sample in group.samples:
            # 为每个样本创建训练数据项
            training_item = {
                "prompt": sample["prompt"],
                "original_query": sample["original_query"],
                "history_chat": sample["history_chat"],
                "group_id": group.group_id,
                "sample_id": sample["sample_id"],
                "generation_params": sample["generation_params"],
                "metadata": {
                    **group.metadata,
                    **sample["metadata"]
                }
            }
            batch_data.append(training_item)
    
    return batch_data


# 工具函数：从训练结果重建GRPO组
def rebuild_groups_from_results(
    training_results: List[Dict[str, Any]], 
    group_id_key: str = "group_id"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从训练结果重建GRPO组（用于奖励计算）
    
    Args:
        training_results: 训练结果列表
        group_id_key: 组ID字段名
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: 按组ID分组的训练结果
    """
    groups_dict = {}
    
    for result in training_results:
        group_id = result.get(group_id_key)
        if not group_id:
            logger.warning(f"训练结果缺少组ID: {result}")
            continue
        
        if group_id not in groups_dict:
            groups_dict[group_id] = []
        
        groups_dict[group_id].append(result)
    
    # 验证每组样本数量的一致性
    group_sizes = [len(samples) for samples in groups_dict.values()]
    if len(set(group_sizes)) > 1:
        logger.warning(f"检测到不一致的组大小: {set(group_sizes)}")
    
    logger.info(f"重建了 {len(groups_dict)} 个GRPO组，平均组大小: {np.mean(group_sizes):.1f}")
    
    return groups_dict