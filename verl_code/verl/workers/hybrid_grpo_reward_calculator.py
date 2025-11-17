"""
混合GRPO奖励计算器 v3.0
结合GRPO组内相对优势 + 双模型对比质量信号
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import time
from dataclasses import dataclass
from math import tanh
import numpy as np
import torch
from collections import defaultdict
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回到verl_code目录
src_path = os.path.join(project_root, '..', 'src')  # 指向 /home/jovyan2/query_rl/src

if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from .gpt5_dual_model_rater import GPT5DualModelRater
    from src.utils.log import logger
except ImportError as e:
    print(f"⚠️  导入src模块失败: {e}")
    print(f"   当前工作目录: {os.getcwd()}")
    print(f"   脚本目录: {current_dir}")
    print(f"   项目根目录: {project_root}")
    print(f"   src路径: {src_path}")
    print(f"   Python路径: {sys.path[:3]}...")
    
    # 创建模拟类以避免崩溃
    class GPT5DualModelRater:
        def __init__(self, *args, **kwargs):
            pass
        
        def rate_responses(self, *args, **kwargs):
            return {"overall_score": 0.5}
    
    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    
    logger = MockLogger()

logger = logging.getLogger(__name__)


@dataclass
class GPT5ScoringResult:
    """GPT-5评分结果"""
    better: str = "same"
    reason: str = ""
    score: dict = None
    brief: str = ""
    success: bool = True
    error_message: str = ""


@dataclass
class HybridRewardConfig:
    """混合奖励配置"""
    grpo_weight: float = 0.7  # GRPO主权重
    auxiliary_weight: float = 0.3  # 辅助奖励权重
    weight_decay_rate: float = 0.4  # 权重衰减率
    min_auxiliary_weight: float = 0.1  # 最小辅助权重
    group_size: int = 5  # GRPO组大小
    enable_dynamic_weight: bool = True  # 启用动态权重


class HybridGRPORewardCalculator:
    """混合GRPO奖励计算器"""
    
    def __init__(self, scoring_model: str = "GPT-5", config: Optional[HybridRewardConfig] = None):
        """
        初始化混合GRPO奖励计算器
        
        Args:
            scoring_model: 评分模型名称
            config: 混合奖励配置
        """
        self.scoring_model = scoring_model
        self.config = config or HybridRewardConfig()
        self.training_progress = 0.0  # 训练进度 0-1
        self._current_aux_weight = self.config.auxiliary_weight
        
    def update_training_progress(self, progress: float):
        """
        更新训练进度，用于动态权重调整
        
        Args:
            progress: 训练进度 (0-1)
        """
        self.training_progress = max(0.0, min(1.0, progress))
        
        if self.config.enable_dynamic_weight:
            # 动态权重：训练初期辅助权重高，后期以GRPO为主
            # 从初始权重衰减到最小权重
            decay_amount = self.config.weight_decay_rate * self.training_progress
            self._current_aux_weight = max(
                self.config.min_auxiliary_weight,
                self.config.auxiliary_weight - decay_amount
            )
    
    def get_current_weights(self) -> Tuple[float, float]:
        """
        获取当前权重配置
        
        Returns:
            Tuple[float, float]: (grpo权重, 辅助权重)
        """
        aux_weight = self._current_aux_weight
        grpo_weight = 1.0 - aux_weight
        return grpo_weight, aux_weight
    
    async def compute_group_rewards(
        self,
        group_samples: List[Dict],
        gpt5_scores: Optional[List[float]] = None
    ) -> List[float]:
        """
        计算组内多个样本的混合奖励（修正版）
        
        Args:
            group_samples: 组内样本列表，每个样本包含GRPO结果
            gpt5_scores: GPT-5质量评分列表（可选）
            
        Returns:
            list: 混合奖励分数列表
        """
        try:
            if not group_samples:
                return []
            
            # 1. 计算GRPO主奖励（组内相对优势）- 核心GRPO逻辑
            grpo_rewards = self._compute_grpo_rewards(group_samples)
            
            # 2. 计算辅助奖励（GPT-5质量评估，组内中心化）
            auxiliary_rewards = self._compute_gpt5_auxiliary_rewards(
                group_samples, gpt5_scores
            )
            
            # 3. 获取当前权重
            grpo_weight, aux_weight = self.get_current_weights()
            
            # 4. 合成混合奖励（保持GRPO的零均值特性）
            mixed_rewards = []
            for grpo_rew, aux_rew in zip(grpo_rewards, auxiliary_rewards):
                final_reward = grpo_weight * grpo_rew + aux_weight * aux_rew
                mixed_rewards.append(final_reward)
            
            logger.info(f"混合奖励计算完成 - GRPO权重: {grpo_weight:.3f}, "
                       f"辅助权重: {aux_weight:.3f}, 奖励范围: [{min(mixed_rewards):.3f}, {max(mixed_rewards):.3f}]")
            
            return mixed_rewards
            
        except Exception as e:
            logger.error(f"混合奖励计算失败: {e}")
            # 失败时退回到纯GRPO奖励
            return self._compute_grpo_rewards(group_samples)
    
    def _compute_grpo_rewards(self, group_samples: List[Dict]) -> List[float]:
        """
        计算GRPO组内相对优势
        
        Args:
            group_samples: 组内样本列表
            
        Returns:
            list: GRPO相对优势列表
        """
        try:
            # 提取组内分数
            group_scores = []
            for sample in group_samples:
                # 从样本中提取分数，支持多种格式
                if 'grpo_score' in sample:
                    score = sample['grpo_score']
                elif 'token_level_scores' in sample:
                    # 从token级别分数计算总分数
                    scores = sample['token_level_scores']
                    if isinstance(scores, (list, np.ndarray)):
                        score = float(np.mean(scores))
                    else:
                        score = float(scores)
                elif 'reward' in sample:
                    score = float(sample['reward'])
                else:
                    # 默认分数
                    score = 0.5
                
                group_scores.append(score)
            
            if not group_scores or len(group_scores) < 2:
                logger.warning("组内样本不足，无法计算相对优势")
                return [0.0] * len(group_samples)
            
            # 计算组内统计
            mean_score = np.mean(group_scores)
            std_score = np.std(group_scores)
            
            # 处理标准差过小的情况
            if std_score < 1e-8:
                logger.warning("组内分数差异过小，返回零优势")
                return [0.0] * len(group_samples)
            
            # 计算相对优势（零均值化）
            advantages = [(score - mean_score) / std_score for score in group_scores]
            
            logger.debug(f"GRPO相对优势计算 - 均值: {mean_score:.4f}, 标准差: {std_score:.4f}, "
                        f"优势范围: [{min(advantages):.3f}, {max(advantages):.3f}]")
            
            return advantages
            
        except Exception as e:
            logger.error(f"GRPO相对优势计算失败: {e}")
            return [0.0] * len(group_samples)
    
    def _compute_gpt5_auxiliary_rewards(
        self,
        group_samples: List[Dict],
        gpt5_scores: Optional[List[float]]
    ) -> List[float]:
        """
        计算GPT-5辅助奖励（组内中心化，保持零均值特性）
        
        Args:
            group_samples: 组内样本列表
            gpt5_scores: GPT-5质量评分列表
            
        Returns:
            list: 中心化后的GPT-5辅助奖励列表
        """
        try:
            if not gpt5_scores or len(gpt5_scores) != len(group_samples):
                logger.debug("无GPT-5评分或数量不匹配，使用零辅助奖励")
                return [0.0] * len(group_samples)
            
            # 将GPT-5分数转换为numpy数组
            gpt5_array = np.array(gpt5_scores, dtype=np.float32)
            
            # 组内中心化（保持与GRPO相同的零均值特性）
            if len(gpt5_array) > 1:
                mean_gpt5 = np.mean(gpt5_array)
                std_gpt5 = np.std(gpt5_array)
                
                if std_gpt5 < 1e-8:
                    # 如果标准差太小，直接中心化
                    centered_scores = gpt5_array - mean_gpt5
                else:
                    # 标准化到[-0.5, 0.5]范围
                    centered_scores = (gpt5_array - mean_gpt5) / (std_gpt5 + 1e-8) * 0.5
            else:
                # 单样本组，返回零（无相对信息）
                centered_scores = np.zeros_like(gpt5_array)
            
            logger.debug(f"GPT-5辅助奖励中心化 - 原始范围: [{gpt5_array.min():.1f}, {gpt5_array.max():.1f}], "
                        f"中心化后范围: [{centered_scores.min():.3f}, {centered_scores.max():.3f}]")
            
            return centered_scores.tolist()
            
        except Exception as e:
            logger.error(f"GPT-5辅助奖励计算失败: {e}")
            return [0.0] * len(group_samples)
    
    async def _compute_single_auxiliary_reward(
        self, 
        sample_result: Dict, 
        baseline_result: Dict
    ) -> float:
        """
        计算单个样本的辅助奖励
        
        Args:
            sample_result: 样本结果（Actor模型）
            baseline_result: 基线结果（参考模型）
            
        Returns:
            float: 辅助奖励分数
        """
        try:
            # 提取Actor模型数据
            actor_data = sample_result.get("complete_response", {})
            
            # 提取参考模型数据
            reference_data = baseline_result.get("complete_response", {})
            
            # 如果没有完整数据，尝试其他字段
            if not actor_data or not reference_data:
                actor_score = self._extract_score_from_result(sample_result)
                ref_score = self._extract_score_from_result(baseline_result)
            else:
                # 从完整响应中提取分数
                actor_score = self._extract_score_from_response(actor_data)
                ref_score = self._extract_score_from_response(reference_data)
            
            # 计算质量差异
            quality_diff = actor_score - ref_score
            
            # 映射到合理范围 [-0.3, 0.3]
            # 使用tanh函数平滑映射
            auxiliary_reward = np.tanh(quality_diff * 2.0) * 0.3
            
            logger.debug(f"辅助奖励: Actor={actor_score:.3f}, Ref={ref_score:.3f}, "
                        f"Diff={quality_diff:.3f}, Reward={auxiliary_reward:.3f}")
            
            return auxiliary_reward
            
        except Exception as e:
            logger.error(f"单个辅助奖励计算失败: {e}")
            return 0.0
    
    def _extract_score_from_result(self, result: Dict) -> float:
        """
        从结果字典中提取分数
        
        Args:
            result: 结果字典
            
        Returns:
            float: 提取的分数
        """
        # 尝试多种可能的字段
        possible_fields = ['score', 'reward', 'quality_score', 'overall_score']
        
        for field in possible_fields:
            if field in result:
                try:
                    return float(result[field])
                except (ValueError, TypeError):
                    continue
        
        # 默认值
        return 0.5
    
    def _extract_score_from_response(self, response_data: Dict) -> float:
        """
        从GPT-5评分结果中提取分数 - 完全依赖GPT-5评估
        
        Args:
            response_data: GPT-5评分结果字典
            
        Returns:
            float: 提取的分数（0-1范围）
        """
        try:
            # 从GPT-5评分结果中提取，完全依赖GPT-5的评估
            if 'score' in response_data and isinstance(response_data['score'], dict):
                score_dict = response_data['score']
                
                # 检查是否有总分
                if 'sum' in score_dict:
                    return float(score_dict['sum']) / 10.0  # GPT-5评分是0-10分制
                
                # 检查是否有各维度分数
                if 'scores' in score_dict and isinstance(score_dict['scores'], list):
                    scores = score_dict['scores']
                    if scores:
                        return float(np.mean(scores)) / 10.0
                
                # 检查是否有32b和8b的评分对比
                if '32b' in score_dict and '8b' in score_dict:
                    # 提取32b的分数作为参考基准
                    ref_scores = score_dict['32b'].get('scores', [5, 5, 5, 5])
                    ref_sum = score_dict['32b'].get('sum', 20)
                    return float(ref_sum) / 10.0
            
            # 如果没有GPT-5评分，返回中性值
            return 0.5
            
        except Exception as e:
            logger.warning(f"从GPT-5评分结果中提取分数失败: {e}，使用默认值")
            return 0.5
    
    async def _call_gpt5_scoring(self, actor_result: Dict, reference_result: Dict) -> GPT5ScoringResult:
        """
        调用GPT-5双模型评分（用于辅助奖励计算）
        
        Args:
            actor_result: Actor模型结果
            reference_result: 参考模型结果
            
        Returns:
            GPT5ScoringResult: 评分结果
        """
        try:
            logger.debug("开始GPT-5双模型评分（辅助奖励）")
            
            # 构建评分payload
            actor_response = actor_result.get("complete_response", {})
            reference_response = reference_result.get("complete_response", {})
            
            # 添加原始对话数据
            original_data = actor_result.get("original_data", {})
            actor_response["history_chat"] = original_data.get("history_chat", "")
            reference_response["history_chat"] = original_data.get("history_chat", "")
            
            # 创建评分器
            rater = GPT5DualModelRater(
                llm=self.scoring_model,
                actor_response=actor_response,
                reference_response=reference_response
            )
            
            # 调用评分
            scoring_result = await rater.rate()
            
            # 转换为GPT5ScoringResult格式
            if isinstance(scoring_result, dict):
                # 构建标准格式的score数据
                score_data = {
                    "32b": {
                        "scores": [int(scoring_result.get("quality_improvement", 0.5) * 10),
                                  int(scoring_result.get("relevance_accuracy", 0.5) * 10),
                                  int(scoring_result.get("info_completeness", 0.5) * 10),
                                  int(scoring_result.get("retrieval_effectiveness", 0.5) * 10)],
                        "sum": int(scoring_result.get("overall_score", 0.5) * 10)
                    },
                    "8b": {
                        "scores": [int(scoring_result.get("quality_improvement", 0.5) * 10),
                                 int(scoring_result.get("relevance_accuracy", 0.5) * 10),
                                 int(scoring_result.get("info_completeness", 0.5) * 10),
                                 int(scoring_result.get("retrieval_effectiveness", 0.5) * 10)],
                        "sum": int(scoring_result.get("overall_score", 0.5) * 10)
                    }
                }
                
                return GPT5ScoringResult(
                    better=scoring_result.get("better_model", "same"),
                    reason=scoring_result.get("analysis", ""),
                    score=score_data,
                    brief=scoring_result.get("analysis", ""),
                    success=True
                )
            else:
                return GPT5ScoringResult(
                    success=False,
                    error_message="评分结果格式错误"
                )
                
        except Exception as e:
            logger.error(f"GPT-5双模型评分失败: {e}")
            return GPT5ScoringResult(
                success=False,
                error_message=str(e)
            )
    
    def get_reward_statistics(self, rewards: List[float]) -> Dict[str, float]:
        """
        获取奖励统计信息
        
        Args:
            rewards: 奖励分数列表
            
        Returns:
            dict: 统计信息
        """
        try:
            if not rewards:
                return {}
            
            rewards_array = np.array(rewards)
            
            statistics = {
                "mean": float(np.mean(rewards_array)),
                "std": float(np.std(rewards_array)),
                "min": float(np.min(rewards_array)),
                "max": float(np.max(rewards_array)),
                "median": float(np.median(rewards_array)),
                "positive_ratio": float(np.mean(rewards_array > 0)),
                "negative_ratio": float(np.mean(rewards_array < 0)),
                "zero_ratio": float(np.mean(rewards_array == 0)),
                "grpo_weight": self.get_current_weights()[0],
                "auxiliary_weight": self.get_current_weights()[1],
                "training_progress": self.training_progress
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"奖励统计失败: {e}")
            return {}
    
    def _get_group_length(self, group) -> int:
        """
        获取组的长度，处理DataProtoItem等特殊情况
        
        Args:
            group: 组数据，可能是List、DataProtoItem等
            
        Returns:
            int: 组的长度
        """
        try:
            # 如果是普通的list，直接返回长度
            if isinstance(group, list):
                return len(group)
            
            # 如果是DataProtoItem，从batch或non_tensor_batch中获取长度
            if hasattr(group, 'batch') and hasattr(group, 'non_tensor_batch'):
                # 优先从batch中获取长度
                if group.batch is not None:
                    try:
                        if hasattr(group.batch, 'batch_size'):
                            batch_size = group.batch.batch_size
                            if isinstance(batch_size, (list, tuple)):
                                # 如果batch_size是tuple或list，返回第一个元素
                                return int(batch_size[0]) if len(batch_size) > 0 else 1
                            elif isinstance(batch_size, (int, np.integer)):
                                return int(batch_size)
                            else:
                                return 1
                        else:
                            return 1
                    except (IndexError, AttributeError, TypeError) as e:
                        logger.warning(f"从batch获取长度失败: {e}，使用默认值1")
                        return 1
                
                # 从non_tensor_batch中获取长度
                elif group.non_tensor_batch and len(group.non_tensor_batch) > 0:
                    try:
                        random_key = list(group.non_tensor_batch.keys())[0]
                        data = group.non_tensor_batch[random_key]
                        
                        if hasattr(data, 'shape'):
                            # 处理numpy数组或tensor的shape
                            shape = data.shape
                            if isinstance(shape, (tuple, list)) and len(shape) > 0:
                                return int(shape[0])
                            else:
                                return 1
                        elif isinstance(data, (list, tuple)):
                            return len(data)
                        elif hasattr(data, '__len__'):
                            return len(data)
                        else:
                            return 1
                    except (IndexError, AttributeError, TypeError) as e:
                        logger.warning(f"从non_tensor_batch获取长度失败: {e}，使用默认值1")
                        return 1
                else:
                    return 1
            
            # 如果是DataProto，使用其__len__方法
            if hasattr(group, '__len__'):
                try:
                    return len(group)
                except Exception as e:
                    logger.warning(f"使用__len__方法获取长度失败: {e}，使用默认值1")
                    return 1
            
            # 其他情况，尝试转换为list
            try:
                return len(list(group))
            except Exception as e:
                # 如果都无法处理，返回1作为默认值
                logger.warning(f"无法确定组长度: {e}，使用默认值1。组类型: {type(group)}")
                return 1
                
        except Exception as e:
            logger.error(f"获取组长度失败: {e}，使用默认值1")
            return 1
    
    def _convert_group_to_list(self, group) -> List[Dict]:
        """
        将组数据转换为标准的List[Dict]格式
        
        Args:
            group: 组数据，可能是List、DataProtoItem等
            
        Returns:
            List[Dict]: 标准格式的组数据
        """
        try:
            # 如果已经是list，直接返回
            if isinstance(group, list):
                return group
            
            # 如果是DataProtoItem，提取数据
            if hasattr(group, 'batch') and hasattr(group, 'non_tensor_batch'):
                result_list = []
                group_length = self._get_group_length(group)
                
                for i in range(group_length):
                    item_dict = {}
                    
                    # 从batch中提取数据
                    if group.batch is not None:
                        try:
                            if hasattr(group.batch, 'select_idxs'):
                                item_data = group.batch.select_idxs([i])
                                if hasattr(item_data, '__dict__'):
                                    item_dict.update(item_data.__dict__)
                                else:
                                    item_dict.update(item_data)
                            else:
                                # 安全地访问batch元素
                                try:
                                    item_data = group.batch[i] if i < len(group.batch) else {}
                                    if hasattr(item_data, '__dict__'):
                                        item_dict.update(item_data.__dict__)
                                    else:
                                        item_dict.update(item_data)
                                except (IndexError, TypeError, AttributeError) as e:
                                    logger.warning(f"访问batch[{i}]失败: {e}")
                                    item_dict.update({})
                        except Exception as e:
                            logger.warning(f"从batch提取数据失败: {e}")
                            item_dict.update({})
                    
                    # 从non_tensor_batch中提取数据
                    try:
                        for key, value in group.non_tensor_batch.items():
                            if hasattr(value, '__getitem__'):
                                try:
                                    # 检查是否可以索引访问
                                    if hasattr(value, '__len__') and len(value) > i:
                                        item_dict[key] = value[i]
                                    else:
                                        item_dict[key] = value
                                except (IndexError, TypeError, AttributeError):
                                    item_dict[key] = value
                            else:
                                item_dict[key] = value
                    except Exception as e:
                        logger.warning(f"从non_tensor_batch提取数据失败: {e}")
                    
                    result_list.append(item_dict)
                
                return result_list
            
            # 如果是DataProto，使用其切片功能
            if hasattr(group, '__getitem__'):
                group_length = self._get_group_length(group)
                result_list = []
                for i in range(group_length):
                    try:
                        item = group[i]
                        if hasattr(item, 'batch') and hasattr(item, 'non_tensor_batch'):
                            # 将DataProtoItem转换为dict
                            item_dict = {}
                            if item.batch is not None:
                                try:
                                    if hasattr(item.batch, '__dict__'):
                                        item_dict.update(item.batch.__dict__)
                                    else:
                                        item_dict.update(item.batch)
                                except Exception as e:
                                    logger.warning(f"更新item.batch到dict失败: {e}")
                            try:
                                item_dict.update(item.non_tensor_batch)
                            except Exception as e:
                                logger.warning(f"更新non_tensor_batch到dict失败: {e}")
                            result_list.append(item_dict)
                        else:
                            result_list.append(item)
                    except Exception as e:
                        logger.warning(f"转换组元素{i}失败: {e}")
                        result_list.append({})
                return result_list
            
            # 其他情况，尝试转换为list
            try:
                converted_list = list(group)
                if converted_list:
                    return converted_list
                else:
                    logger.warning(f"转换为空列表，组类型: {type(group)}")
                    return []
            except Exception as e:
                # 如果都无法处理，返回空列表
                logger.error(f"无法转换组数据: {e}，返回空列表。组类型: {type(group)}")
                return []
                
        except Exception as e:
            logger.error(f"转换组数据失败: {e}，返回空列表")
            return []

    async def batch_compute_rewards(
        self,
        grpo_groups: List,
        baseline_groups: Optional[List] = None,
        max_concurrency: int = 3
    ) -> List[List[float]]:
        """
        批量计算混合奖励 - 修复DataProtoItem支持
        
        Args:
            grpo_groups: GRPO组列表，每组可能是List、DataProtoItem等
            baseline_groups: 基线组列表（可选）
            max_concurrency: 最大并发数
            
        Returns:
            List[List[float]]: 每组混合奖励列表
        """
        try:
            logger.info(f"开始批量计算{len(grpo_groups)}个GRPO组的混合奖励")
            
            # 转换所有组为标准格式
            converted_groups = []
            for i, group in enumerate(grpo_groups):
                converted_group = self._convert_group_to_list(group)
                if converted_group:
                    converted_groups.append(converted_group)
                else:
                    logger.warning(f"组{i}转换失败，使用空列表")
                    converted_groups.append([])
            
            # 转换基线组
            converted_baseline_groups = None
            if baseline_groups:
                converted_baseline_groups = []
                for i, group in enumerate(baseline_groups):
                    converted_group = self._convert_group_to_list(group)
                    converted_baseline_groups.append(converted_group)
            
            if len(converted_groups) != len(grpo_groups):
                logger.warning("转换后的组数量与原组数量不匹配")
            
            if baseline_groups and len(converted_groups) != len(converted_baseline_groups or []):
                logger.warning("GRPO组与基线组数量不匹配")
                converted_baseline_groups = None
            
            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def compute_group_with_semaphore(i, group, baseline_group):
                async with semaphore:
                    return await self.compute_group_rewards(group, baseline_group)
            
            # 创建所有任务
            tasks = []
            for i, (group, baseline_group) in enumerate(zip(converted_groups, converted_baseline_groups or [None] * len(converted_groups))):
                task = compute_group_with_semaphore(i, group, baseline_group)
                tasks.append(task)
            
            # 等待所有任务完成
            all_rewards = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_rewards = []
            for i, rewards in enumerate(all_rewards):
                if isinstance(rewards, Exception):
                    logger.error(f"组{i}奖励计算异常: {rewards}")
                    # 返回默认的GRPO奖励
                    group_length = self._get_group_length(grpo_groups[i])
                    processed_rewards.append([0.0] * group_length)
                else:
                    processed_rewards.append(rewards)
            
            logger.info(f"批量混合奖励计算完成，成功: {len(processed_rewards)}个组")
            return processed_rewards
            
        except Exception as e:
            logger.error(f"批量混合奖励计算失败: {e}")
            # 返回默认奖励
            default_rewards = []
            for group in grpo_groups:
                group_length = self._get_group_length(group)
                default_rewards.append([0.0] * group_length)
            return default_rewards


class HybridRewardManager:
    """混合奖励管理器"""
    
    def __init__(self, scoring_model: str = "GPT-5", config: Optional[HybridRewardConfig] = None):
        """
        初始化混合奖励管理器
        
        Args:
            scoring_model: 评分模型名称
            config: 混合奖励配置
        """
        self.scoring_model = scoring_model
        self.config = config or HybridRewardConfig()
        self.calculator = None
        
    def get_calculator(self) -> HybridGRPORewardCalculator:
        """获取混合奖励计算器实例"""
        if self.calculator is None:
            self.calculator = HybridGRPORewardCalculator(self.scoring_model, self.config)
        return self.calculator
    
    def update_training_progress(self, progress: float):
        """
        更新训练进度
        
        Args:
            progress: 训练进度 (0-1)
        """
        calculator = self.get_calculator()
        calculator.update_training_progress(progress)
    
    async def compute_training_rewards(
        self,
        grpo_groups: List[List[Dict]],
        baseline_groups: Optional[List[List[Dict]]] = None,
        return_dict: bool = False
    ) -> Tuple[List[List[float]], Dict[str, Any]] | Dict[str, Any]:
        """
        计算训练奖励（主接口）
        
        Args:
            grpo_groups: GRPO组列表
            baseline_groups: 基线组列表（可选）
            return_dict: 是否返回字典格式（兼容verl框架）
            
        Returns:
            Tuple[List[List[float]], Dict[str, Any]] | Dict[str, Any]:
                - return_dict=False: (奖励列表, 统计信息)
                - return_dict=True: {"reward_tensor": 奖励张量, "reward_extra_info": 额外信息}
        """
        calculator = self.get_calculator()
        
        # 计算奖励
        all_rewards = await calculator.batch_compute_rewards(
            grpo_groups, baseline_groups
        )
        
        # 如果要求返回字典格式（兼容verl框架）
        if return_dict:
            # 将奖励列表转换为张量格式
            import torch
            
            # 确保所有组的奖励列表长度一致
            if all_rewards:
                # 找到最大长度
                max_length = max(len(group_rewards) for group_rewards in all_rewards)
                
                # 填充或截断所有组到相同长度
                normalized_rewards = []
                for group_rewards in all_rewards:
                    if len(group_rewards) < max_length:
                        # 填充0到最大长度
                        padded_rewards = group_rewards + [0.0] * (max_length - len(group_rewards))
                        normalized_rewards.append(padded_rewards)
                    else:
                        normalized_rewards.append(group_rewards)
                
                reward_tensor = torch.tensor(normalized_rewards, dtype=torch.float32)
            else:
                # 如果没有奖励数据，创建空张量
                reward_tensor = torch.tensor([[]], dtype=torch.float32)
            
            # 收集额外信息 - 确保所有值都是可迭代的列表格式
            reward_extra_info = {}
            if all_rewards:
                # 收集统计信息
                all_stats = []
                for group_rewards in all_rewards:
                    stats = calculator.get_reward_statistics(group_rewards)
                    all_stats.append(stats)
                
                # 汇总统计
                overall_stats = self._aggregate_statistics(all_stats)
                
                # 计算总样本数量（用于确保统计信息长度匹配）
                total_samples = sum(len(group_rewards) for group_rewards in all_rewards)
                
                # 将统计信息转换为列表格式，确保每个值都与样本数量匹配
                for key, value in overall_stats.items():
                    if isinstance(value, (int, float)):
                        # 单个数值需要复制到与样本数量相同的长度
                        reward_extra_info[key] = [value] * total_samples
                    elif isinstance(value, (list, tuple)):
                        # 已经是可迭代类型，但需要确保长度匹配
                        value_list = list(value)
                        if len(value_list) == 1:
                            # 如果只有一个值，复制到所有样本
                            reward_extra_info[key] = value_list * total_samples
                        else:
                            # 多个值的情况，目前也复制第一个值到所有样本
                            reward_extra_info[key] = [value_list[0]] * total_samples
                    else:
                        # 其他类型，尝试转换为列表
                        try:
                            value_list = list(value)
                            if len(value_list) == 1:
                                reward_extra_info[key] = value_list * total_samples
                            else:
                                reward_extra_info[key] = [value_list[0]] * total_samples
                        except:
                            reward_extra_info[key] = [str(value)] * total_samples
            
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        
        # 原始返回格式
        # 收集统计信息
        all_stats = []
        for group_rewards in all_rewards:
            stats = calculator.get_reward_statistics(group_rewards)
            all_stats.append(stats)
        
        # 汇总统计
        overall_stats = self._aggregate_statistics(all_stats)
        
        # 返回奖励列表和统计信息
        return all_rewards, overall_stats
    
    def compute_training_rewards_sync(
        self,
        grpo_groups: List[List[Dict]],
        baseline_groups: Optional[List[List[Dict]]] = None,
        return_dict: bool = False
    ) -> Tuple[List[List[float]], Dict[str, Any]] | Dict[str, Any]:
        """
        计算训练奖励（主接口）- 同步版本，用于验证阶段
        
        Args:
            grpo_groups: GRPO组列表
            baseline_groups: 基线组列表（可选）
            return_dict: 是否返回字典格式（兼容verl框架）
            
        Returns:
            Tuple[List[List[float]], Dict[str, Any]] | Dict[str, Any]:
                - return_dict=False: (奖励列表, 统计信息)
                - return_dict=True: {"reward_tensor": 奖励张量, "reward_extra_info": 额外信息}
        """
        import asyncio
        
        # 创建新的事件循环来运行异步函数
        try:
            # 尝试获取现有的事件循环
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 运行异步函数
        if loop.is_running():
            # 如果事件循环正在运行，我们需要创建任务
            import concurrent.futures
            import threading
            
            # 在新线程中运行事件循环
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self.compute_training_rewards(grpo_groups, baseline_groups, return_dict)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
        else:
            # 如果事件循环没有运行，直接运行
            return loop.run_until_complete(
                self.compute_training_rewards(grpo_groups, baseline_groups, return_dict)
            )
    
    def _aggregate_statistics(self, all_stats: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        汇总多组统计信息
        
        Args:
            all_stats: 多组统计信息列表
            
        Returns:
            Dict[str, Any]: 汇总统计
        """
        if not all_stats:
            return {}
        
        # 计算平均值
        aggregated = {}
        numeric_keys = ['mean', 'std', 'min', 'max', 'median', 'positive_ratio', 'negative_ratio', 'zero_ratio']
        
        for key in numeric_keys:
            values = [stats.get(key, 0.0) for stats in all_stats if key in stats]
            if values:
                aggregated[f'avg_{key}'] = float(np.mean(values))
        
        # 添加当前权重信息
        calculator = self.get_calculator()
        grpo_weight, aux_weight = calculator.get_current_weights()
        aggregated['current_grpo_weight'] = grpo_weight
        aggregated['current_auxiliary_weight'] = aux_weight
        aggregated['training_progress'] = calculator.training_progress
        
        return aggregated


# 便捷函数
async def compute_hybrid_rewards(
    grpo_groups: List[List[Dict]], 
    baseline_groups: Optional[List[List[Dict]]] = None,
    scoring_model: str = "GPT-5",
    training_progress: float = 0.0
) -> List[List[float]]:
    """
    便捷函数：计算混合奖励
    
    Args:
        grpo_groups: GRPO组列表
        baseline_groups: 基线组列表（可选）
        scoring_model: 评分模型名称
        training_progress: 训练进度
        
    Returns:
        List[List[float]]: 混合奖励列表
    """
    manager = HybridRewardManager(scoring_model)
    manager.update_training_progress(training_progress)
    
    rewards, _ = await manager.compute_training_rewards(grpo_groups, baseline_groups)
    return rewards


if __name__ == "__main__":
    # 测试代码
    async def test_hybrid_reward_calculator():
        # 创建测试数据
        grpo_groups = [
            [
                {"grpo_score": 0.8, "complete_response": {"overall_score": 0.7}},
                {"grpo_score": 0.6, "complete_response": {"overall_score": 0.5}},
                {"grpo_score": 0.9, "complete_response": {"overall_score": 0.8}},
                {"grpo_score": 0.7, "complete_response": {"overall_score": 0.6}},
                {"grpo_score": 0.5, "complete_response": {"overall_score": 0.4}}
            ],
            [
                {"grpo_score": 0.9, "complete_response": {"overall_score": 0.8}},
                {"grpo_score": 0.7, "complete_response": {"overall_score": 0.6}},
                {"grpo_score": 0.8, "complete_response": {"overall_score": 0.7}},
                {"grpo_score": 0.6, "complete_response": {"overall_score": 0.5}},
                {"grpo_score": 0.8, "complete_response": {"overall_score": 0.7}}
            ]
        ]
        
        baseline_groups = [
            [
                {"complete_response": {"overall_score": 0.75}},
                {"complete_response": {"overall_score": 0.75}},
                {"complete_response": {"overall_score": 0.75}},
                {"complete_response": {"overall_score": 0.75}},
                {"complete_response": {"overall_score": 0.75}}
            ],
            [
                {"complete_response": {"overall_score": 0.85}},
                {"complete_response": {"overall_score": 0.85}},
                {"complete_response": {"overall_score": 0.85}},
                {"complete_response": {"overall_score": 0.85}},
                {"complete_response": {"overall_score": 0.85}}
            ]
        ]
        
        # 测试混合奖励计算
        manager = HybridRewardManager()
        manager.update_training_progress(0.5)  # 50%训练进度
        
        rewards, stats = await manager.compute_training_rewards(
            grpo_groups, baseline_groups
        )
        
        print("混合奖励计算结果:")
        for i, group_rewards in enumerate(rewards):
            print(f"组{i}: {group_rewards}")
        
        print("\n统计信息:")
        print(stats)
    
    # 运行测试
    asyncio.run(test_hybrid_reward_calculator())