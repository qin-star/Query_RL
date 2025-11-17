"""
双模型Reward计算器 v2.0
用于计算SalesRAG Query改写GRPO强化学习的奖励分数
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import time
from dataclasses import dataclass
from math import tanh
import numpy as np

from .gpt5_dual_model_rater import GPT5DualModelRater
from src.utils.log import logger

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


class DualModelRewardCalculator:
    """双模型Reward计算器"""
    
    def __init__(self, scoring_model: str = "GPT-5"):
        """
        初始化双模型Reward计算器
        
        Args:
            scoring_model: 评分模型名称
        """
        self.scoring_model = scoring_model
        self.rater = None
        self._init_rater()
        
    def _init_rater(self):
        """初始化评分器"""
        try:
            # 注意：这里不需要预先初始化rater，而是在每次评分时动态创建
            logger.info(f"{self.scoring_model}评分器配置完成")
        except Exception as e:
            logger.error(f"{self.scoring_model}评分器配置失败: {e}")
            raise
    
    async def compute_group_rewards(
        self, 
        group_samples: List[Dict], 
        baseline_result: Dict
    ) -> List[float]:
        """
        计算组内多个样本的rewards
        
        Args:
            group_samples: 组内样本列表
            baseline_result: 基线结果（参考模型）
            
        Returns:
            list: 奖励分数列表
        """
        try:
            group_rewards = []
            
            # 为每个8B样本计算reward
            for sample in group_samples:
                reward = await self._compute_single_reward(
                    sample_result=sample,
                    baseline_result=baseline_result
                )
                group_rewards.append(reward)
            
            logger.info(f"组内reward计算完成，共{len(group_rewards)}个样本")
            return group_rewards
            
        except Exception as e:
            logger.error(f"组内reward计算失败: {e}")
            return [0.0] * len(group_samples)
    
    async def _compute_single_reward(
        self, 
        sample_result: Dict, 
        baseline_result: Dict
    ) -> float:
        """
        计算单个样本的reward
        
        Args:
            sample_result: 8B模型结果
            baseline_result: 32B基线结果
            
        Returns:
            float: 奖励分数
        """
        try:
            # 构建GPT-5评分payload
            payload = self._build_gpt5_payload(sample_result, baseline_result)
            
            # 调用GPT-5评分
            gpt5_result = await self._call_gpt5_scoring(sample_result, baseline_result)
            
            # 计算reward
            reward = self._calculate_reward_from_gpt5_result(gpt5_result)
            
            logger.debug(f"单个reward计算完成: {reward:.4f}")
            return reward
            
        except Exception as e:
            logger.error(f"单个reward计算失败: {e}")
            return 0.0
    
    def _build_gpt5_payload(
        self, 
        sample_result: Dict, 
        baseline_result: Dict
    ) -> Dict[str, Any]:
        """
        构建GPT-5评分payload
        
        Args:
            sample_result: 8B模型结果
            baseline_result: 32B基线结果
            
        Returns:
            dict: GPT-5评分payload
        """
        try:
            # 提取8B模型数据
            actor_data = sample_result.get("complete_response", {})
            
            # 提取32B模型数据
            reference_data = baseline_result.get("complete_response", {})
            
            # 获取原始对话数据
            original_data = sample_result.get("original_data", {})
            
            payload = {
                "history_chat": original_data.get("history_chat", ""),
                
                # 🔥 32B基线数据（来自RAG /chat接口）
                "user_profile": reference_data.get("user_profile", ""),
                "rewritten_query": reference_data.get("rewritten_query", ""),
                "history_summary": reference_data.get("history_summary", ""),
                "rag_recall": reference_data.get("rag_recall", []),
                "rag_status": reference_data.get("rag_status", ""),
                
                # 🔥 8B Actor数据（来自RAG /chat_8b接口）
                "user_profile_8B": actor_data.get("user_profile", ""),
                "rewritten_query_8B": actor_data.get("rewritten_query", ""),
                "history_summary_8B": actor_data.get("history_summary", ""),
                "rag_recall_8B": actor_data.get("rag_recall", []),
                "rag_status_8B": actor_data.get("rag_status", ""),
                
                # 处理元数据
                "processing_metadata": {
                    "actor_endpoint": "/chat_8b",
                    "reference_endpoint": "/chat",
                    "actor_processing_time": actor_data.get("processing_metadata", {}).get("processing_time", 0.0),
                    "reference_processing_time": reference_data.get("processing_metadata", {}).get("processing_time", 0.0)
                }
            }
            
            logger.debug(f"GPT-5 payload构建成功")
            return payload
            
        except Exception as e:
            logger.error(f"GPT-5 payload构建失败: {e}")
            return self._get_default_payload()
    
    def _get_default_payload(self) -> Dict[str, Any]:
        """获取默认payload"""
        return {
            "history_chat": "",
            "user_profile": "",
            "rewritten_query": "",
            "history_summary": "",
            "rag_recall": [],
            "rag_status": "",
            "user_profile_8B": "",
            "rewritten_query_8B": "",
            "history_summary_8B": "",
            "rag_recall_8B": [],
            "rag_status_8B": "",
            "processing_metadata": {
                "actor_endpoint": "/chat_8b",
                "reference_endpoint": "/chat",
                "actor_processing_time": 0.0,
                "reference_processing_time": 0.0
            }
        }
    
    async def _call_gpt5_scoring(self, actor_result: Dict, reference_result: Dict) -> Dict:
        """
        调用GPT-5双模型评分
        
        Args:
            actor_result: 8B模型结果
            reference_result: 32B模型结果
            
        Returns:
            Dict: 评分结果
        """
        try:
            logger.debug("开始GPT-5双模型评分")
            
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
    
    def _calculate_reward_from_gpt5_result(self, gpt5_result: GPT5ScoringResult) -> float:
        """
        根据GPT-5评分结果计算reward
        
        Args:
            gpt5_result: GPT-5评分结果
            
        Returns:
            float: 奖励分数
        """
        try:
            better = gpt5_result.better
            score_data = gpt5_result.score
            
            # 提取分数
            if isinstance(score_data, dict):
                scores_32b = score_data.get("32b", {})
                scores_8b = score_data.get("8b", {})
                
                sum_32b = scores_32b.get("sum", 0) if isinstance(scores_32b, dict) else 0
                sum_8b = scores_8b.get("sum", 0) if isinstance(scores_8b, dict) else 0
            else:
                better = "same"
                sum_32b, sum_8b = 0, 0
            
            # 计算reward
            reward = self._compute_reward(better, sum_8b, sum_32b)
            
            logger.debug(f"Reward计算: better={better}, 8b={sum_8b}, 32b={sum_32b}, reward={reward:.4f}")
            return reward
            
        except Exception as e:
            logger.error(f"Reward计算失败: {e}")
            logger.debug(f"失败详情 - sample_result_keys: {list(sample_result.keys())}, "
                        f"baseline_result_keys: {list(baseline_result.keys()) if baseline_result else 'None'}")
            return 0.0
    
    def _compute_reward(self, better: str, sum_8b: float, sum_32b: float) -> float:
        """
        计算奖励分数
        
        Args:
            better: 哪个模型更好
            sum_8b: 8B模型总分
            sum_32b: 32B模型总分
            
        Returns:
            float: 奖励分数，范围[-1, 1]
        """
        try:
            # 奖励规则
            reward_rules = {
                "8b": lambda r: r + 0.2,
                "32b": lambda r: r - 0.2,
                "same": lambda r: r * 0.5,
                "both bad": lambda r: -0.5
            }
            
            # 计算分数差异
            sum_diff = abs(sum_8b - sum_32b) / 100
            base_reward = tanh(sum_diff * 2)
            
            # 应用奖励规则
            if better in reward_rules:
                base_reward = reward_rules[better](base_reward)
            
            # 限制在[-1, 1]范围内
            final_reward = np.clip(base_reward, -1, 1)
            
            return float(final_reward)
            
        except Exception as e:
            logger.error(f"奖励计算失败: {e}")
            return 0.0
    
    async def batch_compute_rewards(
        self, 
        actor_results: List[Dict], 
        reference_results: List[Dict],
        max_concurrency: int = 3
    ) -> List[float]:
        """
        批量计算奖励
        
        Args:
            actor_results: 8B模型结果列表
            reference_results: 32B模型结果列表
            max_concurrency: 最大并发数
            
        Returns:
            list: 奖励分数列表
        """
        try:
            logger.info(f"开始批量计算{len(actor_results)}个样本的奖励")
            
            if len(actor_results) != len(reference_results):
                raise ValueError("8B和32B结果数量不匹配")
            
            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def compute_with_semaphore(actor_result, reference_result):
                async with semaphore:
                    return await self._compute_single_reward(actor_result, reference_result)
            
            # 创建所有任务
            tasks = []
            for actor_result, reference_result in zip(actor_results, reference_results):
                task = compute_with_semaphore(actor_result, reference_result)
                tasks.append(task)
            
            # 等待所有任务完成
            rewards = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_rewards = []
            for i, reward in enumerate(rewards):
                if isinstance(reward, Exception):
                    logger.error(f"样本{i}奖励计算异常: {reward}")
                    processed_rewards.append(0.0)
                else:
                    processed_rewards.append(reward)
            
            logger.info(f"批量奖励计算完成，成功: {len(processed_rewards)}个")
            return processed_rewards
            
        except Exception as e:
            logger.error(f"批量奖励计算失败: {e}")
            return [0.0] * len(actor_results)
    
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
                "zero_ratio": float(np.mean(rewards_array == 0))
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"奖励统计失败: {e}")
            return {}


class DualModelRewardManager:
    """双模型Reward管理器"""
    
    def __init__(self, scoring_model: str = "GPT-5"):
        """
        初始化双模型Reward管理器
        
        Args:
            scoring_model: 评分模型名称
        """
        self.scoring_model = scoring_model
        self.calculator = None
        
    def get_calculator(self) -> DualModelRewardCalculator:
        """获取双模型Reward计算器实例"""
        if self.calculator is None:
            self.calculator = DualModelRewardCalculator(self.scoring_model)
        return self.calculator
    
    async def compute_training_rewards(
        self, 
        actor_results: List[Dict], 
        reference_results: List[Dict]
    ) -> tuple[List[float], Dict[str, float]]:
        """
        计算训练奖励
        
        Args:
            actor_results: 8B模型结果列表
            reference_results: 32B模型结果列表
            
        Returns:
            tuple: (奖励分数列表, 统计信息)
        """
        calculator = self.get_calculator()
        
        # 计算奖励
        rewards = await calculator.batch_compute_rewards(actor_results, reference_results)
        
        # 获取统计信息
        statistics = calculator.get_reward_statistics(rewards)
        
        return rewards, statistics


if __name__ == "__main__":
    # 示例用法
    async def test_reward_calculator():
        calculator = DualModelRewardCalculator()
        
        # 模拟数据
        actor_result = {
            "complete_response": {
                "user_profile": "应届毕业生",
                "rewritten_query": "公务员考试与省考的区别",
                "history_summary": "询问国考省考区别",
                "rag_recall": ["doc1", "doc2"],
                "rag_status": "success"
            },
            "original_data": {
                "history_chat": "[销售][2024-12-09 16:01:58]:哈喽，你好！",
                "query": "[客户][2024-12-09 16:39:41]:国考和省考有什么区别？"
            }
        }
        
        reference_result = {
            "complete_response": {
                "user_profile": "应届毕业生，目标公务员",
                "rewritten_query": "国家公务员考试与省级公务员考试的区别是什么？",
                "history_summary": "客户询问国考和省考的区别",
                "rag_recall": ["doc1", "doc2", "doc3"],
                "rag_status": "success"
            }
        }
        
        reward = await calculator._compute_single_reward(actor_result, reference_result)
        print(f"计算得到的奖励分数: {reward:.4f}")
    
    # 运行测试
    asyncio.run(test_reward_calculator())