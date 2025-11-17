"""
混合GRPO训练管理器 v3.0
用于协调混合训练流程，管理训练进度和辅助奖励计算
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import time
from collections import defaultdict
import numpy as np
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回到verl_code目录
src_path = os.path.join(project_root, '..', 'src')  # 指向 /home/jovyan2/query_rl/src

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 添加项目根目录到Python路径
project_root_path = os.path.dirname(project_root)  # 指向 /home/jovyan2/query_rl
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

try:
    from .hybrid_grpo_reward_calculator import HybridGRPORewardCalculator, HybridRewardConfig
    from .gpt5_dual_model_rater import GPT5DualModelRater
    from src.core.rag_chater import RagChater
    from src.utils.log import logger
except ImportError as e:
    print(f"⚠️  导入src模块失败: {e}")
    print(f"   当前工作目录: {os.getcwd()}")
    print(f"   脚本目录: {current_dir}")
    print(f"   项目根目录: {project_root}")
    print(f"   src路径: {src_path}")
    print(f"   Python路径: {sys.path[:3]}...")
    
    # 创建模拟类以避免崩溃
    class RagChater:
        def __init__(self, *args, **kwargs):
            pass
        
        def generate_response(self, *args, **kwargs):
            return {"response": "模拟RAG响应"}
    
    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    
    logger = MockLogger()

logger = logging.getLogger(__name__)


@dataclass
class HybridTrainingState:
    """混合训练状态"""
    current_step: int = 0
    total_steps: int = 0
    training_progress: float = 0.0  # 0-1
    current_auxiliary_weight: float = 0.3
    grpo_weight: float = 0.7
    cumulative_stats: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    last_update_time: float = 0.0


class HybridTrainingManager:
    """混合GRPO训练管理器"""
    
    def __init__(
        self,
        enable_auxiliary_reward: bool = True,
        auxiliary_weight: float = 0.3,
        weight_decay_rate: float = 0.4,
        min_auxiliary_weight: float = 0.1,
        scoring_model: str = "GPT-5",
        group_size: int = 5,
        enable_dynamic_weight: bool = True
    ):
        """
        初始化混合训练管理器
        
        Args:
            enable_auxiliary_reward: 是否启用辅助奖励
            auxiliary_weight: 初始辅助奖励权重
            weight_decay_rate: 权重衰减率
            min_auxiliary_weight: 最小辅助权重
            scoring_model: 评分模型名称
            group_size: GRPO组大小
            enable_dynamic_weight: 是否启用动态权重
        """
        self.enable_auxiliary_reward = enable_auxiliary_reward
        self.auxiliary_weight = auxiliary_weight
        self.weight_decay_rate = weight_decay_rate
        self.min_auxiliary_weight = min_auxiliary_weight
        self.scoring_model = scoring_model
        self.group_size = group_size
        self.enable_dynamic_weight = enable_dynamic_weight
        
        # 初始化状态
        self.state = HybridTrainingState()
        self.state.grpo_weight = 1.0 - auxiliary_weight
        self.state.current_auxiliary_weight = auxiliary_weight
        
        # 初始化奖励计算器
        self.reward_calculator = None
        self._init_reward_calculator()
        
        # 初始化RAG客户端（用于辅助奖励计算）
        self.rag_client = None
        if self.enable_auxiliary_reward:
            self._init_rag_client()
    
    def _init_reward_calculator(self):
        """初始化奖励计算器"""
        try:
            # 创建混合奖励配置
            hybrid_config = HybridRewardConfig(
                grpo_weight=self.state.grpo_weight,
                auxiliary_weight=self.state.current_auxiliary_weight,
                weight_decay_rate=self.weight_decay_rate,
                min_auxiliary_weight=self.min_auxiliary_weight,
                group_size=self.group_size,
                enable_dynamic_weight=self.enable_dynamic_weight
            )
            
            # 创建奖励计算器
            self.reward_calculator = HybridGRPORewardCalculator(
                scoring_model=self.scoring_model,
                config=hybrid_config
            )
            
            logger.info("混合奖励计算器初始化成功")
            
        except Exception as e:
            logger.error(f"混合奖励计算器初始化失败: {e}")
            self.enable_auxiliary_reward = False
    
    def _init_rag_client(self):
        """初始化RAG客户端"""
        try:
            self.rag_client = RagChater(
                tenant_id="chengla",
                contact_id="chengla_hybrid_grpo_contact",
                account_id="chengla_hybrid_grpo_account",
                message_id="chengla_hybrid_grpo_message_id"
            )
            logger.info("RAG客户端初始化成功")
            
        except Exception as e:
            logger.error(f"RAG客户端初始化失败: {e}")
            self.rag_client = None
    
    def update_training_progress(self, current_step: int, total_steps: int):
        """
        更新训练进度
        
        Args:
            current_step: 当前训练步数
            total_steps: 总训练步数
        """
        self.state.current_step = current_step
        self.state.total_steps = total_steps
        self.state.training_progress = min(1.0, current_step / max(1, total_steps))
        self.state.last_update_time = time.time()
        
        # 更新奖励计算器的训练进度
        if self.reward_calculator:
            self.reward_calculator.update_training_progress(self.state.training_progress)
            
            # 更新当前权重
            grpo_weight, aux_weight = self.reward_calculator.get_current_weights()
            self.state.grpo_weight = grpo_weight
            self.state.current_auxiliary_weight = aux_weight
        
        logger.info(f"训练进度更新: {current_step}/{total_steps} ({self.state.training_progress:.1%}) - "
                   f"GRPO权重: {self.state.grpo_weight:.3f}, 辅助权重: {self.state.current_auxiliary_weight:.3f}")
    
    def get_current_weights(self) -> Tuple[float, float]:
        """
        获取当前权重配置
        
        Returns:
            Tuple[float, float]: (GRPO权重, 辅助权重)
        """
        return self.state.grpo_weight, self.state.current_auxiliary_weight
    
    async def prepare_auxiliary_data(self, grpo_groups: List[List[Dict]]) -> List[List[Dict]]:
        """
        准备辅助奖励计算所需的数据
        
        Args:
            grpo_groups: GRPO组列表
            
        Returns:
            List[List[Dict]]: 辅助数据组列表
        """
        if not self.enable_auxiliary_reward or not self.rag_client:
            logger.debug("辅助奖励未启用或RAG客户端不可用")
            return [[] for _ in grpo_groups]
        
        try:
            auxiliary_data_groups = []
            
            for group_idx, group in enumerate(grpo_groups):
                auxiliary_data_group = []
                
                for sample_idx, sample in enumerate(group):
                    try:
                        # 提取样本数据
                        prompt = sample.get("prompt", "")
                        original_data = sample.get("original_data", {})
                        history_chat = original_data.get("history_chat", "")
                        
                        # 生成参考模型输入（与Actor模型相同的prompt）
                        ref_input = {
                            "prompt": prompt,
                            "history_chat": history_chat,
                            "use_reference_model": True
                        }
                        
                        # 调用参考模型（Qwen-32B通过RAG接口）
                        ref_result = await self._call_reference_model(ref_input)
                        
                        auxiliary_data_group.append({
                            "sample_id": f"group_{group_idx}_sample_{sample_idx}",
                            "ref_input": ref_input,
                            "ref_result": ref_result,
                            "original_sample": sample
                        })
                        
                    except Exception as e:
                        logger.error(f"组{group_idx}样本{sample_idx}辅助数据准备失败: {e}")
                        # 添加错误标记的默认数据
                        auxiliary_data_group.append({
                            "sample_id": f"group_{group_idx}_sample_{sample_idx}",
                            "ref_input": ref_input,
                            "ref_result": {"error": str(e), "overall_score": 0.5},
                            "original_sample": sample
                        })
                
                auxiliary_data_groups.append(auxiliary_data_group)
            
            logger.info(f"辅助数据准备完成，共{len(auxiliary_data_groups)}个组")
            return auxiliary_data_groups
            
        except Exception as e:
            logger.error(f"辅助数据准备失败: {e}")
            return [[] for _ in grpo_groups]
    
    async def _call_reference_model(self, ref_input: Dict) -> Dict:
        """
        调用参考模型（Qwen-32B通过RAG接口）
        
        Args:
            ref_input: 参考模型输入
            
        Returns:
            Dict: 参考模型结果
        """
        try:
            prompt = ref_input.get("prompt", "")
            
            # 调用RAG /chat接口（Qwen-32B）
            rag_result = await self.rag_client.chat(
                context=prompt,
                score_threshold=0.7
            )
            
            response_data, status, request_body, cost_time = rag_result
            
            # 根据您提供的RAG返回格式进行解析 - 不计算任何分数
            if response_data and len(response_data) > 0:
                # 您提供的格式：response_data包含data字段，status是RAGResponseStatus
                first_response = response_data[0] if isinstance(response_data, list) else response_data
                
                if isinstance(first_response, dict) and "data" in first_response:
                    data_field = first_response["data"]
                    
                    # 提取RAG返回的完整数据，不计算任何质量分数
                    # 质量评估将由GPT-5在后续步骤中进行
                    user_profile = data_field.get("user_profile", "")
                    history_summary = data_field.get("history_summary", "")
                    rewritten_query = data_field.get("rewritten_query", "")
                    recall_data = data_field.get("recall", [])
                    
                    # 返回完整数据供GPT-5评分使用 - 不计算overall_score
                    return {
                        "user_profile": user_profile,
                        "rewritten_query": rewritten_query,
                        "history_summary": history_summary,
                        "rag_recall": recall_data,
                        "rag_status": status,
                        "cost_time": cost_time,
                        "success": status == "success" or status == "no_knowledge_required",
                        "raw_data": data_field  # 保留完整原始数据供GPT-5评分
                    }
                else:
                    # 如果没有data字段，使用默认结构
                    return {
                        "user_profile": "",
                        "rewritten_query": "",
                        "history_summary": "",
                        "rag_recall": [],
                        "rag_status": status,
                        "cost_time": cost_time,
                        "success": status == "success" or status == "no_knowledge_required",
                        "raw_data": first_response
                    }
            else:
                return {
                    "overall_score": 0.5,
                    "user_profile": "",
                    "rewritten_query": "",
                    "history_summary": "",
                    "rag_recall": [],
                    "rag_status": "error",
                    "cost_time": cost_time,
                    "success": False,
                    "error": "No response data"
                }
                
        except Exception as e:
            logger.error(f"参考模型调用失败: {e}")
            return {
                "overall_score": 0.5,
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": "",
                "rag_recall": [],
                "rag_status": "error",
                "cost_time": 0.0,
                "success": False,
                "error": str(e)
            }
    
    async def compute_hybrid_rewards(
        self, 
        grpo_groups: List[List[Dict]], 
        auxiliary_data_groups: List[List[Dict]]
    ) -> List[List[float]]:
        """
        计算混合奖励
        
        Args:
            grpo_groups: GRPO组列表
            auxiliary_data_groups: 辅助数据组列表
            
        Returns:
            List[List[float]]: 混合奖励列表
        """
        if not self.enable_auxiliary_reward or not self.reward_calculator:
            logger.debug("混合奖励计算被禁用，返回零奖励")
            return [[0.0] * len(group) for group in grpo_groups]
        
        try:
            # 准备基线结果（从辅助数据中提取）
            baseline_groups = []
            for aux_group in auxiliary_data_groups:
                baseline_group = []
                for aux_data in aux_group:
                    ref_result = aux_data.get("ref_result", {})
                    baseline_group.append({
                        "complete_response": ref_result
                    })
                baseline_groups.append(baseline_group)
            
            # 使用混合奖励计算器计算奖励
            all_rewards = await self.reward_calculator.batch_compute_rewards(
                grpo_groups, baseline_groups
            )
            
            # 记录统计信息
            for i, (group_rewards, grpo_group) in enumerate(zip(all_rewards, grpo_groups)):
                stats = self.reward_calculator.get_reward_statistics(group_rewards)
                self.state.cumulative_stats[f'group_{i}_rewards'].extend(group_rewards)
                self.state.cumulative_stats[f'group_{i}_stats'].append(stats)
            
            logger.info(f"混合奖励计算完成，共{len(all_rewards)}个组")
            return all_rewards
            
        except Exception as e:
            logger.error(f"混合奖励计算失败: {e}")
            # 失败时返回零奖励
            return [[0.0] * len(group) for group in grpo_groups]
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            Dict[str, Any]: 训练统计信息
        """
        try:
            stats = {
                "training_progress": self.state.training_progress,
                "current_step": self.state.current_step,
                "total_steps": self.state.total_steps,
                "current_grpo_weight": self.state.grpo_weight,
                "current_auxiliary_weight": self.state.current_auxiliary_weight,
                "auxiliary_reward_enabled": self.enable_auxiliary_reward,
                "cumulative_stats": dict(self.state.cumulative_stats)
            }
            
            # 计算平均奖励
            if self.state.cumulative_stats:
                reward_stats = []
                for key, values in self.state.cumulative_stats.items():
                    if key.endswith('_rewards') and values:
                        reward_stats.extend(values)
                
                if reward_stats:
                    stats["avg_reward"] = float(np.mean(reward_stats))
                    stats["reward_std"] = float(np.std(reward_stats))
                    stats["reward_range"] = [float(np.min(reward_stats)), float(np.max(reward_stats))]
            
            return stats
            
        except Exception as e:
            logger.error(f"训练统计信息获取失败: {e}")
            return {
                "training_progress": self.state.training_progress,
                "error": str(e)
            }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.state.cumulative_stats.clear()
        logger.info("训练统计信息已重置")


# 便捷函数
async def prepare_and_compute_hybrid_rewards(
    grpo_groups: List[List[Dict]],
    training_manager: HybridTrainingManager,
    update_progress: bool = True
) -> List[List[float]]:
    """
    便捷函数：准备辅助数据并计算混合奖励
    
    Args:
        grpo_groups: GRPO组列表
        training_manager: 混合训练管理器
        update_progress: 是否更新训练进度
        
    Returns:
        List[List[float]]: 混合奖励列表
    """
    if not training_manager.enable_auxiliary_reward:
        # 如果辅助奖励未启用，直接返回零奖励
        return [[0.0] * len(group) for group in grpo_groups]
    
    try:
        # 准备辅助数据
        auxiliary_data_groups = await training_manager.prepare_auxiliary_data(grpo_groups)
        
        # 计算混合奖励
        rewards = await training_manager.compute_hybrid_rewards(
            grpo_groups, auxiliary_data_groups
        )
        
        # 更新训练进度（如果需要）
        if update_progress:
            current_step = len(grpo_groups)  # 假设每个组代表一个训练步骤
            total_steps = current_step * 10  # 假设总共有10倍当前步数
            training_manager.update_training_progress(current_step, total_steps)
        
        return rewards
        
    except Exception as e:
        logger.error(f"混合奖励计算便捷函数失败: {e}")
        return [[0.0] * len(group) for group in grpo_groups]


if __name__ == "__main__":
    # 测试代码
    async def test_hybrid_training_manager():
        # 创建训练管理器
        manager = HybridTrainingManager(
            enable_auxiliary_reward=True,
            auxiliary_weight=0.3,
            weight_decay_rate=0.4,
            scoring_model="GPT-5"
        )
        
        # 模拟GRPO组数据
        grpo_groups = [
            [
                {
                    "prompt": "测试prompt 1",
                    "original_data": {"history_chat": "历史对话1"},
                    "grpo_score": 0.8
                },
                {
                    "prompt": "测试prompt 1", 
                    "original_data": {"history_chat": "历史对话1"},
                    "grpo_score": 0.6
                }
            ],
            [
                {
                    "prompt": "测试prompt 2",
                    "original_data": {"history_chat": "历史对话2"},
                    "grpo_score": 0.9
                },
                {
                    "prompt": "测试prompt 2",
                    "original_data": {"history_chat": "历史对话2"},
                    "grpo_score": 0.7
                }
            ]
        ]
        
        # 更新训练进度
        manager.update_training_progress(5, 100)
        
        # 准备辅助数据
        auxiliary_data = await manager.prepare_auxiliary_data(grpo_groups)
        
        # 计算混合奖励
        rewards = await manager.compute_hybrid_rewards(grpo_groups, auxiliary_data)
        
        print("混合奖励计算结果:")
        for i, group_rewards in enumerate(rewards):
            print(f"组{i}: {group_rewards}")
        
        # 获取统计信息
        stats = manager.get_training_statistics()
        print("\n训练统计信息:")
        print(stats)
    
    # 运行测试
    asyncio.run(test_hybrid_training_manager())