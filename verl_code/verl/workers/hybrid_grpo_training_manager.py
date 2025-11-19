"""
混合 GRPO 训练管理器
用于管理 GRPO + GPT-5 混合奖励训练流程
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class HybridTrainingManager:
    """
    混合 GRPO 训练管理器
    
    功能：
    1. 管理 GRPO 和 GPT-5 奖励的融合权重
    2. 支持动态权重调整
    3. 跟踪训练进度和质量指标
    4. 组内中心化处理
    """
    
    def __init__(
        self,
        enable_auxiliary_reward: bool = True,
        auxiliary_weight: float = 0.15,
        weight_decay_rate: float = 0.4,
        min_auxiliary_weight: float = 0.1,
        scoring_model: str = "GPT-5",
        group_size: int = 5,
        enable_dynamic_weight: bool = False,
        **kwargs
    ):
        """
        初始化混合训练管理器
        
        Args:
            enable_auxiliary_reward: 是否启用辅助奖励（GPT-5）
            auxiliary_weight: 辅助奖励的初始权重（GRPO 权重 = 1 - auxiliary_weight）
            weight_decay_rate: 权重衰减率（用于动态调整）
            min_auxiliary_weight: 辅助奖励的最小权重
            scoring_model: 评分模型名称
            group_size: 每组的候选数量
            enable_dynamic_weight: 是否启用动态权重调整
        """
        self.enable_auxiliary_reward = enable_auxiliary_reward
        self.auxiliary_weight = auxiliary_weight
        self.grpo_weight = 1.0 - auxiliary_weight
        self.weight_decay_rate = weight_decay_rate
        self.min_auxiliary_weight = min_auxiliary_weight
        self.scoring_model = scoring_model
        self.group_size = group_size
        self.enable_dynamic_weight = enable_dynamic_weight
        
        # 训练统计
        self.current_step = 0
        self.total_steps = 0
        self.quality_history = []
        
        logger.info(f"初始化混合训练管理器:")
        logger.info(f"  - 辅助奖励启用: {enable_auxiliary_reward}")
        logger.info(f"  - GRPO 权重: {self.grpo_weight:.2f}")
        logger.info(f"  - GPT-5 权重: {self.auxiliary_weight:.2f}")
        logger.info(f"  - 动态权重: {enable_dynamic_weight}")
        logger.info(f"  - 组大小: {group_size}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        获取当前的奖励权重
        
        Returns:
            包含 grpo_weight 和 auxiliary_weight 的字典
        """
        return {
            "grpo_weight": self.grpo_weight,
            "auxiliary_weight": self.auxiliary_weight,
            "gpt5_weight": self.auxiliary_weight  # 别名
        }
    
    def update_weights(self, step: int, total_steps: int):
        """
        更新奖励权重（如果启用动态调整）
        
        Args:
            step: 当前训练步数
            total_steps: 总训练步数
        """
        self.current_step = step
        self.total_steps = total_steps
        
        if not self.enable_dynamic_weight:
            return
        
        # 动态权重调整：随着训练进行，逐渐降低辅助奖励权重
        progress = step / max(total_steps, 1)
        decay_factor = np.exp(-self.weight_decay_rate * progress)
        
        # 计算新的辅助权重
        new_auxiliary_weight = max(
            self.min_auxiliary_weight,
            self.auxiliary_weight * decay_factor
        )
        
        # 更新权重
        self.auxiliary_weight = new_auxiliary_weight
        self.grpo_weight = 1.0 - new_auxiliary_weight
        
        if step % 100 == 0:
            logger.info(
                f"步骤 {step}/{total_steps}: "
                f"GRPO={self.grpo_weight:.3f}, "
                f"GPT-5={self.auxiliary_weight:.3f}"
            )
    
    def process_rewards(
        self,
        grpo_rewards: np.ndarray,
        auxiliary_rewards: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        处理和融合奖励
        
        Args:
            grpo_rewards: GRPO 奖励数组
            auxiliary_rewards: 辅助奖励数组（GPT-5）
            
        Returns:
            融合后的奖励数组
        """
        if not self.enable_auxiliary_reward or auxiliary_rewards is None:
            return grpo_rewards
        
        # 确保形状匹配
        assert grpo_rewards.shape == auxiliary_rewards.shape, \
            f"奖励形状不匹配: GRPO={grpo_rewards.shape}, Aux={auxiliary_rewards.shape}"
        
        # 融合奖励
        fused_rewards = (
            self.grpo_weight * grpo_rewards +
            self.auxiliary_weight * auxiliary_rewards
        )
        
        return fused_rewards
    
    def normalize_group_rewards(
        self,
        rewards: np.ndarray,
        group_size: Optional[int] = None
    ) -> np.ndarray:
        """
        组内中心化奖励
        
        Args:
            rewards: 奖励数组，形状为 (batch_size,) 或 (batch_size, group_size)
            group_size: 组大小，如果为 None 则使用 self.group_size
            
        Returns:
            中心化后的奖励数组
        """
        if group_size is None:
            group_size = self.group_size
        
        # 如果是一维数组，重塑为 (num_groups, group_size)
        if rewards.ndim == 1:
            num_groups = len(rewards) // group_size
            if len(rewards) % group_size != 0:
                logger.warning(
                    f"奖励数量 {len(rewards)} 不能被组大小 {group_size} 整除"
                )
                # 截断到最近的完整组
                rewards = rewards[:num_groups * group_size]
            
            rewards = rewards.reshape(num_groups, group_size)
        
        # 组内中心化：减去组内均值
        group_means = rewards.mean(axis=1, keepdims=True)
        centered_rewards = rewards - group_means
        
        # 展平回原始形状
        return centered_rewards.flatten()
    
    def log_statistics(self, rewards: Dict[str, Any]):
        """
        记录训练统计信息
        
        Args:
            rewards: 包含各种奖励的字典
        """
        stats = {
            "step": self.current_step,
            "grpo_weight": self.grpo_weight,
            "auxiliary_weight": self.auxiliary_weight,
        }
        
        # 添加奖励统计
        for key, value in rewards.items():
            if isinstance(value, (np.ndarray, list)):
                value = np.array(value)
                stats[f"{key}_mean"] = float(value.mean())
                stats[f"{key}_std"] = float(value.std())
                stats[f"{key}_max"] = float(value.max())
                stats[f"{key}_min"] = float(value.min())
        
        # 记录质量历史
        if "fused_reward" in stats:
            self.quality_history.append(stats["fused_reward_mean"])
        
        return stats
    
    def should_update_weights(self, step: int, update_interval: int = 100) -> bool:
        """
        判断是否应该更新权重
        
        Args:
            step: 当前步数
            update_interval: 更新间隔
            
        Returns:
            是否应该更新
        """
        return self.enable_dynamic_weight and (step % update_interval == 0)
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        获取训练进度信息
        
        Returns:
            包含进度信息的字典
        """
        progress = {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_ratio": self.current_step / max(self.total_steps, 1),
            "grpo_weight": self.grpo_weight,
            "auxiliary_weight": self.auxiliary_weight,
        }
        
        if self.quality_history:
            progress["avg_quality"] = float(np.mean(self.quality_history[-100:]))
            progress["quality_trend"] = self._compute_trend()
        
        return progress
    
    def _compute_trend(self, window: int = 50) -> str:
        """
        计算质量趋势
        
        Args:
            window: 窗口大小
            
        Returns:
            趋势描述: "improving", "stable", "declining"
        """
        if len(self.quality_history) < window * 2:
            return "insufficient_data"
        
        recent = np.mean(self.quality_history[-window:])
        previous = np.mean(self.quality_history[-2*window:-window])
        
        diff = recent - previous
        threshold = 0.01  # 1% 变化阈值
        
        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "declining"
        else:
            return "stable"
    
    def reset(self):
        """重置管理器状态"""
        self.current_step = 0
        self.quality_history = []
        logger.info("混合训练管理器已重置")
    
    def __repr__(self) -> str:
        return (
            f"HybridTrainingManager("
            f"grpo_weight={self.grpo_weight:.3f}, "
            f"auxiliary_weight={self.auxiliary_weight:.3f}, "
            f"dynamic={self.enable_dynamic_weight}, "
            f"step={self.current_step}/{self.total_steps})"
        )
