"""
基于Sales-RAG场景的自定义奖励函数
集成用户反馈、检索质量、对话流畅性的综合奖励计算
"""

import json
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SalesRAGReward:
    """销售RAG专用奖励函数"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 功能开关
        self.use_user_feedback = self.config.get("use_user_feedback", True)
        self.use_retrieval_score = self.config.get("use_retrieval_score", True) 
        self.use_conversation_flow = self.config.get("use_conversation_flow", True)
        
        # 权重配置
        self.weights = self.config.get("weights", {
            "user_feedback": 0.5,      # 用户反馈权重最高
            "retrieval_quality": 0.3,  # 检索质量
            "conversation_flow": 0.2    # 对话流畅性
        })
        
        # 产品关键词库
        self.product_keywords = [
            "胶原蛋白肽", "胶原蛋白", "富铁软糖", "虾青素", "抗糖", 
            "维生素", "钙片", "益生菌", "叶酸", "DHA"
        ]
        
        # 意图关键词库
        self.intent_keywords = {
            "功效": ["功效", "作用", "效果", "好处", "有什么用"],
            "用法": ["怎么吃", "怎么用", "如何服用", "用法", "服用方法", "使用方法"],
            "禁忌": ["禁忌", "不能吃", "注意事项", "副作用", "孕妇", "哺乳期"],
            "成分": ["成分", "配方", "含量", "原料"],
            "适用人群": ["适用人群", "谁能吃", "年龄", "性别"],
            "时间": ["什么时候", "早上", "晚上", "饭前", "饭后", "空腹"]
        }
        
        logger.info(f"SalesRAGReward初始化完成，权重配置: {self.weights}")
    
    def compute_reward(
        self,
        original_query: str,
        rewritten_query: str,
        retrieval_results: List[Dict] = None,
        user_feedback: Dict = None,
        conversation_history: List[Dict] = None,
        **kwargs
    ) -> float:
        """
        计算综合奖励
        
        Args:
            original_query: 原始用户查询
            rewritten_query: 重写后的查询
            retrieval_results: 检索结果列表
            user_feedback: 用户反馈 {"satisfaction": 1/0/-1, "continued": bool, "turn_count": int}
            conversation_history: 对话历史
            
        Returns:
            奖励值 [-1.0, 1.0]
        """
        
        total_reward = 0.0
        reward_breakdown = {}
        
        try:
            # 1. 用户反馈奖励
            if self.use_user_feedback and user_feedback:
                feedback_reward = self._compute_feedback_reward(user_feedback)
                total_reward += self.weights["user_feedback"] * feedback_reward
                reward_breakdown["user_feedback"] = feedback_reward
                logger.debug(f"用户反馈奖励: {feedback_reward}")
            
            # 2. 检索质量奖励
            if self.use_retrieval_score and retrieval_results:
                retrieval_reward = self._compute_retrieval_reward(
                    original_query, rewritten_query, retrieval_results
                )
                total_reward += self.weights["retrieval_quality"] * retrieval_reward
                reward_breakdown["retrieval_quality"] = retrieval_reward
                logger.debug(f"检索质量奖励: {retrieval_reward}")
            
            # 3. 对话流畅性奖励  
            if self.use_conversation_flow and conversation_history:
                flow_reward = self._compute_conversation_flow_reward(
                    original_query, rewritten_query, conversation_history
                )
                total_reward += self.weights["conversation_flow"] * flow_reward
                reward_breakdown["conversation_flow"] = flow_reward
                logger.debug(f"对话流畅性奖励: {flow_reward}")
            
            # 归一化到[-1, 1]区间
            final_reward = np.clip(total_reward, -1.0, 1.0)
            
            logger.info(
                f"奖励计算完成: '{original_query}' -> '{rewritten_query}', "
                f"最终奖励: {final_reward:.3f}, 分解: {reward_breakdown}"
            )
            
            return final_reward
            
        except Exception as e:
            logger.error(f"奖励计算异常: {e}")
            return 0.0  # 异常情况返回中性奖励
    
    def _compute_feedback_reward(self, user_feedback: Dict) -> float:
        """基于用户反馈计算奖励"""
        
        reward = 0.0
        
        # 1. 显式反馈 (点赞/点踩/满意度)
        satisfaction = user_feedback.get("satisfaction", 0)  # 1, 0, -1
        reward += satisfaction  # 直接使用满意度分数
        
        # 2. 隐式反馈 (是否继续对话)
        continued = user_feedback.get("continued", False)
        if continued:
            reward += 0.3  # 继续对话是正向信号
        else:
            reward -= 0.1  # 中断对话是负向信号
        
        # 3. 对话轮数奖励 (更多轮次通常表示更好的用户体验)
        turn_count = user_feedback.get("turn_count", 1)
        if turn_count > 1:
            # 2-4轮是理想的对话长度
            if 2 <= turn_count <= 4:
                reward += 0.2
            elif turn_count > 4:
                reward += 0.1  # 过长对话可能表示问题没有很好解决
        
        # 4. 明确的用户操作反馈
        explicit_action = user_feedback.get("explicit_feedback", "")
        if explicit_action == "thumbs_up":
            reward += 0.4
        elif explicit_action == "thumbs_down":
            reward -= 0.4
        
        return np.clip(reward, -1.5, 1.5)  # 允许超出[-1,1]，后续会归一化
    
    def _compute_retrieval_reward(
        self,
        original_query: str,
        rewritten_query: str, 
        retrieval_results: List[Dict]
    ) -> float:
        """基于检索质量计算奖励"""
        
        if not retrieval_results:
            return -0.5  # 没有检索结果是负向信号
        
        reward = 0.0
        
        # 1. 检索分数奖励
        scores = []
        for doc in retrieval_results[:5]:  # 只考虑top-5
            score = doc.get("score", 0)
            # 处理不同的相似度分数范围
            if score > 1:  # 如果是距离分数，需要转换
                score = 1.0 / (1.0 + score)
            scores.append(score)
        
        if scores:
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            
            # 基于平均分数
            if avg_score > 0.8:
                score_reward = 1.0
            elif avg_score > 0.6:
                score_reward = 0.5
            elif avg_score > 0.4:
                score_reward = 0.0
            else:
                score_reward = -0.3
            
            # 基于最高分数的额外奖励
            if max_score > 0.9:
                score_reward += 0.2
            
            reward += score_reward * 0.4
        
        # 2. 产品匹配奖励 (是否检索到相关产品)
        product_match_count = 0
        product_diversity = set()
        
        for doc in retrieval_results[:3]:  # 检查top-3文档
            doc_text = (doc.get("content", "") + " " + doc.get("title", "")).lower()
            
            for product in self.product_keywords:
                if product in doc_text or product.lower() in doc_text:
                    product_match_count += 1
                    product_diversity.add(product)
                    break  # 每个文档只算一次产品匹配
        
        product_reward = min(1.0, product_match_count / 3.0)  # 归一化
        diversity_bonus = len(product_diversity) * 0.1  # 产品多样性奖励
        
        reward += (product_reward + diversity_bonus) * 0.3
        
        # 3. 查询改进效果奖励
        improvement_reward = self._assess_query_improvement(
            original_query, rewritten_query, retrieval_results
        )
        reward += improvement_reward * 0.3
        
        return np.clip(reward, -1.0, 1.5)
    
    def _compute_conversation_flow_reward(
        self,
        original_query: str,
        rewritten_query: str,
        conversation_history: List[Dict]
    ) -> float:
        """基于对话流畅性计算奖励"""
        
        if len(conversation_history) < 2:
            return 0.0  # 对话太短，无法评估流畅性
        
        reward = 0.0
        
        # 1. 话题一致性奖励
        consistency_reward = self._assess_topic_consistency(conversation_history)
        reward += consistency_reward * 0.4
        
        # 2. 查询重写自然度奖励
        naturalness_reward = self._assess_query_naturalness(original_query, rewritten_query)
        reward += naturalness_reward * 0.3
        
        # 3. 对话进展奖励 (是否在推进问题解决)
        progress_reward = self._assess_conversation_progress(conversation_history)
        reward += progress_reward * 0.3
        
        return np.clip(reward, -1.0, 1.0)
    
    def _assess_query_improvement(
        self,
        original: str, 
        rewritten: str, 
        results: List[Dict]
    ) -> float:
        """评估查询改进效果"""
        
        improvement_score = 0.0
        
        # 1. 产品关键词补充
        original_lower = original.lower()
        rewritten_lower = rewritten.lower()
        
        added_products = 0
        for product in self.product_keywords:
            if product.lower() not in original_lower and product.lower() in rewritten_lower:
                added_products += 1
        
        if added_products > 0:
            improvement_score += min(0.4, added_products * 0.2)
        
        # 2. 意图明确化
        added_intents = 0
        for intent, keywords in self.intent_keywords.items():
            original_has_intent = any(kw in original_lower for kw in keywords)
            rewritten_has_intent = any(kw in rewritten_lower for kw in keywords)
            
            if not original_has_intent and rewritten_has_intent:
                added_intents += 1
        
        if added_intents > 0:
            improvement_score += min(0.3, added_intents * 0.15)
        
        # 3. 检索结果质量提升 (间接指标)
        if results and len(results) > 0:
            top_score = results[0].get("score", 0)
            if top_score > 0.7:
                improvement_score += 0.3
            elif top_score > 0.5:
                improvement_score += 0.1
        
        # 4. 避免过度重写的惩罚
        length_ratio = len(rewritten) / max(1, len(original))
        if length_ratio > 3:  # 重写后过长
            improvement_score -= 0.2
        elif length_ratio < 0.3:  # 重写后过短
            improvement_score -= 0.1
        
        # 5. 完全没有改变的惩罚
        if original.strip() == rewritten.strip():
            improvement_score = -0.1
        
        return np.clip(improvement_score, -0.5, 1.0)
    
    def _assess_topic_consistency(self, history: List[Dict]) -> float:
        """评估话题一致性"""
        
        # 简化版本：检查是否提到了同样的产品
        mentioned_products = []
        
        for turn in history[-5:]:  # 检查最近5轮对话
            content = turn.get("content", "").lower()
            for product in self.product_keywords:
                if product.lower() in content:
                    mentioned_products.append(product)
        
        # 如果多轮对话中提到了同样的产品，给正向奖励
        unique_products = set(mentioned_products)
        if len(unique_products) == 1 and len(mentioned_products) > 1:
            return 0.5  # 专注讨论单一产品
        elif len(unique_products) <= 2 and len(mentioned_products) > 0:
            return 0.3  # 相对集中的话题
        else:
            return 0.1  # 话题较分散
    
    def _assess_query_naturalness(self, original: str, rewritten: str) -> float:
        """评估查询重写的自然度"""
        
        # 1. 长度合理性
        length_ratio = len(rewritten) / max(1, len(original))
        length_score = 0.0
        
        if 0.8 <= length_ratio <= 2.5:  # 合理的长度变化
            length_score = 0.3
        elif 2.5 < length_ratio <= 3.5:  # 稍长但可接受
            length_score = 0.1
        else:  # 过长或过短
            length_score = -0.2
        
        # 2. 保留原始关键信息
        original_tokens = set(re.findall(r'\w+', original.lower()))
        rewritten_tokens = set(re.findall(r'\w+', rewritten.lower()))
        
        if len(original_tokens) > 0:
            overlap_ratio = len(original_tokens & rewritten_tokens) / len(original_tokens)
            if overlap_ratio > 0.5:  # 保留了超过50%的原始信息
                preservation_score = 0.3
            elif overlap_ratio > 0.3:
                preservation_score = 0.1
            else:
                preservation_score = -0.2
        else:
            preservation_score = 0.0
        
        # 3. 避免机械式关键词堆砌
        keyword_density = sum(1 for kw in self.product_keywords + 
                            [item for sublist in self.intent_keywords.values() for item in sublist]
                            if kw.lower() in rewritten.lower())
        
        total_words = len(re.findall(r'\w+', rewritten))
        if total_words > 0:
            density_ratio = keyword_density / total_words
            if density_ratio > 0.5:  # 关键词密度过高
                density_score = -0.3
            elif density_ratio > 0.3:
                density_score = -0.1
            else:
                density_score = 0.2
        else:
            density_score = 0.0
        
        return length_score + preservation_score + density_score
    
    def _assess_conversation_progress(self, history: List[Dict]) -> float:
        """评估对话是否在推进问题解决"""
        
        # 简化版本：检查对话是否从问题询问转向具体讨论
        if len(history) < 3:
            return 0.0
        
        # 检查早期是否有疑问词，后期是否有具体信息
        early_turns = history[:len(history)//2]
        late_turns = history[len(history)//2:]
        
        question_words = ["什么", "怎么", "如何", "哪个", "为什么", "?", "？"]
        answer_indicators = ["建议", "推荐", "可以", "应该", "每天", "服用"]
        
        early_has_questions = any(
            any(qw in turn.get("content", "") for qw in question_words)
            for turn in early_turns
        )
        
        late_has_answers = any(
            any(ai in turn.get("content", "") for ai in answer_indicators)
            for turn in late_turns
        )
        
        if early_has_questions and late_has_answers:
            return 0.5  # 从询问到解答的良好进展
        elif late_has_answers:
            return 0.3  # 有具体建议
        elif early_has_questions:
            return 0.1  # 至少有明确的问题
        else:
            return 0.0


# 工厂函数，供DeepRetrieval调用
def create_sales_rag_reward(config: Dict = None) -> SalesRAGReward:
    """创建销售RAG奖励函数实例"""
    return SalesRAGReward(config)


# 兼容DeepRetrieval框架的接口
def compute_reward(
    original_query: str,
    rewritten_query: str,
    context: Dict
) -> float:
    """
    DeepRetrieval框架兼容接口
    
    Args:
        original_query: 原始查询
        rewritten_query: 重写查询  
        context: 上下文信息，包含retrieval_results, user_feedback等
        
    Returns:
        奖励值 [-1.0, 1.0]
    """
    reward_function = SalesRAGReward()
    
    return reward_function.compute_reward(
        original_query=original_query,
        rewritten_query=rewritten_query,
        retrieval_results=context.get("retrieval_results", []),
        user_feedback=context.get("user_feedback", {}),
        conversation_history=context.get("conversation_history", [])
    )


if __name__ == "__main__":
    # 测试用例
    print("测试SalesRAGReward奖励函数...")
    
    reward_func = SalesRAGReward()
    
    # 测试案例1：好的查询重写
    reward1 = reward_func.compute_reward(
        original_query="胶原蛋白怎么吃",
        rewritten_query="胶原蛋白肽 服用方法 推荐用量 适用人群",
        retrieval_results=[
            {"score": 0.85, "content": "胶原蛋白肽的推荐服用方法是每天1-2次..."},
            {"score": 0.78, "content": "胶原蛋白肽适合25岁以上女性服用..."}
        ],
        user_feedback={
            "satisfaction": 1,
            "continued": True,
            "turn_count": 3,
            "explicit_feedback": "thumbs_up"
        }
    )
    print(f"测试案例1奖励: {reward1:.3f}")
    
    # 测试案例2：糟糕的查询重写  
    reward2 = reward_func.compute_reward(
        original_query="有什么好的保健品",
        rewritten_query="保健品产品信息查询检索",
        retrieval_results=[
            {"score": 0.32, "content": "保健品种类繁多..."}
        ],
        user_feedback={
            "satisfaction": -1,
            "continued": False,
            "turn_count": 1,
            "explicit_feedback": "thumbs_down"
        }
    )
    print(f"测试案例2奖励: {reward2:.3f}")
    
    print("测试完成!")
