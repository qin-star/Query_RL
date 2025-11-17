from math import tanh
import numpy as np
from src.pipeline import get_rag_rl_result,get_rate_result


def call_rag_chat_rl(context: str, user_profile: str, history_summary: str, rewritten_query: str):
    """
    调用 rag 本地模型接口，获取 8B 和 32B 结果
    """
    return get_rag_rl_result(context, user_profile=user_profile, history_summary=history_summary, rewritten_query=rewritten_query)


def call_llm_rate(payload: dict):
    """
    调用大模型获取 rate 分数
    """
    return get_rate_result(payload)


def compute_reward(better, sums):
    """
    用于计算 reward 分数
    
    Args:
        better: 字符串，表示哪个模型更好 ("8b", "32b", "same", "both bad")
        sums: 列表，包含两个模型的评分 [sum_8b, sum_32b]
    
    Returns:
        float: 奖励分数，范围[-1, 1]
    """

    reward_rules = {
        "8b": lambda r: r + 0.2,
        "32b": lambda r: r - 0.2,
        "same": lambda r: r * 0.5,
        "both bad": lambda r: -0.5
    }

    sum0, sum1 = sums[0], sums[1]
    sum_diff = abs(sum0 - sum1) / 100
    base_reward = tanh(sum_diff * 2)

    if better in reward_rules:
        base_reward = reward_rules[better](base_reward)
    
    return np.clip(base_reward, -1, 1)


def compute_score(solution_str, ground_truth, format_reward=1.0, answer_reward=1.0):
    """
    主要的评分函数，符合DeepRetrieval的接口规范
    
    Args:
        solution_str: 用户的历史对话（在这个场景中，不是模型生成的响应）
        ground_truth: 包含评分和比较结果的字典
        format_reward: 格式正确的奖励分数
        answer_reward: 答案质量的奖励权重
    
    Returns:
        float: 总奖励分数
    """
    try:
        # TODO 解析出 context 和 answer， answer 中解析出 user_profile、 history_summary、 rewritten_query
        context = ""
        user_profile = ""
        history_summary = ""
        rewritten_query = ""

        # 调用 RAG 系统获取两个模型的比较结果
        chat_resp, chat_8b_resp = call_rag_chat_rl(context=context, user_profile=user_profile, history_summary=history_summary, rewritten_query=rewritten_query)
        


        # 调用评估系统获取奖励分数
        rate_result = call_llm_rate({
            "chat_resp": chat_resp,
            "chat_8b_resp": chat_8b_resp
        })
        
        # 解析评估结果
        if isinstance(rate_result, dict):
            better = rate_result.get("better", "same")
            
            # 获取两个模型的分数
            score_data = rate_result.get("score", {})
            if isinstance(score_data, dict):
                scores_32b = score_data.get("32b", {})
                scores_8b = score_data.get("8b", {})
                
                sum_32b = scores_32b.get("sum", 0) if isinstance(scores_32b, dict) else 0
                sum_8b = scores_8b.get("sum", 0) if isinstance(scores_8b, dict) else 0
                sums = [sum_8b, sum_32b]
            else:
                # 如果分数格式不正确，使用默认值
                better = "same"
                sums = [0, 0]
        else:
            # 如果评估结果不是字典，使用默认值
            better = "same"
            sums = [0, 0]
        
        # 计算最终奖励
        reward = compute_reward(better, sums)
        
        # 记录调试信息
        print(f"[DEBUG] solution_str: {solution_str[:100]}...")
        print(f"[DEBUG] better: {better}, sums: {sums}, reward: {reward}")
        
        return reward
        
    except Exception as e:
        print(f"[ERROR] compute_score failed: {e}")
        # 出现错误时返回中性奖励
        return 0.0
