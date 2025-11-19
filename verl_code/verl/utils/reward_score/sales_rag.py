import numpy as np
import json


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    """
    简化的GRPO评分函数，仅用于组内优势计算
    不调用RAG，只基于生成质量给出初步奖励
    
    Args:
        data_source: 数据源标识（verl 框架传递）
        solution_str: 模型生成的改写查询
        ground_truth: 包含原始数据的字典（prompt, context 等）
        extra_info: 额外信息字典
        **kwargs: 其他可能的参数
    
    Returns:
        float: 初步奖励分数，范围 [-1, 1]
    """
    try:
        # solution_str 就是模型生成的改写查询
        rewritten_query = solution_str.strip()
        
        # 基础检查：空查询
        if not rewritten_query:
            print("[DEBUG] Empty rewritten_query")
            return -0.5
        
        # 尝试解析JSON格式
        try:
            parsed_output = json.loads(solution_str)
            if isinstance(parsed_output, dict):
                has_structure = True
                rewritten_query = parsed_output.get("rewritten_query", "")
                user_profile = parsed_output.get("user_profile", "")
                history_summary = parsed_output.get("history_summary", "")
                print(f"[DEBUG] JSON解析成功 - rewritten_query长度: {len(rewritten_query)}")
            else:
                # JSON解析成功但不是字典（比如是字符串或列表）
                has_structure = False
                user_profile = ""
                history_summary = ""
                print(f"[DEBUG] JSON解析成功但不是字典，类型: {type(parsed_output)}")
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            # JSON解析失败或其他错误
            has_structure = False
            user_profile = ""
            history_summary = ""
            print(f"[DEBUG] JSON解析失败: {e}, 内容前100字符: {solution_str[:100]}")
        
        # 简单的启发式评分（用于GRPO组内比较）
        reward = 0.0
        
        # 1. 结构完整性 (+0.2)
        if has_structure:
            reward += 0.2
        
        # 2. 查询长度合理性 (+0.2)
        query_len = len(rewritten_query)
        if 10 < query_len < 200:
            reward += 0.2
        elif query_len >= 200:
            reward += 0.1
        
        # 3. 包含有效内容 (+0.3)
        if rewritten_query and not rewritten_query.startswith("</think>"):
            reward += 0.3
        
        # 4. 用户画像和历史总结 (+0.3)
        if user_profile or history_summary:
            reward += 0.15
        if user_profile and history_summary:
            reward += 0.15
        
        # 归一化到[-1, 1]
        reward = np.clip(reward * 2 - 1, -1, 1)
        
        return reward
        
    except Exception as e:
        import traceback
        print(f"[ERROR] compute_score failed: {e}")
        print(traceback.format_exc())
        return 0.0
