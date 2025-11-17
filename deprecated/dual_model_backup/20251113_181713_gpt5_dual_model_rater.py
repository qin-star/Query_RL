import pprint
import json
from src.utils.llm import get_chat_llm, SafeParser
from src.utils.prompt import g_sa_prompt_manager


class GPT5DualModelRater:
    """GPT-5双模型对比评分器"""
    
    def __init__(
        self,
        llm: str,
        actor_response: dict,
        reference_response: dict
    ):
        self.llm = get_chat_llm(llm)
        self.actor_response = actor_response
        self.reference_response = reference_response
    
    def _prepare_eval_payload(self) -> dict:
        """准备评估所需的payload数据"""
        return {
            'history_chat': self.actor_response.get('history_chat', ''),
            'user_profile': self.reference_response.get('user_profile', ''),
            'rewritten_query': self.reference_response.get('rewritten_query', ''),
            'history_summary': self.reference_response.get('history_summary', ''),
            'rag_recall': self.reference_response.get('rag_recall', ''),
            'user_profile_8B': self.actor_response.get('user_profile', ''),
            'rewritten_query_8B': self.actor_response.get('rewritten_query', ''),
            'history_summary_8B': self.actor_response.get('history_summary', ''),
            'rag_recall_8B': self.actor_response.get('rag_recall', '')
        }
    
    async def rate(self) -> dict:
        """执行双模型对比评分"""
        # 准备评估数据
        eval_payload = self._prepare_eval_payload()
        
        # 使用现有的auto_eval模板进行双模型对比评分
        prompt = g_sa_prompt_manager.render_prompt(
            prompt_name="auto_eval",
            **eval_payload
        )
        
        # 调用LLM进行评分，添加超时和重试机制
        import asyncio
        try:
            raw_response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=10.0  #10秒超时
            )
            content = raw_response.content
        except asyncio.TimeoutError:
            logger.error("GPT-5评分超时")
            raise Exception("GPT-5评分超时")
        
        pprint.pprint(raw_response)
        pprint.pprint(content)
        
        # 解析评分结果
        try:
            result = SafeParser.parse_json_to_dict(content)
            
            # 将评分结果转换为0-1范围的标准化分数
            standardized_result = self._standardize_scores(result)
            
            return standardized_result
        except Exception as e:
            print(f"评分结果解析失败: {e}")
            # 返回默认评分结果
            return {
                "quality_improvement": 0.5,
                "relevance_accuracy": 0.5,
                "info_completeness": 0.5,
                "retrieval_effectiveness": 0.5,
                "overall_score": 0.5,
                "analysis": "评分解析失败，使用默认分数",
                "raw_response": content
            }
    
    def _standardize_scores(self, result: dict) -> dict:
        """将0-10分的评分结果标准化为0-1范围"""
        try:
            # 获取32b和8b的评分
            score_32b = result.get('score', {}).get('32b', {})
            score_8b = result.get('score', {}).get('8b', {})
            
            # 获取各维度分数
            scores_32b = score_32b.get('scores', [5, 5, 5, 5])
            scores_8b = score_8b.get('scores', [5, 5, 5, 5])
            
            # 根据better字段确定哪个模型表现更好
            better_model = result.get('better', 'same')
            
            # 计算标准化分数（0-10分 -> 0-1分）
            if better_model == '8b':
                # 8b模型表现更好，给予更高分数
                quality_improvement = scores_8b[0] / 10.0
                relevance_accuracy = scores_8b[1] / 10.0
                info_completeness = scores_8b[2] / 10.0
                retrieval_effectiveness = scores_8b[3] / 10.0
            elif better_model == '32b':
                # 32b模型表现更好，8b作为baseline，分数应该较低
                quality_improvement = max(0.1, scores_8b[0] / 10.0 - 0.2)
                relevance_accuracy = max(0.1, scores_8b[1] / 10.0 - 0.2)
                info_completeness = max(0.1, scores_8b[2] / 10.0 - 0.2)
                retrieval_effectiveness = max(0.1, scores_8b[3] / 10.0 - 0.2)
            else:
                # same或both bad情况，使用中等分数
                quality_improvement = 0.5
                relevance_accuracy = 0.5
                info_completeness = 0.5
                retrieval_effectiveness = 0.5
            
            # 计算总体评分
            overall_score = (
                quality_improvement * 0.4 +
                relevance_accuracy * 0.2 +
                info_completeness * 0.2 +
                retrieval_effectiveness * 0.2
            )
            
            return {
                "quality_improvement": round(quality_improvement, 2),
                "relevance_accuracy": round(relevance_accuracy, 2),
                "info_completeness": round(info_completeness, 2),
                "retrieval_effectiveness": round(retrieval_effectiveness, 2),
                "overall_score": round(overall_score, 2),
                "analysis": result.get('reason', ''),
                "better_model": better_model,
                "raw_scores": {
                    "32b": score_32b,
                    "8b": score_8b
                }
            }
        except Exception as e:
            print(f"分数标准化失败: {e}")
            return {
                "quality_improvement": 0.5,
                "relevance_accuracy": 0.5,
                "info_completeness": 0.5,
                "retrieval_effectiveness": 0.5,
                "overall_score": 0.5,
                "analysis": f"分数标准化失败: {e}",
                "better_model": "same",
                "raw_scores": {}
            }


# 便捷函数
async def rate_dual_models(
    llm: str,
    actor_response: dict,
    reference_response: dict
) -> dict:
    """便捷函数：对双模型进行对比评分"""
    rater = GPT5DualModelRater(llm, actor_response, reference_response)
    return await rater.rate()