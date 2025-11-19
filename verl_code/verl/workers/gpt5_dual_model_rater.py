import pprint
import json
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回到verl_code目录
src_path = os.path.join(project_root, '..', 'src')  # 指向 /home/jovyan2/query_rl/src

if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.utils.llm import get_chat_llm, SafeParser
    from src.utils.prompt import g_sa_prompt_manager
    from src.utils.log import logger
except ImportError as e:
    print(f"⚠️  导入src模块失败: {e}")
    print(f"   当前工作目录: {os.getcwd()}")
    print(f"   脚本目录: {current_dir}")
    print(f"   项目根目录: {project_root}")
    print(f"   src路径: {src_path}")
    print(f"   Python路径: {sys.path[:3]}...")
    # 创建模拟函数以避免崩溃
    def get_chat_llm(llm_name):
        return lambda x: f"模拟{llm_name}响应"
    
    class SafeParser:
        @staticmethod
        def parse(text):
            return {"score": 0.5}
    
    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    
    logger = MockLogger()
    g_sa_prompt_manager = {}


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
    
    async def rate(self) -> float:
        """执行双模型对比评分，返回单一质量分数
        
        Returns:
            float: 质量分数 [0, 1]
                - 0.9-1.0: 8B明显优于32B
                - 0.7-0.9: 8B略优于32B
                - 0.5-0.7: 相当
                - 0.3-0.5: 32B略优于8B
                - 0.0-0.3: 32B明显优于8B
        """
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
            return 0.5  # 超时返回中性分数
        
        logger.debug(f"GPT-5原始响应: {content[:200]}...")
        
        # 解析评分结果
        try:
            result = SafeParser.parse_json_to_dict(content)
            
            # 提取单一质量分数
            quality_score = self._extract_quality_score(result)
            
            logger.info(f"GPT-5评分: {quality_score:.3f}")
            return quality_score
            
        except Exception as e:
            logger.error(f"评分结果解析失败: {e}")
            return 0.5  # 解析失败返回中性分数
    
    def _extract_quality_score(self, result: dict) -> float:
        """从GPT-5结果提取单一质量分数
        
        Args:
            result: GPT-5返回的评分结果
            
        Returns:
            float: [0, 1]范围的质量分数
        """
        try:
            # 获取32b和8b的评分
            score_32b = result.get('score', {}).get('32b', {})
            score_8b = result.get('score', {}).get('8b', {})
            
            # 获取各维度分数
            scores_32b = score_32b.get('scores', [5, 5, 5, 5])
            scores_8b = score_8b.get('scores', [5, 5, 5, 5])
            
            # 根据better字段确定哪个模型表现更好
            better = result.get('better', 'same')
            
            # 计算8B的总分（0-40分）
            sum_8b = score_8b.get('sum', sum(scores_8b))
            sum_32b = score_32b.get('sum', sum(scores_32b))
            
            # 清晰的分段映射到[0, 1]
            if better == '8b':
                # 8B更好，映射到[0.7, 1.0]
                quality = 0.7 + (sum_8b / 40.0) * 0.3
            elif better == '32b':
                # 32B更好，映射到[0.0, 0.5]
                quality = 0.5 - (sum_32b / 40.0) * 0.5
            else:
                # 相当或both_bad，映射到[0.5, 0.7]
                quality = 0.5 + (sum_8b / 40.0) * 0.2
            
            # 限制范围
            quality = max(0.0, min(1.0, quality))
            
            logger.debug(
                f"质量分数提取: better={better}, "
                f"8B总分={sum_8b}, 32B总分={sum_32b}, "
                f"质量={quality:.3f}"
            )
            
            return quality
            
        except Exception as e:
            logger.error(f"分数提取失败: {e}")
            return 0.5  # 失败返回中性分数


# 便捷函数
async def rate_dual_models(
    llm: str,
    actor_response: dict,
    reference_response: dict
) -> float:
    """便捷函数：对双模型进行对比评分
    
    Returns:
        float: 质量分数 [0, 1]
    """
    rater = GPT5DualModelRater(llm, actor_response, reference_response)
    return await rater.rate()