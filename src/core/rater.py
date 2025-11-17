import pprint
from src.utils.llm import get_chat_llm, SafeParser
from src.utils.prompt import g_sa_prompt_manager


class Rater:
    def __init__(
        self,
        llm: str,
        actor_response: dict,
        reference_response: dict
    ):
        self.llm = get_chat_llm(llm)
        self.payload = {
            "actor_response": actor_response,
            "reference_response": reference_response,
            "comparison_metrics": ["quality_improvement", "relevance_accuracy", "info_completeness", "retrieval_effectiveness"]
        }

    async def rate(self):
        # 使用双模型对比评分模板
        prompt = g_sa_prompt_manager.render_prompt(
            prompt_name="gpt5_dual_model_rating",
            actor_response=self.payload['actor_response'],
            reference_response=self.payload['reference_response'],
            comparison_metrics=self.payload['comparison_metrics']
        )
        raw_response = await self.llm.ainvoke(prompt)
        content = raw_response.content
        pprint.pprint(raw_response)
        pprint.pprint(content)
        return SafeParser.parse_json_to_dict(content)
