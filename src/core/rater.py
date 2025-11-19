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
        self.actor_response = actor_response  # 8B模型响应
        self.reference_response = reference_response  # 32B模型响应

    async def rate(self):
        # 使用auto_eval prompt进行双模型对比评分
        # 从响应中提取需要的字段
        actor_data = self.actor_response.get("response_data", {}) if isinstance(self.actor_response, dict) else {}
        ref_data = self.reference_response.get("response_data", {}) if isinstance(self.reference_response, dict) else {}
        
        # 提取历史聊天记录（假设在reference_response中）
        history_chat = ref_data.get("history_chat", "")
        
        # 32B方案数据
        user_profile = ref_data.get("user_profile", "")
        rewritten_query = ref_data.get("rewritten_query", "")
        history_summary = ref_data.get("history_summary", "")
        rag_recall = ref_data.get("rag_recall", "")
        
        # 8B方案数据
        user_profile_8B = actor_data.get("user_profile", "")
        rewritten_query_8B = actor_data.get("rewritten_query", "")
        history_summary_8B = actor_data.get("history_summary", "")
        rag_recall_8B = actor_data.get("rag_recall", "")
        
        prompt = g_sa_prompt_manager.render_prompt(
            prompt_name="auto_eval",
            history_chat=history_chat,
            user_profile=user_profile,
            rewritten_query=rewritten_query,
            history_summary=history_summary,
            rag_recall=rag_recall,
            user_profile_8B=user_profile_8B,
            rewritten_query_8B=rewritten_query_8B,
            history_summary_8B=history_summary_8B,
            rag_recall_8B=rag_recall_8B
        )
        
        raw_response = await self.llm.ainvoke(prompt)
        content = raw_response.content
        pprint.pprint(raw_response)
        pprint.pprint(content)
        return SafeParser.parse_json_to_dict(content)
