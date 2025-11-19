import asyncio

from src.core.rag_chater import RagChater
from src.core.rater import Rater
from src.utils.log import logger

LLM = "gpt-5"


async def get_rag_rl_result(context: str, 
                            user_profile: str = "", history_summary: str = "", rewritten_query: str = "") -> list:
    logger.info("get_rag_rl_result")
    logger.debug(f"RAG调用参数:")
    logger.debug(f"  - context长度: {len(context)}")
    logger.debug(f"  - user_profile: {user_profile[:100] if user_profile else '(空)'}")
    logger.debug(f"  - history_summary: {history_summary[:100] if history_summary else '(空)'}")
    logger.debug(f"  - rewritten_query: {rewritten_query[:100] if rewritten_query else '(空)'}")
    
    rag = RagChater(
         tenant_id="chengla",
         contact_id="Customer_knowledge_17",
         account_id="Sale_knowledge_17",
         message_id="chengla_query_rl_message_id"
    )
    
    # 32B使用原始context
    # 8B使用原始context + 生成的user_profile, history_summary, rewritten_query
    tasks = [
        rag.chat(
            context=context,
            score_threshold=0.9,
            top_k=3
        ), 
        rag.chat_8b(
            context=context,  # 8B也需要传入context
            user_profile=user_profile, 
            history_summary=history_summary, 
            rewritten_query=rewritten_query,
            score_threshold=0.9,
            top_k=3
        )
    ]

    chat_resp, chat_8b_resp = await asyncio.gather(*tasks)

    response_data, status, request_body, cost_time = chat_resp
    response_data_8b, status_8b, request_body_8b, cost_time_8b = chat_8b_resp
    
    logger.debug(f"RAG调用结果:")
    logger.debug(f"  - 32B状态: {status}, 数据长度: {len(response_data) if response_data else 0}")
    logger.debug(f"  - 8B状态: {status_8b}, 数据长度: {len(response_data_8b) if response_data_8b else 0}")

    return response_data, response_data_8b

async def get_rate_result(payload: dict) -> dict:
    logger.info("get_rate_result")
    # 从payload中提取RAG响应数据
    chat_resp = payload.get("chat_resp", [])
    chat_8b_resp = payload.get("chat_8b_resp", [])
    
    # 构建符合Rater期望的数据结构
    # 32B reference响应
    reference_response = {
        "response_data": {
            "history_chat": payload.get("history_chat", ""),
            "user_profile": payload.get("user_profile", ""),
            "rewritten_query": payload.get("rewritten_query", ""),
            "history_summary": payload.get("history_summary", ""),
            "rag_recall": str(chat_resp) if chat_resp else ""
        }
    }
    
    # 8B actor响应
    actor_response = {
        "response_data": {
            "user_profile": payload.get("user_profile_8b", ""),
            "rewritten_query": payload.get("rewritten_query_8b", ""),
            "history_summary": payload.get("history_summary_8b", ""),
            "rag_recall": str(chat_8b_resp) if chat_8b_resp else ""
        }
    }
    
    rater = Rater(llm=LLM, actor_response=actor_response, reference_response=reference_response)
    return await rater.rate()
