import asyncio

from src.core.rag_chater import RagChater
from src.core.rater import Rater
from src.utils.log import logger

LLM = "gpt-5"


async def get_rag_rl_result(context: str, 
                            user_profile: str = "", history_summary: str = "", rewritten_query: str = "") -> list:
    logger.info("get_rag_rl_result")
    rag = RagChater(
         tenant_id="chengla",
         contact_id="chengla_query_rl_contact",
         account_id="chengla_query_rl_account",
         message_id="chengla_query_rl_message_id"
    )
    tasks = [
        rag.chat(context= context), 
        rag.chat_8b(context=context, user_profile=user_profile, history_summary=history_summary, rewritten_query=rewritten_query)
    ]

    chat_resp, chat_8b_resp = await asyncio.gather(*tasks)

    response_data, status, request_body, cost_time  = chat_resp
    response_data_8b, status_8b, request_body_8b, cost_time_8b = chat_8b_resp

    return response_data, response_data_8b

async def get_rate_result(payload: dict) -> dict:
    logger.info("get_rate_result")
    rater = Rater(llm=LLM, payload=payload)
    return await rater.rate()
