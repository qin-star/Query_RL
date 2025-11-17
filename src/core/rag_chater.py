from src.constant.http_constants import RAGResponseStatus
from src.utils.http import HttpUtil
from src.utils.log import logger
from src.utils.settings import SETTINGS
from src.utils.time import TimingContext


class RagChater():
    def __init__(
        self,
        tenant_id: str,
        contact_id: str,
        account_id: str,
        message_id: str,
    ) -> None:
        self.tenant_id = tenant_id
        self.contact_id = contact_id
        self.account_id = account_id
        self.message_id = message_id

    def _build_base_request_body(
        self,
        thought_unit: str,
        score_threshold: float,
        **kwargs
    ) -> dict:
        """构建基础请求体
        
        Args:
            thought_unit: 思考单元
            score_threshold: 分数阈值
            **kwargs: 其他可选参数
            
        Returns:
            构建好的请求体字典
        """
        request_body = {
            "tenant_id": self.tenant_id,
            "contact_id": self.contact_id,
            "account_id": self.account_id,
            "message_id": self.message_id,
            "kb_name": self.tenant_id,
            "thought_unit": thought_unit,
            "score_threshold": score_threshold,
        }
        
        # 添加其他可选参数
        for key, value in kwargs.items():
            if value:  # 只添加非空值
                request_body[key] = value
                
        return request_body

    def _handle_contexts(
        self,
        request_body: dict,
        contexts: list[dict] = None,
        context: str = ""
    ) -> bool:
        """处理上下文参数
        
        Args:
            request_body: 请求体字典
            contexts: 上下文列表
            context: 单个上下文字符串
            
        Returns:
            bool: 处理成功返回True，否则返回False
        """
        if contexts:
            request_body["contexts"] = contexts
            return True
        elif context:
            request_body["context"] = context
            return True
        else:
            logger.warning("no contexts and no context.")
            return False

    async def _make_api_call(
        self,
        request_body: dict,
        endpoint: str = "/rag/chat"
    ) -> tuple[list[dict], str, dict, float]:
        """执行API调用并处理响应
        
        Args:
            request_body: 请求体字典
            endpoint: API端点，默认为"/rag/chat"
            
        Returns:
            tuple: (响应数据, 状态码, 请求体, 耗时)
        """
        logger.info(SETTINGS.BASIC_SETTINGS.RAG_URL)
        
        with TimingContext() as timing:
            response_data = await HttpUtil.apost(
                SETTINGS.BASIC_SETTINGS.RAG_URL + endpoint, 
                data=request_body
            )

        logger.debug(f"request: {request_body}, response: {response_data}")
        logger.info(f"RAG chat API cost time: {timing.cost_time}s")

        if response_data is None:
            logger.error("get rag chat results failed")
            return [], RAGResponseStatus.INTERNAL_SERVICE_ERROR, request_body, timing.cost_time
        
        logger.info("get rag chat results success")
        return response_data, RAGResponseStatus.SUCCESS, request_body, timing.cost_time

    async def chat(
                self,
                contexts: list[dict] = None,
                context: str = "",
                thought_unit: str = "",
                score_threshold: float = 0.95
    ) -> tuple[list[dict], str, dict, float]:
        """基础聊天方法
        适合Qwen-32B 生成
        Args:
            contexts: 上下文列表
            context: 单个上下文字符串
            thought_unit: 思考单元
            score_threshold: 分数阈值
            
        Returns:
            tuple: (响应数据, 状态码, 请求体, 耗时)
        """
        # 构建基础请求体
        request_body = self._build_base_request_body(
            thought_unit=thought_unit,
            score_threshold=score_threshold
        )
        
        # 处理上下文
        if not self._handle_contexts(request_body, contexts, context):
            return [], RAGResponseStatus.INTERNAL_SERVICE_ERROR, request_body, 0.0
            
        # 调用API
        return await self._make_api_call(request_body, "/rag/chat")
    
    async def chat_8b(
                self,
                contexts: list[dict] = None,
                context: str = "",
                thought_unit: str = "",
                score_threshold: float = 0.95,
                user_profile: str = "",
                history_summary: str = "",
                rewritten_query: str = "",
    ) -> tuple[list[dict], str, dict, float]:
        """增强版聊天方法，支持用户画像、历史摘要和重写查询
        适合Qwen-8B 生成
        Args:
            contexts: 上下文列表
            context: 单个上下文字符串
            thought_unit: 思考单元
            score_threshold: 分数阈值
            user_profile: 用户画像
            history_summary: 历史摘要
            rewritten_query: 重写查询
            
        Returns:
            tuple: (响应数据, 状态码, 请求体, 耗时)
        """
        # 构建基础请求体，包含额外参数
        request_body = self._build_base_request_body(
            thought_unit=thought_unit,
            score_threshold=score_threshold,
            user_profile=user_profile,
            history_summary=history_summary,
            rewritten_query=rewritten_query
        )
        
        # 处理上下文
        if not self._handle_contexts(request_body, contexts, context):
            return [], RAGResponseStatus.INTERNAL_SERVICE_ERROR, request_body, 0.0
            
        # 调用API
        return await self._make_api_call(request_body, "/rag/chat_8b")
