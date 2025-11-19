"""
统一RAG接口调用管理器 v2.0
用于管理Actor模型和参考模型的并行RAG调用
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import time
from dataclasses import dataclass

from src.core.rag_chater import RagChater
from src.utils.log import logger

logger = logging.getLogger(__name__)


@dataclass
class RAGCallResult:
    """RAG调用结果"""
    endpoint: str
    response_data: list = None
    status: str = ""
    request_body: dict = None
    cost_time: float = 0.0
    success: bool = True
    error_message: str = ""


class UnifiedRAGInterface:
    """统一RAG接口调用管理器"""
    
    def __init__(self):
        """初始化统一RAG接口管理器"""
        self.rag_client = None
        self._init_rag_client()
        
    def _init_rag_client(self):
        """初始化RAG客户端"""
        try:
            self.rag_client = RagChater(
                tenant_id="chengla",
                contact_id="chengla_query_rl_contact",
                account_id="chengla_query_rl_account",
                message_id="chengla_query_rl_message_id"
            )
            logger.info("统一RAG客户端初始化成功")
        except Exception as e:
            logger.error(f"统一RAG客户端初始化失败: {e}")
            raise
    
    async def parallel_rag_calls(
        self, 
        actor_params: dict, 
        reference_prompt: str
    ) -> Tuple[dict, dict]:
        """
        并行调用两个RAG接口
        
        Args:
            actor_params: Actor模型RAG参数
            reference_prompt: 参考模型RAG prompt
            
        Returns:
            tuple: (actor_result, reference_result)
        """
        try:
            logger.debug("开始并行RAG调用")
            
            # 创建异步任务
            actor_task = self.call_actor_rag(**actor_params)
            reference_task = self.call_reference_rag(reference_prompt)
            
            # 🔥 并行执行
            actor_result, reference_result = await asyncio.gather(
                actor_task, reference_task, return_exceptions=True
            )
            
            # 处理异常情况
            if isinstance(actor_result, Exception):
                actor_result = self._get_error_result("/chat_8b", str(actor_result))
            
            if isinstance(reference_result, Exception):
                reference_result = self._get_error_result("/chat", str(reference_result))
            
            logger.debug("并行RAG调用完成")
            return actor_result, reference_result
            
        except Exception as e:
            logger.error(f"并行RAG调用失败: {e}")
            return self._get_error_result("/chat_8b"), self._get_error_result("/chat")
    
    async def call_actor_rag(
        self,
        context: str,
        user_profile: str,
        rewritten_query: str, 
        history_summary: str,
        score_threshold: float = 0.95
    ) -> dict:
        """
        调用Actor模型的RAG接口（/chat_8b）
        
        Args:
            context: 对话上下文（必需）
            user_profile: 用户画像
            rewritten_query: 重写查询
            history_summary: 历史摘要
            score_threshold: 分数阈值
            
        Returns:
            dict: RAG调用结果
        """
        try:
            logger.debug("调用Actor RAG /chat_8b接口")
            
            rag_result = await self.rag_client.chat_8b(
                context=context,
                user_profile=user_profile,
                rewritten_query=rewritten_query,
                history_summary=history_summary,
                score_threshold=score_threshold
            )
            
            response_data, status, request_body, cost_time = rag_result
            
            return RAGCallResult(
                endpoint="/chat_8b",
                response_data=response_data,
                status=status,
                request_body=request_body,
                cost_time=cost_time,
                success=True
            ).__dict__
            
        except Exception as e:
            logger.error(f"Actor RAG /chat_8b调用失败: {e}")
            return self._get_error_result("/chat_8b", str(e))
    
    async def call_reference_rag(
        self,
        context: str,
        score_threshold: float = 0.95
    ) -> dict:
        """
        调用参考模型的RAG接口（/chat）
        
        Args:
            context: 完整的prompt作为context
            score_threshold: 分数阈值
            
        Returns:
            dict: RAG调用结果
        """
        try:
            logger.debug("调用参考模型 RAG /chat接口")
            
            rag_result = await self.rag_client.chat(
                context=context,
                score_threshold=score_threshold
            )
            
            response_data, status, request_body, cost_time = rag_result
            
            return RAGCallResult(
                endpoint="/chat",
                response_data=response_data,
                status=status,
                request_body=request_body,
                cost_time=cost_time,
                success=True
            ).__dict__
            
        except Exception as e:
            logger.error(f"参考模型 RAG /chat调用失败: {e}")
            return self._get_error_result("/chat", str(e))
    
    def _get_error_result(self, endpoint: str, error_message: str) -> dict:
        """获取错误结果"""
        return RAGCallResult(
            endpoint=endpoint,
            status="error",
            success=False,
            error_message=error_message
        ).__dict__
    
    async def process_dual_model_batch(
        self, 
        actor_samples: List[dict], 
        reference_samples: List[dict],
        max_concurrency: int = 3
    ) -> List[dict]:
        """
        处理双模型批次
        
        Args:
            actor_samples: Actor模型样本列表
            reference_samples: 参考模型样本列表
            max_concurrency: 最大并发数
            
        Returns:
            list: 处理结果列表
        """
        try:
            logger.info(f"开始处理双模型批次，Actor: {len(actor_samples)}, Reference: {len(reference_samples)}")
            
            if len(actor_samples) != len(reference_samples):
                raise ValueError("Actor和参考模型样本数量不匹配")
            
            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def process_pair(actor_sample, reference_sample):
                async with semaphore:
                    return await self._process_sample_pair(actor_sample, reference_sample)
            
            # 创建所有任务
            tasks = []
            for actor_sample, reference_sample in zip(actor_samples, reference_samples):
                task = process_pair(actor_sample, reference_sample)
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"样本对{i}处理异常: {result}")
                    processed_results.append(self._get_error_sample_pair(
                        actor_samples[i], reference_samples[i], str(result)
                    ))
                else:
                    processed_results.append(result)
            
            success_count = sum(1 for r in processed_results if r.get("dual_success", False))
            logger.info(f"双模型批次处理完成，成功: {success_count}/{len(processed_results)}")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"双模型批次处理失败: {e}")
            return []
    
    async def _process_sample_pair(self, actor_sample: dict, reference_sample: dict) -> dict:
        """
        处理单个样本对
        
        Args:
            actor_sample: Actor模型样本
            reference_sample: 参考模型样本
            
        Returns:
            dict: 处理结果
        """
        try:
            start_time = time.time()
            
            # 准备Actor模型参数（需要context）
            actor_params = {
                "context": actor_sample.get("context", ""),  # 🔥 添加context参数
                "user_profile": actor_sample.get("user_profile", ""),
                "rewritten_query": actor_sample.get("rewritten_query", ""),
                "history_summary": actor_sample.get("history_summary", "")
            }
            
            # 准备参考模型参数
            reference_prompt = reference_sample.get("prompt", "")
            
            # 并行调用RAG
            actor_result, reference_result = await self.parallel_rag_calls(
                actor_params, reference_prompt
            )
            
            # 组合结果
            processing_time = time.time() - start_time
            
            result = {
                "prompt_id": actor_sample.get("prompt_id", "unknown"),
                "actor_result": actor_result,
                "reference_result": reference_result,
                "dual_success": actor_result.get("success", False) and reference_result.get("success", False),
                "processing_time": processing_time,
                "total_rag_time": actor_result.get("cost_time", 0.0) + reference_result.get("cost_time", 0.0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"样本对处理失败: {e}")
            return self._get_error_sample_pair(actor_sample, reference_sample, str(e))
    
    def _get_error_sample_pair(self, actor_sample: dict, reference_sample: dict, error_message: str) -> dict:
        """获取错误样本对结果"""
        return {
            "prompt_id": actor_sample.get("prompt_id", "unknown"),
            "actor_result": self._get_error_result("/chat_8b", error_message),
            "reference_result": self._get_error_result("/chat", error_message),
            "dual_success": False,
            "processing_time": 0.0,
            "total_rag_time": 0.0,
            "error_message": error_message
        }


class RAGOutputParser:
    """RAG输出解析器"""
    
    @staticmethod
    def parse_actor_output(rag_result: dict) -> dict:
        """
        解析Actor模型的RAG输出 (/chat_8b)
        
        Args:
            rag_result: RAG调用结果
            
        Returns:
            dict: 解析后的输出
        """
        try:
            response_data = rag_result.get("response_data", [])
            status = rag_result.get("status", "")
            
            # /chat_8b 返回直接的检索结果列表
            return {
                "user_profile": "",  # 由外部传入，不在RAG响应中
                "rewritten_query": "",  # 由外部传入，不在RAG响应中
                "history_summary": "",  # 由外部传入，不在RAG响应中
                "rag_recall": response_data if response_data else [],
                "rag_status": status,
                "processing_metadata": {
                    "endpoint": "/chat_8b",
                    "cost_time": rag_result.get("cost_time", 0.0),
                    "success": rag_result.get("success", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Actor RAG输出解析失败: {e}")
            return RAGOutputParser._get_default_output("/chat_8b")
    
    @staticmethod
    def parse_reference_output(rag_result: dict) -> dict:
        """
        解析参考模型的RAG输出 (/chat)
        
        Args:
            rag_result: RAG调用结果
            
        Returns:
            dict: 解析后的输出
        """
        try:
            response_data = rag_result.get("response_data", [])
            status = rag_result.get("status", "")
            
            # /chat 返回完整的数据结构
            if response_data and len(response_data) > 0:
                model_data = response_data[0].get("data", {})
                
                return {
                    "user_profile": model_data.get("user_profile", ""),
                    "rewritten_query": model_data.get("rewritten_query", ""),
                    "history_summary": model_data.get("history_summary", ""),
                    "rag_recall": model_data.get("recall", []),
                    "rag_status": status,
                    "additional_info": {
                        "query_analysis": model_data.get("query_analysis", {}),
                        "intent_recognition": model_data.get("intent_recognition", {}),
                        "internal_processing": True
                    }
                }
            else:
                return RAGOutputParser._get_default_output("/chat")
                
        except Exception as e:
            logger.error(f"Reference RAG输出解析失败: {e}")
            return RAGOutputParser._get_default_output("/chat")
    
    @staticmethod
    def _get_default_output(endpoint: str) -> dict:
        """获取默认输出"""
        if endpoint == "/chat_8b":
            return {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": "",
                "rag_recall": [],
                "rag_status": "error",
                "processing_metadata": {
                    "endpoint": "/chat_8b",
                    "cost_time": 0.0,
                    "success": False
                }
            }
        else:  # /chat
            return {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": "",
                "rag_recall": [],
                "rag_status": "error",
                "additional_info": {
                    "query_analysis": {},
                    "intent_recognition": {},
                    "internal_processing": False
                }
            }


class UnifiedRAGManager:
    """统一RAG管理器"""
    
    def __init__(self):
        """初始化统一RAG管理器"""
        self.interface = None
        self.parser = RAGOutputParser()
        
    def get_interface(self) -> UnifiedRAGInterface:
        """获取统一RAG接口实例"""
        if self.interface is None:
            self.interface = UnifiedRAGInterface()
        return self.interface
    
    def get_parser(self) -> RAGOutputParser:
        """获取RAG输出解析器实例"""
        return self.parser
    
    async def process_training_batch(self, actor_results: list, reference_results: list) -> list:
        """
        处理训练批次
        
        Args:
            actor_results: Actor模型处理结果列表
            reference_results: 参考模型处理结果列表
            
        Returns:
            list: 解析后的结果列表
        """
        try:
            logger.info(f"开始解析训练批次，Actor: {len(actor_results)}, Reference: {len(reference_results)}")
            
            if len(actor_results) != len(reference_results):
                raise ValueError("Actor和参考模型结果数量不匹配")
            
            parsed_results = []
            
            for actor_result, reference_result in zip(actor_results, reference_results):
                try:
                    # 解析Actor模型结果
                    actor_parsed = self.parser.parse_actor_output(
                        actor_result.get("rag_result", {})
                    )
                    
                    # 解析参考模型结果
                    reference_parsed = self.parser.parse_reference_output(
                        reference_result.get("rag_result", {})
                    )
                    
                    # 组合解析结果
                    parsed_result = {
                        "prompt_id": actor_result.get("prompt_id", "unknown"),
                        "actor_parsed": actor_parsed,
                        "reference_parsed": reference_parsed,
                        "actor_success": actor_result.get("processing_success", False),
                        "reference_success": reference_result.get("processing_success", False),
                        "dual_success": (
                            actor_result.get("processing_success", False) and 
                            reference_result.get("processing_success", False)
                        )
                    }
                    
                    parsed_results.append(parsed_result)
                    
                except Exception as e:
                    logger.error(f"解析样本对失败: {e}")
                    # 添加错误结果
                    parsed_results.append({
                        "prompt_id": actor_result.get("prompt_id", "unknown"),
                        "actor_parsed": self.parser._get_default_output("/chat_8b"),
                        "reference_parsed": self.parser._get_default_output("/chat"),
                        "actor_success": False,
                        "reference_success": False,
                        "dual_success": False,
                        "error_message": str(e)
                    })
            
            success_count = sum(1 for r in parsed_results if r.get("dual_success", False))
            logger.info(f"训练批次解析完成，成功: {success_count}/{len(parsed_results)}")
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"训练批次解析失败: {e}")
            return []


if __name__ == "__main__":
    # 示例用法
    async def test_unified_rag():
        interface = UnifiedRAGInterface()
        
        # 测试并行调用
        actor_params = {
            "context": "测试对话上下文",  # 🔥 添加context参数
            "user_profile": "测试用户画像",
            "rewritten_query": "测试重写查询",
            "history_summary": "测试历史摘要"
        }
        reference_prompt = "测试参考模型prompt"
        
        actor_result, reference_result = await interface.parallel_rag_calls(
            actor_params, reference_prompt
        )
        
        print("Actor RAG结果:")
        print(json.dumps(actor_result, ensure_ascii=False, indent=2))
        print("Reference RAG结果:")
        print(json.dumps(reference_result, ensure_ascii=False, indent=2))
    
    # 运行测试
    asyncio.run(test_unified_rag())