"""
参考模型（Qwen-32B）处理器 v2.0
用于处理SalesRAG Query改写的GRPO强化学习训练
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from src.core.rag_chater import RagChater
from src.utils.log import logger

logger = logging.getLogger(__name__)


@dataclass
class ReferenceModelOutput:
    """参考模型输出数据结构"""
    user_profile: str = ""
    rewritten_query: str = ""
    history_summary: str = ""
    rag_recall: list = None
    rag_status: str = ""
    processing_time: float = 0.0
    success: bool = True
    error_message: str = ""
    query_analysis: dict = None
    intent_recognition: dict = None


class ReferenceModelProcessor:
    """参考模型（Qwen-32B）处理器"""
    
    def __init__(self, model_name: str = "Qwen3-32B-Instruct"):
        """
        初始化参考模型处理器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
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
            logger.info("参考模型RAG客户端初始化成功")
        except Exception as e:
            logger.error(f"参考模型RAG客户端初始化失败: {e}")
            raise
    
    async def process_sample(self, prompt: str, sample_data: dict) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            prompt: 完整的prompt
            sample_data: 样本数据字典
            
        Returns:
            dict: 处理结果
        """
        try:
            start_time = time.time()
            
            # 🔥 直接调用RAG /chat接口
            logger.debug(f"开始处理参考模型样本: {sample_data.get('prompt_id', 'unknown')}")
            rag_result = await self._call_rag_chat(prompt)
            
            # 解析RAG输出
            parsed_rag_output = self._parse_rag_output(rag_result)
            
            # 组合完整输出格式
            complete_output = self._combine_complete_output(
                parsed_rag_output, rag_result, sample_data, start_time
            )
            
            logger.info(f"参考模型样本处理成功: {sample_data.get('prompt_id', 'unknown')}")
            return complete_output
            
        except Exception as e:
            logger.error(f"参考模型处理失败: {e}")
            return self._get_error_output(sample_data, str(e))
    
    async def _call_rag_chat(self, prompt: str) -> Dict[str, Any]:
        """
        调用RAG /chat接口
        
        Args:
            prompt: 完整的prompt作为context
            
        Returns:
            dict: RAG调用结果
        """
        try:
            logger.debug("调用RAG /chat接口")
            
            # 使用rag_chater.py中的chat方法
            rag_result = await self.rag_client.chat(
                context=prompt,  # 🔥 传入完整prompt作为context
                score_threshold=0.95
            )
            
            response_data, status, request_body, cost_time = rag_result
            
            result = {
                "response_data": response_data,
                "status": status,
                "request_body": request_body,
                "cost_time": cost_time,
                "success": True
            }
            
            logger.debug(f"RAG /chat调用成功，耗时: {cost_time}s")
            return result
            
        except Exception as e:
            logger.error(f"RAG /chat调用失败: {e}")
            return self._get_error_rag_result(str(e))
    
    def _parse_rag_output(self, rag_result: dict) -> ReferenceModelOutput:
        """
        解析RAG输出
        
        Args:
            rag_result: RAG调用结果
            
        Returns:
            ReferenceModelOutput: 解析后的输出
        """
        try:
            response_data = rag_result["response_data"]
            status = rag_result["status"]
            
            # /chat 返回完整的数据结构
            if response_data and len(response_data) > 0:
                model_data = response_data[0].get("data", {})
                
                return ReferenceModelOutput(
                    user_profile=model_data.get("user_profile", ""),
                    rewritten_query=model_data.get("rewritten_query", ""),
                    history_summary=model_data.get("history_summary", ""),
                    rag_recall=model_data.get("recall", []),
                    rag_status=status,
                    processing_time=rag_result.get("cost_time", 0.0),
                    success=True,
                    query_analysis=model_data.get("query_analysis", {}),
                    intent_recognition=model_data.get("intent_recognition", {})
                )
            else:
                return ReferenceModelOutput(
                    rag_status=status,
                    success=False,
                    error_message="RAG返回空数据"
                )
                
        except Exception as e:
            logger.error(f"参考模型RAG输出解析失败: {e}")
            return ReferenceModelOutput(
                success=False,
                error_message=str(e)
            )
    
    def _combine_complete_output(
        self, 
        parsed_output: ReferenceModelOutput, 
        rag_result: Dict[str, Any], 
        sample_data: dict,
        start_time: float
    ) -> Dict[str, Any]:
        """
        组合完整输出格式
        
        Args:
            parsed_output: 解析后的RAG输出
            rag_result: RAG调用结果
            sample_data: 原始样本数据
            start_time: 开始时间
            
        Returns:
            dict: 完整的输出结果
        """
        try:
            processing_time = time.time() - start_time
            
            # 构建完整响应
            complete_response = {
                "user_profile": parsed_output.user_profile,
                "rewritten_query": parsed_output.rewritten_query,
                "history_summary": parsed_output.history_summary,
                "rag_recall": parsed_output.rag_recall,
                "rag_status": parsed_output.rag_status,
                "additional_info": {
                    "query_analysis": parsed_output.query_analysis,
                    "intent_recognition": parsed_output.intent_recognition,
                    "internal_processing": True
                },
                "processing_metadata": {
                    "model_name": self.model_name,
                    "endpoint": "/chat",
                    "processing_time": processing_time,
                    "rag_cost_time": parsed_output.processing_time,
                    "success": parsed_output.success
                }
            }
            
            result = {
                "prompt_id": sample_data.get("prompt_id", "unknown"),
                "complete_response": complete_response,
                "original_data": sample_data.get("parsed_data", {}),
                "rag_result": rag_result,
                "processing_success": parsed_output.success,
                "total_processing_time": processing_time
            }
            
            logger.debug(f"参考模型完整输出组合成功，总耗时: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"参考模型完整输出组合失败: {e}")
            raise
    
    def _get_error_output(self, sample_data: dict, error_message: str) -> Dict[str, Any]:
        """获取错误输出"""
        return {
            "prompt_id": sample_data.get("prompt_id", "unknown"),
            "complete_response": {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": "",
                "rag_recall": [],
                "rag_status": "error",
                "additional_info": {
                    "query_analysis": {},
                    "intent_recognition": {},
                    "internal_processing": False
                },
                "processing_metadata": {
                    "model_name": self.model_name,
                    "endpoint": "/chat",
                    "processing_time": 0.0,
                    "rag_cost_time": 0.0,
                    "success": False,
                    "error_message": error_message
                }
            },
            "original_data": sample_data.get("parsed_data", {}),
            "rag_result": {"success": False, "error_message": error_message},
            "processing_success": False,
            "total_processing_time": 0.0,
            "error_message": error_message
        }
    
    def _get_error_rag_result(self, error_message: str) -> Dict[str, Any]:
        """获取RAG错误结果"""
        return {
            "response_data": [],
            "status": "error",
            "request_body": {},
            "cost_time": 0.0,
            "success": False,
            "error_message": error_message
        }
    
    async def batch_process(self, samples: list, max_concurrency: int = 5) -> list:
        """
        批量处理样本
        
        Args:
            samples: 样本列表
            max_concurrency: 最大并发数
            
        Returns:
            list: 处理结果列表
        """
        try:
            logger.info(f"开始批量处理{len(samples)}个参考模型样本，最大并发数: {max_concurrency}")
            
            # 使用信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def process_with_semaphore(sample):
                async with semaphore:
                    return await self.process_sample(sample["prompt"], sample)
            
            # 创建所有任务
            tasks = [process_with_semaphore(sample) for sample in samples]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"参考模型样本{i}处理异常: {result}")
                    processed_results.append(self._get_error_output(samples[i], str(result)))
                else:
                    processed_results.append(result)
            
            success_count = sum(1 for r in processed_results if r["processing_success"])
            logger.info(f"参考模型批量处理完成，成功: {success_count}/{len(processed_results)}")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"参考模型批量处理失败: {e}")
            return []


class ReferenceModelManager:
    """参考模型管理器"""
    
    def __init__(self, model_name: str = "Qwen3-32B-Instruct"):
        """
        初始化参考模型管理器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.processor = None
        
    def get_processor(self) -> ReferenceModelProcessor:
        """获取参考模型处理器实例"""
        if self.processor is None:
            self.processor = ReferenceModelProcessor(self.model_name)
        return self.processor
    
    async def process_training_batch(self, training_samples: list) -> list:
        """
        处理训练批次
        
        Args:
            training_samples: 训练样本列表
            
        Returns:
            list: 处理结果列表
        """
        processor = self.get_processor()
        return await processor.batch_process(training_samples)


if __name__ == "__main__":
    # 示例用法
    async def test_reference_processor():
        processor = ReferenceModelProcessor()
        
        # 模拟样本数据
        sample_data = {
            "prompt_id": "test_001",
            "prompt": "这是一个测试prompt",
            "parsed_data": {
                "history_chat": "[销售][2024-12-09 16:01:58]:哈喽，你好！",
                "query": "[客户][2024-12-09 16:39:41]:国考和省考有什么区别？"
            }
        }
        
        result = await processor.process_sample(sample_data["prompt"], sample_data)
        print("参考模型处理结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 运行测试
    asyncio.run(test_reference_processor())