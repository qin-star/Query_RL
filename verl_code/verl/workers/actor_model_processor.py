

"""
Actor模型（Qwen-8B）处理器 v2.0
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
class ModelOutput:
    """模型输出数据结构"""
    user_profile: str = ""
    rewritten_query: str = ""
    history_summary: str = ""
    rag_recall: list = None
    rag_status: str = ""
    processing_time: float = 0.0
    success: bool = True
    error_message: str = ""


class ActorModelProcessor:
    """Actor模型（Qwen-8B）处理器"""
    
    def __init__(self, model_name: str = "Qwen3-8B-Instruct"):
        """
        初始化Actor模型处理器
        
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
            logger.info("RAG客户端初始化成功")
        except Exception as e:
            logger.error(f"RAG客户端初始化失败: {e}")
            raise
    
    async def process_sample(self, prompt: str, sample_data: dict) -> Dict[str, Any]:
        """
        处理单个训练样本
        
        Args:
            prompt: 完整的prompt
            sample_data: 样本数据字典
            
        Returns:
            dict: 处理结果
        """
        try:
            start_time = time.time()
            
            # 1. 模型生成结构化输出
            logger.debug(f"开始处理样本: {sample_data.get('prompt_id', 'unknown')}")
            model_output = await self._generate_model_output(prompt)
            
            # 2. 解析JSON输出
            parsed_output = self._parse_model_output(model_output)
            
            # 3. 🔥 调用RAG /chat_8b接口
            rag_result = await self._call_rag_chat_8b(
                user_profile=parsed_output["user_profile"],
                rewritten_query=parsed_output["rewritten_query"],
                history_summary=parsed_output["history_summary"]
            )
            
            # 4. 重新组合完整输出格式
            complete_output = self._combine_complete_output(
                parsed_output, rag_result, sample_data, start_time
            )
            
            logger.info(f"样本处理成功: {sample_data.get('prompt_id', 'unknown')}")
            return complete_output
            
        except Exception as e:
            logger.error(f"Actor模型处理失败: {e}")
            return self._get_error_output(sample_data, str(e))
    
    async def _generate_model_output(self, prompt: str) -> str:
        """
        生成模型输出
        
        Args:
            prompt: 输入prompt
            
        Returns:
            str: 模型生成的文本
        """
        try:
            logger.debug("生成模型输出")
            
            # 使用本地Qwen-8B模型生成输出
            import sys
            import os
            
            # 添加src目录到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # 回到verl_code目录
            src_path = os.path.join(project_root, '..', 'src')  # 指向 /home/jovyan2/query_rl/src
            
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            try:
                from src.utils.llm import get_chat_llm
            except ImportError as e:
                print(f"⚠️  导入src模块失败: {e}")
                # 创建模拟函数以避免崩溃
                def get_chat_llm(llm_name):
                    return lambda x: f"模拟{llm_name}响应"
            
            llm = get_chat_llm("qwen3-8b")
            response = await llm.ainvoke(prompt)
            
            # 验证输出格式，确保是有效的JSON
            model_output = response.content
            
            # 尝试解析JSON，确保格式正确
            try:
                parsed_output = json.loads(model_output)
                # 确保包含必需字段
                required_fields = ["user_profile", "rewritten_query", "history_summary"]
                for field in required_fields:
                    if field not in parsed_output:
                        logger.warning(f"模型输出缺少必需字段: {field}")
                        parsed_output[field] = ""
                
                # 重新序列化为JSON字符串
                return json.dumps(parsed_output, ensure_ascii=False)
                
            except json.JSONDecodeError as json_error:
                logger.error(f"模型输出不是有效的JSON格式: {json_error}")
                # 如果不是JSON格式，返回结构化的默认值
                default_output = {
                    "user_profile": "",
                    "rewritten_query": "",
                    "history_summary": ""
                }
                return json.dumps(default_output, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"模型生成失败: {e}")
            # 返回默认的JSON结构而不是抛出异常
            default_output = {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": ""
            }
            return json.dumps(default_output, ensure_ascii=False)
    
    def _parse_model_output(self, model_output: str) -> Dict[str, str]:
        """
        解析模型输出的JSON
        
        Args:
            model_output: 模型生成的文本
            
        Returns:
            dict: 解析后的结构化数据
        """
        try:
            # 尝试解析JSON
            parsed = json.loads(model_output.strip())
            
            # 验证必需字段
            required_fields = ["user_profile", "rewritten_query", "history_summary"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = ""
            
            logger.debug(f"模型输出解析成功: {list(parsed.keys())}")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            # 返回默认值
            return {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": ""
            }
        except Exception as e:
            logger.error(f"模型输出解析失败: {e}")
            raise
    
    async def _call_rag_chat_8b(
        self, 
        user_profile: str, 
        rewritten_query: str, 
        history_summary: str
    ) -> Dict[str, Any]:
        """
        调用RAG /chat_8b接口
        
        Args:
            user_profile: 用户画像
            rewritten_query: 重写查询
            history_summary: 历史摘要
            
        Returns:
            dict: RAG调用结果
        """
        try:
            logger.debug("调用RAG /chat_8b接口")
            
            # 使用rag_chater.py中的chat_8b方法
            rag_result = await self.rag_client.chat_8b(
                user_profile=user_profile,
                rewritten_query=rewritten_query,
                history_summary=history_summary,
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
            
            logger.debug(f"RAG /chat_8b调用成功，耗时: {cost_time}s")
            return result
            
        except Exception as e:
            logger.error(f"RAG /chat_8b调用失败: {e}")
            return self._get_error_rag_result(str(e))
    
    def _combine_complete_output(
        self, 
        parsed_output: Dict[str, str], 
        rag_result: Dict[str, Any], 
        sample_data: dict,
        start_time: float
    ) -> Dict[str, Any]:
        """
        组合完整输出格式
        
        Args:
            parsed_output: 解析后的模型输出
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
                "user_profile": parsed_output["user_profile"],
                "rewritten_query": parsed_output["rewritten_query"],
                "history_summary": parsed_output["history_summary"],
                "rag_recall": rag_result["response_data"] if rag_result["success"] else [],
                "rag_status": rag_result["status"],
                "processing_metadata": {
                    "model_name": self.model_name,
                    "endpoint": "/chat_8b",
                    "processing_time": processing_time,
                    "rag_cost_time": rag_result.get("cost_time", 0.0),
                    "success": rag_result["success"]
                }
            }
            
            result = {
                "prompt_id": sample_data.get("prompt_id", "unknown"),
                "complete_response": complete_response,
                "original_data": sample_data.get("parsed_data", {}),
                "model_output": parsed_output,
                "rag_result": rag_result,
                "processing_success": True,
                "total_processing_time": processing_time
            }
            
            logger.debug(f"完整输出组合成功，总耗时: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"完整输出组合失败: {e}")
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
                "processing_metadata": {
                    "model_name": self.model_name,
                    "endpoint": "/chat_8b",
                    "processing_time": 0.0,
                    "rag_cost_time": 0.0,
                    "success": False,
                    "error_message": error_message
                }
            },
            "original_data": sample_data.get("parsed_data", {}),
            "model_output": {},
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
            logger.info(f"开始批量处理{len(samples)}个样本，最大并发数: {max_concurrency}")
            
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
                    logger.error(f"样本{i}处理异常: {result}")
                    processed_results.append(self._get_error_output(samples[i], str(result)))
                else:
                    processed_results.append(result)
            
            success_count = sum(1 for r in processed_results if r["processing_success"])
            logger.info(f"批量处理完成，成功: {success_count}/{len(processed_results)}")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return []


class ActorModelManager:
    """Actor模型管理器"""
    
    def __init__(self, model_name: str = "Qwen3-8B-Instruct"):
        """
        初始化Actor模型管理器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.processor = None
        
    def get_processor(self) -> ActorModelProcessor:
        """获取Actor模型处理器实例"""
        if self.processor is None:
            self.processor = ActorModelProcessor(self.model_name)
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
    async def test_actor_processor():
        processor = ActorModelProcessor()
        
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
        print("处理结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 运行测试
    asyncio.run(test_actor_processor())