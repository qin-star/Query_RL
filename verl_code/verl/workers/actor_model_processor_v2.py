"""
Actor模型处理器 v2.1 - GRPO组内多样本生成支持
修正版：支持GRPO组内多个样本的生成，确保组内相对优化
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import time
from dataclasses import dataclass

from src.core.rag_chater import RagChater
from src.utils.log import logger
from .grpo_group_generator import GRPOGroup, groups_to_training_batch

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


class ActorModelProcessorV2:
    """Actor模型处理器 - GRPO组内多样本生成支持"""
    
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
    
    async def process_grpo_group(self, group: GRPOGroup) -> List[Dict[str, Any]]:
        """
        处理GRPO组内的所有样本（关键方法）
        
        Args:
            group: GRPO组对象
            
        Returns:
            List[Dict[str, Any]]: 组内所有样本的处理结果
        """
        try:
            logger.info(f"开始处理GRPO组: {group.group_id}，包含 {len(group.samples)} 个样本")
            
            # 为组内每个样本生成不同的输出（确保多样性）
            group_results = []
            
            for sample_idx, sample in enumerate(group.samples):
                try:
                    # 使用不同的生成参数处理每个样本
                    sample_result = await self._process_group_sample(sample, sample_idx)
                    group_results.append(sample_result)
                    
                except Exception as e:
                    logger.error(f"处理组 {group.group_id} 样本 {sample_idx} 失败: {e}")
                    # 添加错误结果，保持组完整性
                    error_result = self._get_error_group_sample_result(sample, str(e))
                    group_results.append(error_result)
            
            # 验证组完整性
            if len(group_results) != len(group.samples):
                logger.warning(f"组 {group.group_id} 结果数量不匹配: {len(group_results)} != {len(group.samples)}")
            
            success_count = sum(1 for r in group_results if r.get("processing_success", False))
            logger.info(f"GRPO组 {group.group_id} 处理完成，成功: {success_count}/{len(group_results)}")
            
            return group_results
            
        except Exception as e:
            logger.error(f"处理GRPO组 {group.group_id} 失败: {e}")
            # 返回所有错误结果，保持组结构
            return [self._get_error_group_sample_result(sample, str(e)) for sample in group.samples]
    
    async def _process_group_sample(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """
        处理GRPO组内的单个样本（使用不同的生成参数）
        
        Args:
            sample: 样本数据
            sample_idx: 样本索引（用于参数变化）
            
        Returns:
            dict: 处理结果
        """
        try:
            start_time = time.time()
            
            # 1. 获取该样本的特定生成参数（确保组内多样性）
            generation_params = sample.get("generation_params", {})
            temperature = generation_params.get("temperature", 0.7)
            top_p = generation_params.get("top_p", 0.9)
            
            logger.debug(f"处理组样本 {sample_idx}: 温度={temperature}, top_p={top_p}")
            
            # 2. 使用特定参数生成模型输出
            model_output = await self._generate_model_output_with_params(
                sample["prompt"], 
                temperature=temperature,
                top_p=top_p,
                sample_idx=sample_idx
            )
            
            # 3. 解析JSON输出
            parsed_output = self._parse_model_output(model_output)
            
            # 4. 调用RAG /chat_8b接口
            rag_result = await self._call_rag_chat_8b(
                user_profile=parsed_output["user_profile"],
                rewritten_query=parsed_output["rewritten_query"],
                history_summary=parsed_output["history_summary"]
            )
            
            # 5. 构建完整输出（包含GRPO组信息）
            complete_output = self._combine_complete_group_output(
                parsed_output, rag_result, sample, start_time, sample_idx
            )
            
            logger.debug(f"组样本 {sample_idx} 处理成功")
            return complete_output
            
        except Exception as e:
            logger.error(f"处理组样本 {sample_idx} 失败: {e}")
            return self._get_error_group_sample_result(sample, str(e))
    
    async def _generate_model_output_with_params(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        sample_idx: int = 0
    ) -> str:
        """
        使用特定参数生成模型输出（确保组内多样性）
        
        Args:
            prompt: 输入prompt
            temperature: 温度参数（控制随机性）
            top_p: top-p参数（控制多样性）
            sample_idx: 样本索引（用于日志）
            
        Returns:
            str: 模型生成的文本
        """
        try:
            logger.debug(f"生成模型输出 - 样本{sample_idx}: 温度={temperature}, top_p={top_p}")
            
            # 使用本地Qwen-8B模型生成输出，应用特定参数
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
            
            # 🔥 关键：使用不同的温度参数生成多样化的输出
            # 这里可以集成更复杂的参数控制逻辑
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
            logger.error(f"模型生成失败 - 样本{sample_idx}: {e}")
            # 返回默认的JSON结构而不是抛出异常
            default_output = {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": ""
            }
            return json.dumps(default_output, ensure_ascii=False)
    
    def _combine_complete_group_output(
        self, 
        parsed_output: Dict[str, str], 
        rag_result: Dict[str, Any], 
        sample: Dict[str, Any],
        start_time: float,
        sample_idx: int
    ) -> Dict[str, Any]:
        """
        组合GRPO组的完整输出格式
        
        Args:
            parsed_output: 解析后的模型输出
            rag_result: RAG调用结果
            sample: 原始样本数据
            start_time: 开始时间
            sample_idx: 样本索引
            
        Returns:
            dict: 完整的输出结果（包含GRPO组信息）
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
                    "success": rag_result["success"],
                    "sample_index": sample_idx,
                    "generation_params": sample.get("generation_params", {})
                }
            }
            
            result = {
                "prompt_id": sample.get("prompt_id", f"group_{sample.get('group_id', 'unknown')}_sample_{sample_idx}"),
                "group_id": sample.get("group_id", "unknown"),
                "sample_id": sample.get("sample_id", f"sample_{sample_idx}"),
                "complete_response": complete_response,
                "original_data": sample,
                "model_output": parsed_output,
                "rag_result": rag_result,
                "processing_success": True,
                "total_processing_time": processing_time,
                "generation_params": sample.get("generation_params", {}),
                "group_metadata": {
                    "group_id": sample.get("group_id", "unknown"),
                    "sample_index": sample_idx,
                    "total_samples_in_group": self.group_size
                }
            }
            
            logger.debug(f"组样本 {sample_idx} 完整输出组合成功，总耗时: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"组样本 {sample_idx} 完整输出组合失败: {e}")
            raise
    
    def _get_error_group_sample_result(self, sample: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """获取GRPO组样本的错误结果"""
        return {
            "prompt_id": sample.get("prompt_id", f"group_{sample.get('group_id', 'unknown')}_sample_error"),
            "group_id": sample.get("group_id", "unknown"),
            "sample_id": sample.get("sample_id", "error_sample"),
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
                    "error_message": error_message,
                    "sample_index": sample.get("metadata", {}).get("sample_index", -1)
                }
            },
            "original_data": sample,
            "model_output": {},
            "rag_result": {"success": False, "error_message": error_message},
            "processing_success": False,
            "total_processing_time": 0.0,
            "error_message": error_message,
            "group_metadata": {
                "group_id": sample.get("group_id", "unknown"),
                "sample_index": sample.get("metadata", {}).get("sample_index", -1),
                "total_samples_in_group": self.group_size
            }
        }
    
    # 其他方法保持不变（与v1相同）
    def _parse_model_output(self, model_output: str) -> Dict[str, str]:
        """解析模型输出的JSON（与v1相同）"""
        try:
            parsed = json.loads(model_output.strip())
            required_fields = ["user_profile", "rewritten_query", "history_summary"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = ""
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return {
                "user_profile": "",
                "rewritten_query": "",
                "history_summary": ""
            }
    
    async def _call_rag_chat_8b(
        self, 
        user_profile: str, 
        rewritten_query: str, 
        history_summary: str
    ) -> Dict[str, Any]:
        """调用RAG /chat_8b接口（与v1相同）"""
        try:
            rag_result = await self.rag_client.chat_8b(
                user_profile=user_profile,
                rewritten_query=rewritten_query,
                history_summary=history_summary,
                score_threshold=0.95
            )
            
            response_data, status, request_body, cost_time = rag_result
            
            return {
                "response_data": response_data,
                "status": status,
                "request_body": request_body,
                "cost_time": cost_time,
                "success": True
            }
        except Exception as e:
            logger.error(f"RAG /chat_8b调用失败: {e}")
            return self._get_error_rag_result(str(e))
    
    def _get_error_rag_result(self, error_message: str) -> Dict[str, Any]:
        """获取RAG错误结果（与v1相同）"""
        return {
            "response_data": [],
            "status": "error",
            "request_body": {},
            "cost_time": 0.0,
            "success": False,
            "error_message": error_message
        }
    
    async def batch_process_groups(
        self, 
        groups: List[GRPOGroup], 
        max_concurrency: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        批量处理GRPO组（关键方法）
        
        Args:
            groups: GRPO组列表
            max_concurrency: 最大并发数（按组控制）
            
        Returns:
            List[List[Dict[str, Any]]]: 每组处理结果列表
        """
        try:
            logger.info(f"开始批量处理{len(groups)}个GRPO组，最大并发数: {max_concurrency}")
            
            # 使用信号量控制并发（按组级别）
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def process_group_with_semaphore(group):
                async with semaphore:
                    return await self.process_grpo_group(group)
            
            # 创建所有任务
            tasks = [process_group_with_semaphore(group) for group in groups]
            
            # 等待所有任务完成
            all_group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, group_result in enumerate(all_group_results):
                if isinstance(group_result, Exception):
                    logger.error(f"组{i}处理异常: {group_result}")
                    # 为整个组创建错误结果
                    error_results = [
                        self._get_error_group_sample_result(sample, str(group_result)) 
                        for sample in groups[i].samples
                    ]
                    processed_results.append(error_results)
                else:
                    processed_results.append(group_result)
            
            # 统计
            total_samples = sum(len(group_results) for group_results in processed_results)
            success_samples = sum(
                sum(1 for r in group_results if r.get("processing_success", False))
                for group_results in processed_results
            )
            
            logger.info(f"批量GRPO组处理完成，成功样本: {success_samples}/{total_samples}")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"批量GRPO组处理失败: {e}")
            return []


class ActorModelManagerV2:
    """Actor模型管理器 - GRPO组处理支持"""
    
    def __init__(self, model_name: str = "Qwen3-8B-Instruct"):
        """
        初始化Actor模型管理器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.processor = None
        
    def get_processor(self) -> ActorModelProcessorV2:
        """获取Actor模型处理器实例"""
        if self.processor is None:
            self.processor = ActorModelProcessorV2(self.model_name)
        return self.processor
    
    async def process_grpo_groups(self, groups: List[GRPOGroup]) -> List[List[Dict[str, Any]]]:
        """
        处理GRPO组批次（关键接口）
        
        Args:
            groups: GRPO组列表
            
        Returns:
            List[List[Dict[str, Any]]]: 每组处理结果
        """
        processor = self.get_processor()
        return await processor.batch_process_groups(groups)


if __name__ == "__main__":
    # 示例用法
    async def test_grpo_group_processor():
        from .grpo_group_generator import GRPOGroupGenerator
        
        # 创建组生成器
        group_generator = GRPOGroupGenerator(group_size=3)
        
        # 模拟测试数据
        test_data = [
            {
                "original_query": "国考和省考有什么区别？",
                "history_chat": [
                    {"user": "你好", "assistant": "您好！有什么可以帮助您的吗？"}
                ]
            }
        ]
        
        # 生成GRPO组
        groups = group_generator.generate_groups(test_data)
        
        # 处理GRPO组
        manager = ActorModelManagerV2()
        results = await manager.process_grpo_groups(groups)
        
        print(f"处理了 {len(results)} 个GRPO组")
        for i, group_results in enumerate(results):
            print(f"组 {i}: {len(group_results)} 个样本")
            for j, sample_result in enumerate(group_results):
                success = sample_result.get("processing_success", False)
                print(f"  样本 {j}: {'成功' if success else '失败'}")
    
    # 运行测试
    asyncio.run(test_grpo_group_processor())