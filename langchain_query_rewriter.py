# -*- coding: utf-8 -*-
"""
LangChain-Chatchat 查询重写模块
用于在LangChain-Chatchat中集成DeepRetrieval训练的Query重写模型
"""

from openai import OpenAI
import re
import json
import logging
from typing import Optional, Dict, List
from functools import lru_cache
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询重写器 - 基于DeepRetrieval训练的模型"""
    
    def __init__(
        self, 
        api_url: str = "http://localhost:8001/v1/chat/completions",
        model_name: str = "query-rewrite",
        cache_size: int = 1000,
        timeout: float = 2.0
    ):
        """
        初始化查询重写器
        
        Args:
            api_url: vLLM API地址
            model_name: 模型名称
            cache_size: 缓存大小
            timeout: 超时时间(秒)
        """
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_url,
            timeout=timeout
        )
        self.model_name = model_name
        self.cache_size = cache_size
        
        logger.info(f"QueryRewriter初始化完成 - API: {api_url}, Model: {model_name}")
    
    @lru_cache(maxsize=1000)
    def rewrite(
        self, 
        query: str, 
        context: str = "",
        domain: str = "保健品知识问答"
    ) -> str:
        """
        重写用户查询,使其更适合检索
        
        Args:
            query: 原始用户查询
            context: 对话上下文(可选)
            domain: 领域描述
            
        Returns:
            重写后的查询
        """
        
        start_time = time.time()
        
        try:
            # 构建提示词
            prompt = self._build_prompt(query, context, domain)
            
            # 调用模型
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
                top_p=0.9
            )
            
            # 解析结果
            content = response.choices[0].message.content
            rewritten_query = self._parse_response(content, query)
            
            latency = (time.time() - start_time) * 1000
            
            logger.info(f"Query重写完成 [{latency:.1f}ms]: '{query}' -> '{rewritten_query}'")
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Query重写失败: {e}, 返回原始query")
            return query
    
    def _build_prompt(self, query: str, context: str, domain: str) -> str:
        """构建查询重写提示词"""
        
        prompt = f"""你是一个专业的{domain}查询优化助手。

请分析以下用户查询,并将其重写为更适合知识库检索的形式。

原始查询: {query}
"""
        
        if context:
            prompt += f"对话上下文: {context}\n"
        
        prompt += """
优化要求:
1. 提取核心关键词
2. 补充必要的领域术语
3. 消除歧义和口语化表达
4. 保持查询简洁性
5. 如果是产品咨询,明确产品名称

输出格式:
<think>分析查询意图和关键信息...</think>
<answer>{"query": "优化后的查询"}</answer>
"""
        
        return prompt
    
    def _parse_response(self, content: str, fallback: str) -> str:
        """解析模型响应"""
        
        try:
            # 提取<answer>标签内容
            match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if match:
                answer_json = match.group(1).strip()
                result = json.loads(answer_json)
                return result.get("query", fallback)
            else:
                logger.warning("未找到<answer>标签,返回原始query")
                return fallback
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 返回原始query")
            return fallback
    
    def rewrite_with_fallback(
        self, 
        query: str, 
        context: str = ""
    ) -> Dict[str, str]:
        """
        带降级策略的查询重写
        
        Returns:
            {
                "original": 原始查询,
                "rewritten": 重写后的查询,
                "method": 重写方法 (model/rule/none)
            }
        """
        
        try:
            # 尝试使用模型重写
            rewritten = self.rewrite(query, context)
            
            if rewritten != query:
                return {
                    "original": query,
                    "rewritten": rewritten,
                    "method": "model"
                }
            else:
                # 模型未能重写,使用规则方法
                rewritten = self._rule_based_rewrite(query)
                return {
                    "original": query,
                    "rewritten": rewritten,
                    "method": "rule" if rewritten != query else "none"
                }
                
        except Exception as e:
            logger.error(f"重写失败: {e}")
            return {
                "original": query,
                "rewritten": query,
                "method": "none"
            }
    
    def _rule_based_rewrite(self, query: str) -> str:
        """基于规则的简单查询重写(降级方案)"""
        
        # 规则1: 产品使用方法查询
        if any(word in query for word in ["怎么用", "怎么吃", "怎么喝", "如何服用"]):
            if "胶原蛋白" in query:
                return f"胶原蛋白肽 服用方法 用量 时间"
            elif "虾青素" in query:
                return f"虾青素 服用方法 用量"
            elif "富铁" in query or "软糖" in query:
                return f"富铁软糖 使用方法 用量"
        
        # 规则2: 效果查询
        if any(word in query for word in ["效果", "多久", "见效"]):
            if "胶原蛋白" in query:
                return f"胶原蛋白肽 效果 见效时间"
        
        # 规则3: 禁忌查询
        if any(word in query for word in ["孕妇", "备孕", "哺乳"]):
            return f"{query} 禁忌 注意事项"
        
        # 规则4: 副作用查询
        if "副作用" in query or "安全" in query:
            return f"{query} 安全性 注意事项"
        
        # 默认: 返回原始查询
        return query
    
    def batch_rewrite(
        self, 
        queries: List[str], 
        context: str = ""
    ) -> List[Dict[str, str]]:
        """
        批量查询重写
        
        Args:
            queries: 查询列表
            context: 共享的上下文
            
        Returns:
            重写结果列表
        """
        
        results = []
        for query in queries:
            result = self.rewrite_with_fallback(query, context)
            results.append(result)
        
        return results


class QueryRewriteLogger:
    """查询重写日志记录器"""
    
    def __init__(self, log_file: str = "query_rewrite.log"):
        self.log_file = log_file
        
        # 配置文件日志
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        
        self.logger = logging.getLogger("QueryRewriteLogger")
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_rewrite(
        self, 
        original: str,
        rewritten: str,
        method: str,
        latency_ms: float,
        success: bool = True
    ):
        """记录查询重写日志"""
        
        log_data = {
            "original": original,
            "rewritten": rewritten,
            "method": method,
            "latency_ms": latency_ms,
            "success": success
        }
        
        self.logger.info(json.dumps(log_data, ensure_ascii=False))


# ============= 集成示例 =============

def integrate_with_langchain_chatchat():
    """
    在LangChain-Chatchat中集成查询重写的示例代码
    
    集成位置: Langchain-Chatchat/server/knowledge_base/kb_service/base.py
    """
    
    example_code = '''
# 在 KBService 类中添加

from langchain_query_rewriter import QueryRewriter, QueryRewriteLogger

class KBService:
    def __init__(self, ...):
        # ... 原有初始化代码 ...
        
        # 添加查询重写器
        self.query_rewriter = QueryRewriter(
            api_url="http://localhost:8001/v1/chat/completions"
        )
        self.rewrite_logger = QueryRewriteLogger()
    
    def search_docs(
        self, 
        query: str, 
        top_k: int = 10,
        use_query_rewrite: bool = True,
        **kwargs
    ):
        """知识库检索"""
        
        import time
        start_time = time.time()
        
        # 1. Query重写
        if use_query_rewrite:
            result = self.query_rewriter.rewrite_with_fallback(
                query=query,
                context=kwargs.get("history", "")
            )
            
            optimized_query = result["rewritten"]
            rewrite_method = result["method"]
            
            # 记录日志
            latency = (time.time() - start_time) * 1000
            self.rewrite_logger.log_rewrite(
                original=query,
                rewritten=optimized_query,
                method=rewrite_method,
                latency_ms=latency
            )
            
            logger.info(f"Query重写[{rewrite_method}]: {query} -> {optimized_query}")
        else:
            optimized_query = query
        
        # 2. 使用重写后的query进行向量检索
        docs = self.do_search(
            query=optimized_query,
            top_k=top_k * 2  # 先召回更多候选
        )
        
        # 3. (可选) 使用原始query进行重排序
        if use_query_rewrite and len(docs) > 0:
            docs = self.rerank(
                query=query,  # 使用原始query
                docs=docs,
                top_k=top_k
            )
        
        return docs[:top_k]
'''
    
    return example_code


if __name__ == "__main__":
    # 测试查询重写器
    print("=" * 60)
    print("查询重写器测试")
    print("=" * 60)
    
    # 注意: 需要先启动vLLM服务
    # vllm serve <model_path> --port 8001
    
    try:
        rewriter = QueryRewriter(api_url="http://localhost:8001/v1/chat/completions")
        
        test_queries = [
            "胶原蛋白怎么吃",
            "孕妇能喝吗",
            "早上还是晚上喝好",
            "喝多久能看到效果"
        ]
        
        for query in test_queries:
            result = rewriter.rewrite_with_fallback(query)
            print(f"\n原始: {result['original']}")
            print(f"重写: {result['rewritten']}")
            print(f"方法: {result['method']}")
            print("-" * 60)
            
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保vLLM服务已启动!")
        
    # 输出集成示例代码
    print("\n" + "=" * 60)
    print("LangChain-Chatchat 集成示例代码")
    print("=" * 60)
    print(integrate_with_langchain_chatchat())

