# -*- coding: utf-8 -*-
"""
DeepRetrieval × LangChain-Chatchat 集成测试
"""

import requests
import time
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """集成测试工具"""
    
    def __init__(
        self,
        langchain_api: str = "http://localhost:7861",
        query_rewrite_api: str = "http://localhost:8001/v1/chat/completions",
        kb_name: str = "wuboshi_faq"
    ):
        self.langchain_api = langchain_api
        self.query_rewrite_api = query_rewrite_api
        self.kb_name = kb_name
    
    def test_query_rewrite(self, queries: List[str]) -> List[Dict]:
        """测试查询重写功能"""
        
        logger.info("\n" + "="*60)
        logger.info("测试1: 查询重写")
        logger.info("="*60)
        
        results = []
        
        for query in queries:
            try:
                # 调用查询重写API
                response = requests.post(
                    self.query_rewrite_api,
                    json={
                        "model": "query-rewrite",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"请优化查询: {query}"
                            }
                        ],
                        "max_tokens": 256,
                        "temperature": 0.3
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    
                    result = {
                        "original": query,
                        "rewritten": content,
                        "success": True
                    }
                    
                    logger.info(f"\n原始Query: {query}")
                    logger.info(f"重写Query: {content[:100]}...")
                    
                else:
                    result = {
                        "original": query,
                        "rewritten": query,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    
                    logger.error(f"重写失败: {query}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"重写异常: {e}")
                results.append({
                    "original": query,
                    "rewritten": query,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def test_retrieval(
        self, 
        queries: List[str], 
        use_rewrite: bool = False
    ) -> List[Dict]:
        """测试检索功能"""
        
        mode = "使用查询重写" if use_rewrite else "直接检索"
        
        logger.info("\n" + "="*60)
        logger.info(f"测试2: 知识库检索 ({mode})")
        logger.info("="*60)
        
        results = []
        
        for query in queries:
            try:
                # 如果使用重写,先重写查询
                if use_rewrite:
                    rewrite_results = self.test_query_rewrite([query])
                    search_query = rewrite_results[0].get('rewritten', query)
                else:
                    search_query = query
                
                # 调用知识库检索
                response = requests.post(
                    f"{self.langchain_api}/knowledge_base/search_docs",
                    json={
                        "query": search_query,
                        "knowledge_base_name": self.kb_name,
                        "top_k": 5,
                        "score_threshold": 0.0
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    docs = response.json().get('data', [])
                    
                    result = {
                        "query": query,
                        "search_query": search_query,
                        "docs_count": len(docs),
                        "top_doc": docs[0] if docs else None,
                        "success": True
                    }
                    
                    logger.info(f"\nQuery: {query}")
                    if use_rewrite:
                        logger.info(f"重写为: {search_query[:100]}...")
                    logger.info(f"检索到 {len(docs)} 个文档")
                    
                    if docs:
                        logger.info(f"Top1 Score: {docs[0].get('score', 0):.3f}")
                        logger.info(f"Top1 Content: {docs[0]['page_content'][:150]}...")
                    
                else:
                    result = {
                        "query": query,
                        "search_query": search_query,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    
                    logger.error(f"检索失败: {query}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"检索异常: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def test_chat(
        self, 
        queries: List[str],
        use_rewrite: bool = False
    ) -> List[Dict]:
        """测试完整对话流程"""
        
        mode = "使用查询重写" if use_rewrite else "直接对话"
        
        logger.info("\n" + "="*60)
        logger.info(f"测试3: 完整对话 ({mode})")
        logger.info("="*60)
        
        results = []
        
        for query in queries:
            try:
                response = requests.post(
                    f"{self.langchain_api}/chat/knowledge_base_chat",
                    json={
                        "query": query,
                        "knowledge_base_name": self.kb_name,
                        "top_k": 3,
                        "score_threshold": 0.0,
                        "history": [],
                        "stream": False,
                        "model_name": "qwen-plus",
                        "temperature": 0.7,
                        "max_tokens": 1024,
                        "prompt_name": "default"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    answer = response.json().get('answer', '')
                    docs = response.json().get('docs', [])
                    
                    result = {
                        "query": query,
                        "answer": answer,
                        "docs_count": len(docs),
                        "success": True
                    }
                    
                    logger.info(f"\nQuery: {query}")
                    logger.info(f"Answer: {answer[:200]}...")
                    logger.info(f"参考文档数: {len(docs)}")
                    
                else:
                    result = {
                        "query": query,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    
                    logger.error(f"对话失败: {query}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"对话异常: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def compare_with_without_rewrite(
        self, 
        queries: List[str]
    ) -> Dict:
        """对比使用/不使用查询重写的效果"""
        
        logger.info("\n" + "="*60)
        logger.info("测试4: A/B对比测试")
        logger.info("="*60)
        
        # 测试不使用重写
        logger.info("\n[A组] 不使用查询重写")
        results_without = self.test_retrieval(queries, use_rewrite=False)
        
        time.sleep(2)
        
        # 测试使用重写
        logger.info("\n[B组] 使用查询重写")
        results_with = self.test_retrieval(queries, use_rewrite=True)
        
        # 对比分析
        logger.info("\n" + "="*60)
        logger.info("A/B对比结果")
        logger.info("="*60)
        
        comparison = []
        
        for i, query in enumerate(queries):
            without = results_without[i]
            with_rewrite = results_with[i]
            
            if without['success'] and with_rewrite['success']:
                score_without = without.get('top_doc', {}).get('score', 0)
                score_with = with_rewrite.get('top_doc', {}).get('score', 0)
                
                improvement = score_with - score_without
                
                comp = {
                    "query": query,
                    "score_without_rewrite": score_without,
                    "score_with_rewrite": score_with,
                    "improvement": improvement,
                    "improvement_pct": (improvement / max(score_without, 0.001)) * 100
                }
                
                logger.info(f"\nQuery: {query}")
                logger.info(f"  不重写得分: {score_without:.3f}")
                logger.info(f"  重写后得分: {score_with:.3f}")
                logger.info(f"  提升幅度: {improvement:+.3f} ({comp['improvement_pct']:+.1f}%)")
                
                comparison.append(comp)
        
        # 计算平均提升
        if comparison:
            avg_improvement = sum(c['improvement'] for c in comparison) / len(comparison)
            avg_improvement_pct = sum(c['improvement_pct'] for c in comparison) / len(comparison)
            
            logger.info("\n" + "-"*60)
            logger.info(f"平均提升: {avg_improvement:+.3f} ({avg_improvement_pct:+.1f}%)")
            logger.info("-"*60)
        
        return {
            "without_rewrite": results_without,
            "with_rewrite": results_with,
            "comparison": comparison
        }
    
    def run_all_tests(self):
        """运行所有测试"""
        
        print("\n" + "#"*60)
        print("# DeepRetrieval × LangChain-Chatchat 集成测试")
        print("#"*60)
        
        # 测试查询集
        test_queries = [
            "胶原蛋白怎么吃",
            "孕妇能喝吗",
            "早上还是晚上喝好",
            "喝多久能看到效果",
            "和其他保健品能一起吃吗"
        ]
        
        # 1. 测试查询重写
        try:
            rewrite_results = self.test_query_rewrite(test_queries)
        except Exception as e:
            logger.error(f"查询重写测试失败: {e}")
            rewrite_results = []
        
        time.sleep(1)
        
        # 2. 测试检索(不重写)
        try:
            retrieval_results = self.test_retrieval(test_queries, use_rewrite=False)
        except Exception as e:
            logger.error(f"检索测试失败: {e}")
            retrieval_results = []
        
        time.sleep(1)
        
        # 3. A/B对比测试
        try:
            comparison_results = self.compare_with_without_rewrite(test_queries)
        except Exception as e:
            logger.error(f"对比测试失败: {e}")
            comparison_results = {}
        
        # 保存测试结果
        all_results = {
            "query_rewrite": rewrite_results,
            "retrieval": retrieval_results,
            "comparison": comparison_results
        }
        
        output_file = "test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n测试结果已保存到: {output_file}")


def main():
    """主函数"""
    
    # 初始化测试器
    tester = IntegrationTester(
        langchain_api="http://localhost:7861",
        query_rewrite_api="http://localhost:8001/v1/chat/completions",
        kb_name="wuboshi_faq"
    )
    
    # 运行所有测试
    tester.run_all_tests()


if __name__ == "__main__":
    main()

