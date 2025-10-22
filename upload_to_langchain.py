# -*- coding: utf-8 -*-
"""
将女博士FAQ数据上传到LangChain-Chatchat知识库
"""

import requests
import pandas as pd
import logging
import time
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainKBUploader:
    """LangChain-Chatchat 知识库上传工具"""
    
    def __init__(
        self, 
        api_base: str = "http://localhost:7861",
        kb_name: str = "wuboshi_faq",
        embed_model: str = "bge-large-zh-v1.5"
    ):
        """
        初始化上传工具
        
        Args:
            api_base: LangChain-Chatchat API地址
            kb_name: 知识库名称
            embed_model: 嵌入模型名称
        """
        self.api_base = api_base
        self.kb_name = kb_name
        self.embed_model = embed_model
        
        logger.info(f"初始化完成 - API: {api_base}, KB: {kb_name}")
    
    def create_knowledge_base(self) -> bool:
        """创建知识库"""
        
        url = f"{self.api_base}/knowledge_base/create_knowledge_base"
        
        data = {
            "knowledge_base_name": self.kb_name,
            "vector_store_type": "faiss",
            "embed_model": self.embed_model
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    logger.info(f"知识库创建成功: {self.kb_name}")
                    return True
                elif "already exists" in str(result):
                    logger.info(f"知识库已存在: {self.kb_name}")
                    return True
                else:
                    logger.error(f"创建失败: {result}")
                    return False
            else:
                logger.error(f"请求失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"创建知识库异常: {e}")
            return False
    
    def upload_docs(self, docs: List[Dict]) -> bool:
        """
        批量上传文档
        
        Args:
            docs: 文档列表,每个文档包含:
                - text: 文档内容
                - metadata: 元数据(可选)
        """
        
        url = f"{self.api_base}/knowledge_base/upload_docs"
        
        # 构建文件数据
        files = []
        for idx, doc in enumerate(docs):
            # 创建临时文本文件
            content = doc['text']
            filename = f"doc_{idx}.txt"
            
            files.append(
                ('files', (filename, content, 'text/plain'))
            )
        
        # 构建表单数据
        data = {
            "knowledge_base_name": self.kb_name,
            "override": "false",
            "to_vector_store": "true",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "zh_title_enhance": "true"
        }
        
        try:
            response = requests.post(
                url, 
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"上传成功: {len(docs)} 个文档")
                return True
            else:
                logger.error(f"上传失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"上传异常: {e}")
            return False
    
    def upload_from_csv(self, csv_file: str, batch_size: int = 10):
        """
        从CSV文件批量上传
        
        Args:
            csv_file: CSV文件路径
            batch_size: 批次大小
        """
        
        logger.info(f"开始从CSV上传: {csv_file}")
        
        # 1. 读取CSV
        df = pd.read_csv(csv_file, encoding='utf-8')
        logger.info(f"读取到 {len(df)} 条数据")
        
        # 2. 创建知识库
        if not self.create_knowledge_base():
            logger.error("知识库创建失败,终止上传")
            return
        
        # 3. 构建文档
        docs = []
        for idx, row in df.iterrows():
            # 组合问题和答案作为文档内容
            doc_text = f"""问题: {row['query']}

答案: {row['answer']}
"""
            
            docs.append({
                "text": doc_text,
                "metadata": {
                    "source": "five_deal_answer_res.csv",
                    "query": row['query'],
                    "doc_id": f"doc{idx}"
                }
            })
        
        # 4. 分批上传
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(docs), batch_size):
            batch_docs = docs[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1
            
            logger.info(f"上传批次 {current_batch}/{total_batches} ({len(batch_docs)} 条)")
            
            success = self.upload_docs(batch_docs)
            
            if not success:
                logger.error(f"批次 {current_batch} 上传失败")
                break
            
            # 避免请求过快
            time.sleep(1)
        
        logger.info("上传完成!")
    
    def test_search(self, query: str, top_k: int = 5):
        """测试检索"""
        
        url = f"{self.api_base}/knowledge_base/search_docs"
        
        data = {
            "query": query,
            "knowledge_base_name": self.kb_name,
            "top_k": top_k
        }
        
        try:
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info(f"\n检索测试 - Query: {query}")
                logger.info("=" * 60)
                
                for idx, doc in enumerate(result['data'], 1):
                    logger.info(f"\n[{idx}] Score: {doc.get('score', 0):.3f}")
                    logger.info(f"Content: {doc['page_content'][:200]}...")
                
                return result['data']
            else:
                logger.error(f"检索失败: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"检索异常: {e}")
            return []


def main():
    """主函数"""
    
    print("=" * 60)
    print("LangChain-Chatchat 知识库上传工具")
    print("=" * 60)
    
    # 初始化上传器
    uploader = LangChainKBUploader(
        api_base="http://localhost:7861",
        kb_name="wuboshi_faq",
        embed_model="bge-large-zh-v1.5"
    )
    
    # 上传数据
    csv_file = "code/data/five_deal_answer_res.csv"
    uploader.upload_from_csv(csv_file, batch_size=20)
    
    # 测试检索
    print("\n" + "=" * 60)
    print("检索测试")
    print("=" * 60)
    
    test_queries = [
        "胶原蛋白怎么吃",
        "孕妇能喝吗",
        "喝多久能看到效果"
    ]
    
    for query in test_queries:
        uploader.test_search(query, top_k=3)
        time.sleep(1)


if __name__ == "__main__":
    main()

