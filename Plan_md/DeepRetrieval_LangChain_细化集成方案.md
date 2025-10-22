# DeepRetrieval × LangChain-Chatchat 细化集成方案

> 基于LangChain-Chatchat实际代码结构的详细集成指南

## 📁 项目结构分析

### LangChain-Chatchat 核心架构

```
Langchain-Chatchat-master/
├── libs/chatchat-server/chatchat/          # 核心服务器代码
│   ├── server/
│   │   ├── chat/                            # 对话模块
│   │   │   ├── kb_chat.py                   # 知识库对话 ⭐ 主要集成点
│   │   │   ├── chat.py                      # 普通对话
│   │   │   ├── file_chat.py                 # 文件对话
│   │   │   └── utils.py                     # 工具函数
│   │   ├── knowledge_base/                  # 知识库模块
│   │   │   ├── kb_service/                  # 知识库服务
│   │   │   │   ├── base.py                  # 基础服务 ⭐ 集成点
│   │   │   │   ├── faiss_kb_service.py      # FAISS实现
│   │   │   │   ├── milvus_kb_service.py     # Milvus实现
│   │   │   │   └── ...
│   │   │   ├── kb_doc_api.py                # 文档检索API ⭐ 集成点
│   │   │   └── utils.py
│   │   ├── api_server/                      # API服务器
│   │   │   ├── chat_routes.py               # 对话路由 ⭐ 集成点
│   │   │   ├── kb_routes.py                 # 知识库路由
│   │   │   └── server_app.py                # FastAPI应用
│   │   └── file_rag/                        # RAG相关
│   │       ├── retrievers/                  # 检索器
│   │       └── text_splitter/               # 文本切分
│   ├── settings.py                          # 全局配置 ⭐ 配置点
│   └── startup.py                           # 启动入口
├── frontend/                                # Next.js前端
└── libs/python-sdk/                         # Python SDK
```

---

## 🎯 集成策略详解

### 策略1: 非侵入式集成 (推荐)

**核心思路**: 在不修改LangChain-Chatchat核心代码的前提下,通过插件/中间件方式集成

#### 1.1 创建Query重写中间件

```python
# Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/middleware/query_rewriter.py

from typing import Optional, Callable
import logging
from functools import wraps
import time

from openai import OpenAI
import re
import json

logger = logging.getLogger(__name__)


class QueryRewriterMiddleware:
    """查询重写中间件 - 无侵入式集成DeepRetrieval"""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8001/v1/chat/completions",
        enable: bool = True,
        timeout: float = 2.0,
        fallback_enabled: bool = True
    ):
        """
        初始化查询重写中间件
        
        Args:
            api_url: DeepRetrieval Query重写服务地址
            enable: 是否启用查询重写
            timeout: 请求超时时间
            fallback_enabled: 失败时是否启用降级策略
        """
        self.enable = enable
        self.timeout = timeout
        self.fallback_enabled = fallback_enabled
        
        if self.enable:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=api_url,
                timeout=timeout
            )
            logger.info(f"QueryRewriterMiddleware initialized - API: {api_url}")
        else:
            logger.info("QueryRewriterMiddleware disabled")
    
    def rewrite(self, query: str, context: str = "") -> dict:
        """
        重写查询
        
        Returns:
            {
                "original": 原始查询,
                "rewritten": 重写后查询,
                "method": "model" | "rule" | "none",
                "success": bool,
                "latency_ms": float
            }
        """
        if not self.enable:
            return {
                "original": query,
                "rewritten": query,
                "method": "none",
                "success": True,
                "latency_ms": 0
            }
        
        start_time = time.time()
        
        try:
            # 调用DeepRetrieval模型
            response = self.client.chat.completions.create(
                model="query-rewrite",
                messages=[{
                    "role": "user",
                    "content": self._build_prompt(query, context)
                }],
                max_tokens=512,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            rewritten = self._parse_response(content, query)
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "original": query,
                "rewritten": rewritten,
                "method": "model",
                "success": True,
                "latency_ms": latency
            }
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            
            # 降级策略
            if self.fallback_enabled:
                rewritten = self._rule_based_rewrite(query)
                method = "rule" if rewritten != query else "none"
            else:
                rewritten = query
                method = "none"
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "original": query,
                "rewritten": rewritten,
                "method": method,
                "success": False,
                "latency_ms": latency
            }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """构建提示词"""
        prompt = f"""你是一个专业的保健品知识库查询优化助手。

请优化以下查询,使其更适合检索:

原始查询: {query}
"""
        if context:
            prompt += f"对话上下文: {context}\n"
        
        prompt += """
输出格式:
<think>分析...</think>
<answer>{"query": "优化后的查询"}</answer>
"""
        return prompt
    
    def _parse_response(self, content: str, fallback: str) -> str:
        """解析响应"""
        try:
            match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
                return result.get("query", fallback)
        except:
            pass
        return fallback
    
    def _rule_based_rewrite(self, query: str) -> str:
        """基于规则的查询重写(降级方案)"""
        # 规则1: 产品使用方法
        if any(word in query for word in ["怎么用", "怎么吃", "如何服用"]):
            if "胶原蛋白" in query:
                return f"胶原蛋白肽 服用方法 用量"
            elif "虾青素" in query:
                return f"虾青素 服用方法"
        
        # 规则2: 禁忌咨询
        if any(word in query for word in ["孕妇", "备孕", "哺乳"]):
            return f"{query} 禁忌 注意事项"
        
        return query


# 全局单例
_query_rewriter = None


def get_query_rewriter() -> QueryRewriterMiddleware:
    """获取全局Query重写器"""
    global _query_rewriter
    if _query_rewriter is None:
        from chatchat.settings import Settings
        
        # 从配置读取参数
        config = getattr(Settings, 'query_rewrite_settings', {})
        
        _query_rewriter = QueryRewriterMiddleware(
            api_url=config.get('api_url', 'http://localhost:8001/v1/chat/completions'),
            enable=config.get('enable', False),  # 默认关闭,需手动开启
            timeout=config.get('timeout', 2.0),
            fallback_enabled=config.get('fallback_enabled', True)
        )
    
    return _query_rewriter


def with_query_rewrite(func: Callable):
    """
    装饰器: 自动为函数添加查询重写功能
    
    使用方法:
        @with_query_rewrite
        def search_docs(query, ...):
            # query会被自动重写
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 提取query参数
        query = kwargs.get('query', None)
        
        if query:
            rewriter = get_query_rewriter()
            result = rewriter.rewrite(query)
            
            # 记录重写信息
            logger.info(
                f"Query Rewrite [{result['method']}] "
                f"({result['latency_ms']:.1f}ms): "
                f"'{result['original']}' -> '{result['rewritten']}'"
            )
            
            # 替换为重写后的query
            kwargs['query'] = result['rewritten']
            
            # 保存原始query到metadata (可选)
            if 'metadata' not in kwargs:
                kwargs['metadata'] = {}
            kwargs['metadata']['_original_query'] = result['original']
            kwargs['metadata']['_rewrite_method'] = result['method']
        
        # 调用原函数
        return await func(*args, **kwargs)
    
    return wrapper
```

#### 1.2 修改配置文件

```yaml
# Langchain-Chatchat-master/libs/chatchat-server/data/query_rewrite_settings.yaml

# Query重写配置
query_rewrite_settings:
  enable: true  # 是否启用查询重写
  api_url: "http://localhost:8001/v1/chat/completions"  # DeepRetrieval服务地址
  timeout: 2.0  # 超时时间(秒)
  fallback_enabled: true  # 是否启用降级策略
  
  # A/B测试配置(可选)
  ab_test:
    enabled: false  # 是否启用A/B测试
    group_ratio: 0.5  # A组比例 (0-1)
    
  # 缓存配置
  cache:
    enabled: true
    max_size: 1000
    ttl: 3600  # 缓存有效期(秒)
```

#### 1.3 在settings.py中加载配置

```python
# 在 Langchain-Chatchat-master/libs/chatchat-server/chatchat/settings.py 中添加

class QueryRewriteSettings(BaseFileSettings):
    """查询重写配置"""
    
    model_config = SettingsConfigDict(
        yaml_file=CHATCHAT_ROOT / "data/query_rewrite_settings.yaml"
    )
    
    enable: bool = False
    api_url: str = "http://localhost:8001/v1/chat/completions"
    timeout: float = 2.0
    fallback_enabled: bool = True
    
    # A/B测试
    ab_test: Dict = {
        "enabled": False,
        "group_ratio": 0.5
    }
    
    # 缓存
    cache: Dict = {
        "enabled": True,
        "max_size": 1000,
        "ttl": 3600
    }


class Settings:
    # ... 其他配置 ...
    
    query_rewrite_settings: QueryRewriteSettings = QueryRewriteSettings()
```

#### 1.4 集成到kb_doc_api.py (方法1: 装饰器)

```python
# 修改 Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/knowledge_base/kb_doc_api.py

from chatchat.server.middleware.query_rewriter import with_query_rewrite


# 使用装饰器方式集成
@with_query_rewrite  # 只需添加这一行!
def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称"),
        top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K),
        score_threshold: float = Body(Settings.kb_settings.SCORE_THRESHOLD),
        file_name: str = Body(""),
        metadata: dict = Body({}),
) -> List[Dict]:
    # 原有代码保持不变
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k, score_threshold)
            data = [DocumentWithVSId(**{"id": x.metadata.get("id"), **x.dict()}) for x in docs]
    return [x.dict() for x in data]
```

#### 1.5 集成到kb_chat.py (方法2: 显式调用)

```python
# 修改 Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/chat/kb_chat.py

from chatchat.server.middleware.query_rewriter import get_query_rewriter


async def kb_chat(query: str = Body(...), ...):
    # ... 原有代码 ...
    
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal query  # 重要: 声明为nonlocal以便修改
            
            # ===== 添加Query重写逻辑 =====
            rewriter = get_query_rewriter()
            rewrite_result = rewriter.rewrite(query)
            
            original_query = query
            optimized_query = rewrite_result['rewritten']
            
            logger.info(
                f"Query重写 [{rewrite_result['method']}]: "
                f"'{original_query}' -> '{optimized_query}'"
            )
            # ===== Query重写结束 =====
            
            if mode == "local_kb":
                kb = KBServiceFactory.get_service_by_name(kb_name)
                
                # 使用重写后的query进行检索
                docs = await run_in_threadpool(
                    search_docs,
                    query=optimized_query,  # 使用重写后的query
                    knowledge_base_name=kb_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    file_name="",
                    metadata={}
                )
                
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            
            # ... 后续代码保持不变 ...
```

---

### 策略2: 深度集成 (高级)

#### 2.1 扩展KBService基类

```python
# 创建新文件: Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/knowledge_base/kb_service/deepretrieval_kb_service.py

from typing import List, Tuple
from langchain.docstore.document import Document

from chatchat.server.knowledge_base.kb_service.base import KBService
from chatchat.server.middleware.query_rewriter import get_query_rewriter


class DeepRetrievalKBService(KBService):
    """
    集成DeepRetrieval查询重写的知识库服务
    
    继承任意向量库实现(如FAISS),并在检索前自动进行查询重写
    """
    
    # 选择基础实现(可以是FAISS/Milvus/Chroma等)
    _base_service_class = None  # 运行时动态设置
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_rewriter = get_query_rewriter()
    
    def search_docs(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        重写的search_docs方法: 在检索前自动优化查询
        """
        # 1. 查询重写
        rewrite_result = self.query_rewriter.rewrite(query)
        optimized_query = rewrite_result['rewritten']
        
        logger.info(
            f"[DeepRetrieval] Query重写: "
            f"'{query}' -> '{optimized_query}'"
        )
        
        # 2. 使用基类的检索方法
        docs = super().search_docs(optimized_query, top_k, score_threshold)
        
        # 3. (可选) 使用原始query进行重排序
        # docs = self._rerank_with_original_query(query, docs)
        
        return docs
    
    def _rerank_with_original_query(
        self,
        original_query: str,
        docs: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        使用原始query对结果进行重排序
        
        策略: 
        1. 用重写query召回候选文档(cast wide net)
        2. 用原始query重排序(precision)
        """
        # TODO: 实现重排序逻辑
        # 可以使用cross-encoder或其他reranker
        return docs


# 工厂方法: 动态创建DeepRetrieval版本的KB Service
def create_deepretrieval_kb_service(base_service_class):
    """
    为任意KBService创建DeepRetrieval增强版本
    
    使用示例:
        from chatchat.server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
        
        DeepRetrievalFaissKB = create_deepretrieval_kb_service(FaissKBService)
        kb = DeepRetrievalFaissKB(kb_name="my_kb")
    """
    class DeepRetrievalEnhancedKB(base_service_class, DeepRetrievalKBService):
        pass
    
    return DeepRetrievalEnhancedKB
```

#### 2.2 在KBServiceFactory中注册

```python
# 修改 Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/knowledge_base/kb_service/base.py

from chatchat.server.knowledge_base.kb_service.deepretrieval_kb_service import (
    create_deepretrieval_kb_service
)


class KBServiceFactory:
    # ... 原有代码 ...
    
    @staticmethod
    def get_service(
        kb_name: str,
        vector_store_type: str = None,
        embed_model: str = None,
        use_deepretrieval: bool = None  # 新增参数
    ) -> KBService:
        """
        获取知识库服务实例
        
        Args:
            use_deepretrieval: 是否使用DeepRetrieval增强
                - None: 从配置读取
                - True/False: 强制启用/禁用
        """
        from chatchat.settings import Settings
        
        # 确定是否使用DeepRetrieval
        if use_deepretrieval is None:
            use_deepretrieval = Settings.query_rewrite_settings.enable
        
        # 获取基础service class
        if vector_store_type == SupportedVSType.FAISS:
            from .faiss_kb_service import FaissKBService
            base_class = FaissKBService
        elif vector_store_type == SupportedVSType.MILVUS:
            from .milvus_kb_service import MilvusKBService
            base_class = MilvusKBService
        # ... 其他类型 ...
        
        # 如果启用DeepRetrieval,创建增强版本
        if use_deepretrieval:
            service_class = create_deepretrieval_kb_service(base_class)
        else:
            service_class = base_class
        
        return service_class(
            knowledge_base_name=kb_name,
            embed_model=embed_model or get_default_embedding()
        )
```

---

### 策略3: 混合检索增强

```python
# 创建新文件: Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/knowledge_base/hybrid_retriever.py

from typing import List, Dict, Tuple
from langchain.docstore.document import Document

from chatchat.server.middleware.query_rewriter import get_query_rewriter


class HybridRetriever:
    """
    混合检索器: 结合原始query和重写query的结果
    
    检索策略:
    1. 用原始query检索 (保留用户意图)
    2. 用重写query检索 (扩展召回)
    3. 融合两路结果
    """
    
    def __init__(self, kb_service):
        self.kb_service = kb_service
        self.query_rewriter = get_query_rewriter()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
        strategy: str = "rrf"  # rrf | weighted | cascade
    ) -> List[Tuple[Document, float]]:
        """
        混合检索
        
        Args:
            strategy: 融合策略
                - rrf: Reciprocal Rank Fusion
                - weighted: 加权融合
                - cascade: 级联(先重写,不满意再原始)
        """
        if strategy == "rrf":
            return self._rrf_search(query, top_k, score_threshold)
        elif strategy == "weighted":
            return self._weighted_search(query, top_k, score_threshold)
        elif strategy == "cascade":
            return self._cascade_search(query, top_k, score_threshold)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _rrf_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion策略
        
        公式: RRF_score(d) = Σ 1/(k + rank_i(d))
        其中k是常数(通常60),rank_i是文档在第i个结果列表中的排名
        """
        # 1. 原始query检索
        docs_original = self.kb_service.search_docs(
            query, top_k * 2, score_threshold
        )
        
        # 2. 重写query检索
        rewrite_result = self.query_rewriter.rewrite(query)
        docs_rewritten = self.kb_service.search_docs(
            rewrite_result['rewritten'], top_k * 2, score_threshold
        )
        
        # 3. RRF融合
        k = 60  # RRF常数
        doc_scores = {}
        
        # 处理原始query结果
        for rank, (doc, score) in enumerate(docs_original):
            doc_id = doc.metadata.get('id', id(doc))
            rrf_score = 1 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        
        # 处理重写query结果
        for rank, (doc, score) in enumerate(docs_rewritten):
            doc_id = doc.metadata.get('id', id(doc))
            rrf_score = 1 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        
        # 4. 合并并排序
        all_docs = {id(doc): doc for doc, _ in docs_original + docs_rewritten}
        
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [(all_docs[doc_id], score) for doc_id, score in sorted_docs]
    
    def _weighted_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        original_weight: float = 0.3,
        rewritten_weight: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """
        加权融合策略
        
        根据重写质量动态调整权重
        """
        # 1. 双路检索
        docs_original = self.kb_service.search_docs(query, top_k * 2, score_threshold)
        
        rewrite_result = self.query_rewriter.rewrite(query)
        docs_rewritten = self.kb_service.search_docs(
            rewrite_result['rewritten'], top_k * 2, score_threshold
        )
        
        # 2. 动态调整权重(根据重写方法)
        if rewrite_result['method'] == 'model':
            # 模型重写,给更高权重
            w_orig, w_rewr = 0.2, 0.8
        elif rewrite_result['method'] == 'rule':
            # 规则重写,权重平衡
            w_orig, w_rewr = 0.4, 0.6
        else:
            # 未重写,只用原始query
            w_orig, w_rewr = 1.0, 0.0
        
        # 3. 加权融合
        doc_scores = {}
        
        for doc, score in docs_original:
            doc_id = doc.metadata.get('id', id(doc))
            doc_scores[doc_id] = w_orig * score
        
        for doc, score in docs_rewritten:
            doc_id = doc.metadata.get('id', id(doc))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + w_rewr * score
        
        # 4. 排序返回
        all_docs = {id(doc): doc for doc, _ in docs_original + docs_rewritten}
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(all_docs[doc_id], score) for doc_id, score in sorted_docs]
    
    def _cascade_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        min_score: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """
        级联策略: 先尝试重写query,如果结果不好再用原始query
        """
        # 1. 先用重写query
        rewrite_result = self.query_rewriter.rewrite(query)
        docs = self.kb_service.search_docs(
            rewrite_result['rewritten'], top_k, score_threshold
        )
        
        # 2. 检查结果质量
        if docs and docs[0][1] >= min_score:
            # 结果满意,直接返回
            return docs
        else:
            # 结果不理想,用原始query
            logger.info(f"Cascade fallback to original query")
            return self.kb_service.search_docs(query, top_k, score_threshold)
```

---

## 🚀 完整集成步骤(推荐流程)

### 第1步: 创建中间件目录和文件

```bash
cd Langchain-Chatchat-master/libs/chatchat-server/chatchat/server
mkdir -p middleware
touch middleware/__init__.py
touch middleware/query_rewriter.py
```

### 第2步: 复制查询重写器代码

将上述`query_rewriter.py`的完整代码复制到:
`Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/middleware/query_rewriter.py`

### 第3步: 创建配置文件

```bash
cd Langchain-Chatchat-master/libs/chatchat-server
mkdir -p data
cat > data/query_rewrite_settings.yaml << EOF
query_rewrite_settings:
  enable: true
  api_url: "http://localhost:8001/v1/chat/completions"
  timeout: 2.0
  fallback_enabled: true
  
  ab_test:
    enabled: false
    group_ratio: 0.5
    
  cache:
    enabled: true
    max_size: 1000
    ttl: 3600
EOF
```

### 第4步: 修改settings.py

在`chatchat/settings.py`末尾添加:

```python
class QueryRewriteSettings(BaseFileSettings):
    model_config = SettingsConfigDict(
        yaml_file=CHATCHAT_ROOT / "data/query_rewrite_settings.yaml"
    )
    
    enable: bool = False
    api_url: str = "http://localhost:8001/v1/chat/completions"
    timeout: float = 2.0
    fallback_enabled: bool = True
    ab_test: Dict = {"enabled": False, "group_ratio": 0.5}
    cache: Dict = {"enabled": True, "max_size": 1000, "ttl": 3600}

# 在Settings类中添加
class Settings:
    # ...existing code...
    query_rewrite_settings: QueryRewriteSettings = QueryRewriteSettings()
```

### 第5步: 集成到kb_chat.py

在`chatchat/server/chat/kb_chat.py`中添加查询重写逻辑:

```python
# 在文件顶部导入
from chatchat.server.middleware.query_rewriter import get_query_rewriter

# 在kb_chat函数内部的knowledge_base_chat_iterator函数中添加
async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
    try:
        nonlocal query  # 添加这一行
        
        # ===== 添加Query重写 =====
        rewriter = get_query_rewriter()
        rewrite_result = rewriter.rewrite(query)
        original_query = query
        query = rewrite_result['rewritten']  # 更新query
        
        logger.info(f"Query重写[{rewrite_result['method']}]: '{original_query}' -> '{query}'")
        # ===== 重写结束 =====
        
        # 后续代码保持不变...
```

### 第6步: 启动DeepRetrieval服务

```bash
# 启动vLLM服务
vllm serve /path/to/trained/model \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.7
```

### 第7步: 启动LangChain-Chatchat

```bash
cd Langchain-Chatchat-master/libs/chatchat-server
python -m chatchat.startup
```

### 第8步: 测试集成效果

```python
import requests

# 测试查询重写效果
response = requests.post(
    "http://localhost:7861/chat/kb_chat",
    json={
        "query": "胶原蛋白怎么吃",
        "kb_name": "wuboshi_faq",
        "stream": False
    }
)

print(response.json())
```

---

## 📊 集成效果监控

### 创建监控面板

```python
# Langchain-Chatchat-master/libs/chatchat-server/chatchat/server/monitor/query_rewrite_monitor.py

from typing import Dict, List
import json
from collections import defaultdict
from datetime import datetime, timedelta


class QueryRewriteMonitor:
    """查询重写监控"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "failed": 0,
            "model_count": 0,
            "rule_count": 0,
            "none_count": 0,
            "total_latency": 0,
            "queries": []
        })
    
    def record(self, rewrite_result: Dict):
        """记录重写结果"""
        date_key = datetime.now().strftime("%Y-%m-%d")
        stats = self.stats[date_key]
        
        stats["total"] += 1
        if rewrite_result["success"]:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        stats[f"{rewrite_result['method']}_count"] += 1
        stats["total_latency"] += rewrite_result["latency_ms"]
        
        # 保存查询示例(最近100条)
        if len(stats["queries"]) < 100:
            stats["queries"].append({
                "original": rewrite_result["original"],
                "rewritten": rewrite_result["rewritten"],
                "method": rewrite_result["method"],
                "timestamp": datetime.now().isoformat()
            })
    
    def get_daily_report(self, date: str = None) -> Dict:
        """获取日报"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        stats = self.stats[date]
        
        if stats["total"] == 0:
            return {"message": "No data for this date"}
        
        return {
            "date": date,
            "total_queries": stats["total"],
            "success_rate": stats["success"] / stats["total"],
            "method_distribution": {
                "model": stats["model_count"],
                "rule": stats["rule_count"],
                "none": stats["none_count"]
            },
            "avg_latency_ms": stats["total_latency"] / stats["total"],
            "sample_queries": stats["queries"][:10]
        }


# 全局监控实例
monitor = QueryRewriteMonitor()
```

### API端点

```python
# 在 chatchat/server/api_server/server_routes.py 中添加

from chatchat.server.monitor.query_rewrite_monitor import monitor

@server_router.get("/monitor/query_rewrite", summary="查询重写监控")
async def get_query_rewrite_stats(date: str = None):
    return monitor.get_daily_report(date)
```

---

## 🎯 高级特性

### 1. A/B测试框架

```python
# chatchat/server/middleware/ab_testing.py

import hashlib


class ABTester:
    """A/B测试管理器"""
    
    def __init__(self, group_ratio: float = 0.5):
        self.group_ratio = group_ratio
    
    def get_group(self, user_id: str) -> str:
        """
        根据user_id分组
        
        Returns:
            "A" or "B"
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "A" if (hash_value % 100) / 100 < self.group_ratio else "B"
    
    def should_use_rewrite(self, user_id: str) -> bool:
        """判断是否使用查询重写"""
        return self.get_group(user_id) == "A"
```

使用示例:

```python
from chatchat.server.middleware.ab_testing import ABTester

ab_tester = ABTester(group_ratio=0.5)

# 在kb_chat中
async def kb_chat(query, ..., user_id: str = None):
    if user_id and ab_tester.should_use_rewrite(user_id):
        # A组: 使用查询重写
        query = rewriter.rewrite(query)["rewritten"]
    # B组: 不重写
    
    # 继续检索...
```

### 2. 自适应阈值

```python
class AdaptiveRewriter:
    """自适应查询重写器"""
    
    def __init__(self):
        self.performance_history = []
    
    def should_rewrite(self, query: str, context: Dict) -> bool:
        """
        根据历史性能动态决定是否重写
        
        考虑因素:
        - 查询长度
        - 查询复杂度
        - 历史重写效果
        - 用户反馈
        """
        # 简单查询不重写
        if len(query) < 5:
            return False
        
        # 计算历史平均提升
        if len(self.performance_history) > 10:
            avg_improvement = sum(self.performance_history[-10:]) / 10
            if avg_improvement < 0.05:  # 提升小于5%
                return False
        
        return True
    
    def record_performance(self, improvement: float):
        """记录性能提升"""
        self.performance_history.append(improvement)
```

---

## 📋 完整代码清单

### 需要创建的新文件

1. `chatchat/server/middleware/__init__.py`
2. `chatchat/server/middleware/query_rewriter.py` ⭐核心
3. `chatchat/server/middleware/ab_testing.py`
4. `chatchat/server/monitor/__init__.py`
5. `chatchat/server/monitor/query_rewrite_monitor.py`
6. `chatchat/server/knowledge_base/kb_service/deepretrieval_kb_service.py`
7. `chatchat/server/knowledge_base/hybrid_retriever.py`
8. `data/query_rewrite_settings.yaml`

### 需要修改的现有文件

1. `chatchat/settings.py` - 添加配置类
2. `chatchat/server/chat/kb_chat.py` - 集成查询重写
3. `chatchat/server/knowledge_base/kb_doc_api.py` - (可选)添加装饰器
4. `chatchat/server/knowledge_base/kb_service/base.py` - (可选)扩展工厂
5. `chatchat/server/api_server/server_routes.py` - (可选)添加监控API

---

## 🔧 调试建议

### 1. 启用详细日志

```python
# 在启动时设置
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 查看查询重写日志
logger = logging.getLogger("chatchat.server.middleware.query_rewriter")
logger.setLevel(logging.DEBUG)
```

### 2. 查看实时日志

```bash
# 在LangChain-Chatchat日志中查看
tail -f data/logs/api.log | grep "Query重写"
```

### 3. 测试单个组件

```python
# 测试查询重写器
from chatchat.server.middleware.query_rewriter import QueryRewriterMiddleware

rewriter = QueryRewriterMiddleware(
    api_url="http://localhost:8001/v1/chat/completions",
    enable=True
)

result = rewriter.rewrite("胶原蛋白怎么吃")
print(result)
```

---

## ✅ 验收测试

```python
# test_integration.py

import requests
import time


def test_query_rewrite_integration():
    """测试查询重写集成"""
    
    # 测试用例
    test_cases = [
        "胶原蛋白怎么吃",
        "孕妇能喝吗",
        "喝多久能看到效果"
    ]
    
    for query in test_cases:
        print(f"\n测试Query: {query}")
        
        # 发送请求
        response = requests.post(
            "http://localhost:7861/chat/kb_chat",
            json={
                "query": query,
                "kb_name": "wuboshi_faq",
                "stream": False,
                "top_k": 5
            }
        )
        
        result = response.json()
        
        # 验证响应
        assert response.status_code == 200
        assert "answer" in result
        
        print(f"✓ 响应成功")
        print(f"回答: {result['answer'][:100]}...")


if __name__ == "__main__":
    test_query_rewrite_integration()
    print("\n✅ 所有测试通过!")
```

---

## 🎓 总结

本细化方案提供了三种集成策略:

1. **非侵入式集成**(推荐⭐): 通过中间件/装饰器方式,最小化对原代码的修改
2. **深度集成**: 扩展KBService基类,提供更灵活的控制
3. **混合检索**: 结合多种检索策略,最大化召回效果

**推荐实施路径**:
1. 从策略1开始(快速验证效果)
2. 如果效果好,逐步采用策略3的混合检索
3. 最后可选策略2进行深度定制

关键优势:
- ✅ 模块化设计,易于维护
- ✅ 可配置开关,灵活控制
- ✅ 降级机制,保证稳定性
- ✅ A/B测试支持
- ✅ 完整的监控体系

立即开始第1步,创建中间件并测试效果!

