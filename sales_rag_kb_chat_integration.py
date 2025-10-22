"""
Sales-RAG kb_chat.py 集成DeepRetrieval的修改示例
将此代码集成到 sales-rag/libs/chatchat-server/chatchat/server/chat/kb_chat.py 中
"""

# 在文件顶部添加导入
from chatchat.server.chat.deepretrieval_enhancer import get_query_enhancer
import uuid

# 在 kb_chat 函数中添加 session_id 参数
async def kb_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
                kb_name: str = Body("", description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称", examples=["samples"]),
                top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                score_threshold: float = Body(
                    Settings.kb_settings.SCORE_THRESHOLD,
                    description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                    ge=0,
                    le=2,
                ),
                history: List[History] = Body(
                    [],
                    description="历史对话",
                    examples=[[
                        {"role": "user",
                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
                        {"role": "assistant",
                        "content": "虎头虎脑"}]]
                ),
                stream: bool = Body(True, description="流式输出"),
                model: str = Body(get_default_llm(), description="LLM 模型名称。"),
                temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                max_tokens: Optional[int] = Body(
                    Settings.model_settings.MAX_TOKENS,
                    description="限制LLM生成Token数量，默认None代表模型最大值"
                ),
                prompt_name: str = Body(
                    "default",
                    description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                ),
                return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                request: Request = None,
                sql_enhance: bool = Body(False, description="SQL 增强"),
                table_names: List[str] = Body([],description="为SQL工具指定的关系表"),
                table_comments: dict = Body({},description="关系表的说明"),
                sql_top_k: int = Body(0, description="sql返回行数"),
                only_recall_flag: bool = Body(False, description="只返回Recall文档，无需answer"),
                # ========== 新增参数 ==========
                user_id: str = Body("", description="用户ID，用于个性化和反馈收集"),
                session_id: str = Body("", description="会话ID，用于RL训练反馈关联"),
                ):
    
    # 如果没有提供session_id，自动生成一个
    if not session_id:
        session_id = f"{user_id}_{int(time.time())}" if user_id else str(uuid.uuid4())
    
    if mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")

    # ========== 替换原有的 advanced_query 和 hyde 逻辑 ==========
    
    # 原来的代码:
    # advanced_query = Settings.kb_settings.ADVANCED_QUERY
    # if advanced_query:
    #     query = await llm_chat("advanced_query", "default", {"question": query})
    #     logger.info(f"Advanced Query: {query}")
    # 
    # hyde = Settings.kb_settings.HYDE  
    # if hyde:
    #     query = await llm_chat("hyde", "default", {"question": query})
    #     logger.info(f"HYDE for Query: {query}")

    # 新的集成逻辑:
    enhanced_query = query
    enhancement_metadata = {}
    
    # 集成DeepRetrieval查询增强 (替代原有ADVANCED_QUERY和HYDE)
    if (Settings.kb_settings.ADVANCED_QUERY or 
        getattr(Settings, 'DEEPRETRIEVAL_ENABLED', False)):
        
        try:
            enhancer = get_query_enhancer()
            enhanced_query, enhancement_metadata = await enhancer.enhance_query(
                query=query, 
                history=[h.dict() for h in history] if history else [], 
                user_id=user_id,
                session_id=session_id
            )
            
            logger.info(
                f"查询增强[{enhancement_metadata['method']}]: "
                f"'{query}' -> '{enhanced_query}' "
                f"({enhancement_metadata.get('latency_ms', 0):.1f}ms)"
            )
            
        except Exception as e:
            logger.error(f"查询增强失败: {e}")
            enhancement_metadata = {"method": "error", "error": str(e)}
    
    # 如果DeepRetrieval没有改进查询，回退到原有逻辑
    if enhanced_query == query and Settings.kb_settings.HYDE:
        enhanced_query = await llm_chat("hyde", "default", {"question": query})
        logger.info(f"HYDE for Query: {enhanced_query}")
        enhancement_metadata = {"method": "hyde_fallback"}
    
    # 使用增强后的查询替换原query
    query = enhanced_query

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal history, prompt_name, max_tokens

            history = [History.from_data(h) for h in history]

            if max_tokens in [None, 0]:
                max_tokens = Settings.model_settings.MAX_TOKENS
            if str(sql_enhance) == str(True):
                try:
                    sql_tool = get_tool('wrapped_text2sql')
                    sql_exec_failed = False
                    sql_input = {
                        'query':query,
                        'top_k':sql_top_k,
                        'table_names':table_names,
                        'table_comments': table_comments,
                    }
                    for result, error in sql_tool.run(sql_input):
                        if error is not None:
                            if 'Permission denied' in str(error):
                                sql_exec_failed = True
                                break
                            logger.error(error)
                except Exception as e:
                    logger.error(e)

            if mode == "local_kb":
                kb = KBServiceFactory.get_service_by_name(kb_name)
                if sql_enhance and not sql_exec_failed:
                    yield json.dumps({"docs": []}, ensure_ascii=False)
                    docs = []
                elif kb is not None:
                    if query:
                        docs = await run_in_threadpool(search_docs,
                                                       query=query,
                                                       knowledge_base_name=kb_name,
                                                       top_k=top_k,
                                                       score_threshold=score_threshold,
                                                       file_name="",
                                                       metadata={})

                        # ========== 新增: 记录检索结果用于RL训练 ==========
                        if (user_id and enhancement_metadata.get("method") == "deepretrieval"):
                            try:
                                enhancer = get_query_enhancer()
                                enhancer.feedback_collector.record_retrieval_results(
                                    session_id=session_id,
                                    results=[doc.dict() if hasattr(doc, 'dict') else doc for doc in docs] if docs else []
                                )
                            except Exception as e:
                                logger.warning(f"记录检索结果失败: {e}")

                        yield json.dumps({"docs": docs}, ensure_ascii=False)
                    else:
                        docs = []
                        yield json.dumps({"docs": docs}, ensure_ascii=False)
                    source_documents = format_reference(kb_name, docs, api_address(is_public=True))

            elif mode == "temp_kb":
                # ... 原有temp_kb逻辑保持不变 ...
                
            elif mode == "search_engine":
                # ... 原有search_engine逻辑保持不变 ...

            if return_direct:
                # ... 原有return_direct逻辑保持不变 ...
            else:
                # ... 原有LLM生成逻辑保持不变 ...
                
                # ========== 新增: 在生成完答案后记录对话指标 ==========
                if user_id and enhancement_metadata.get("method") == "deepretrieval":
                    try:
                        enhancer = get_query_enhancer()
                        # 计算对话轮数
                        turn_count = len(history) + 1
                        
                        enhancer.feedback_collector.record_conversation_metrics(
                            session_id=session_id,
                            turn_count=turn_count,
                            continued=True  # 生成了答案说明对话在继续
                        )
                    except Exception as e:
                        logger.warning(f"记录对话指标失败: {e}")

        except Exception as e:
            logger.error(f"Error in knowledge_base_chat: {e}")
            yield json.dumps({"error": str(e)}, ensure_ascii=False)

    # 返回流式响应
    return EventSourceResponse(knowledge_base_chat_iterator(), media_type="text/plain")


# ========== 新增: 用户反馈记录API ==========

async def record_user_feedback_api(
    session_id: str = Body(..., description="会话ID"),
    feedback: Dict = Body(..., description="用户反馈", examples=[{
        "satisfaction": 1,  # 1=满意, 0=一般, -1=不满意
        "explicit_feedback": "thumbs_up"  # "thumbs_up", "thumbs_down", "none"
    }])
):
    """
    记录用户反馈API，用于RL训练数据收集
    
    可以在前端添加点赞/点踩按钮，调用此API记录用户反馈
    """
    
    try:
        from chatchat.server.chat.deepretrieval_enhancer import record_user_feedback
        
        record_user_feedback(session_id, feedback)
        
        return BaseResponse(code=200, msg="反馈记录成功")
        
    except Exception as e:
        logger.error(f"记录用户反馈失败: {e}")
        return BaseResponse(code=500, msg=f"反馈记录失败: {e}")


# ========== 新增: 查询增强统计API ==========

async def get_query_enhancement_stats():
    """获取查询增强统计信息"""
    
    try:
        from chatchat.server.chat.deepretrieval_enhancer import get_query_enhancer
        
        enhancer = get_query_enhancer()
        stats = enhancer.get_enhancement_stats()
        
        return BaseResponse(code=200, data=stats)
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return BaseResponse(code=500, msg=f"获取统计信息失败: {e}")


"""
========== 配置文件修改说明 ==========

在 sales-rag/libs/chatchat-server/chatchat/basic_settings.yaml 中添加:

# DeepRetrieval配置
DEEPRETRIEVAL_ENABLED: false  # 是否启用DeepRetrieval (生产环境先设为false)
DEEPRETRIEVAL_API_URL: "http://localhost:8001/v1/chat/completions"  # DeepRetrieval服务地址
DEEPRETRIEVAL_FALLBACK: true  # 失败时是否降级到ADVANCED_QUERY
DEEPRETRIEVAL_TIMEOUT: 2.0    # 超时时间(秒)

# A/B测试配置
DEEPRETRIEVAL_AB_TEST: false  # 是否启用A/B测试
DEEPRETRIEVAL_AB_RATIO: 0.3   # A组比例 (使用DeepRetrieval的用户占比)

# RL训练配置
DEEPRETRIEVAL_RL_DATA_DIR: "data/rl_training_samples"  # RL训练数据存储目录
DEEPRETRIEVAL_FEEDBACK_ENABLED: true  # 是否收集用户反馈

========== API路由注册 ==========

在 sales-rag/libs/chatchat-server/chatchat/server/api_server/chat_routes.py 中添加:

# 用户反馈记录
app.post("/chat/feedback", 
         tags=["Chat"], 
         summary="记录用户反馈")(record_user_feedback_api)

# 查询增强统计
app.get("/chat/enhancement_stats", 
        tags=["Chat"], 
        summary="查询增强统计")(get_query_enhancement_stats)

========== 前端集成建议 ==========

在前端聊天界面添加反馈按钮:

```javascript
// 在每条回答后添加点赞/点踩按钮
function addFeedbackButtons(sessionId, messageElement) {
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'feedback-buttons';
    
    const thumbsUp = document.createElement('button');
    thumbsUp.innerHTML = '👍';
    thumbsUp.onclick = () => sendFeedback(sessionId, 1, 'thumbs_up');
    
    const thumbsDown = document.createElement('button'); 
    thumbsDown.innerHTML = '👎';
    thumbsDown.onclick = () => sendFeedback(sessionId, -1, 'thumbs_down');
    
    feedbackDiv.appendChild(thumbsUp);
    feedbackDiv.appendChild(thumbsDown);
    messageElement.appendChild(feedbackDiv);
}

function sendFeedback(sessionId, satisfaction, explicit_feedback) {
    fetch('/chat/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: sessionId,
            feedback: {
                satisfaction: satisfaction,
                explicit_feedback: explicit_feedback
            }
        })
    });
}
```

========== 使用流程 ==========

1. 启动DeepRetrieval服务:
   vllm serve <trained_model_path> --port 8001

2. 在配置文件中启用DeepRetrieval:
   DEEPRETRIEVAL_ENABLED: true

3. 启动sales-rag服务:
   python -m chatchat.startup

4. 开始A/B测试 (可选):
   DEEPRETRIEVAL_AB_TEST: true
   DEEPRETRIEVAL_AB_RATIO: 0.2  # 20%用户使用DeepRetrieval

5. 监控和训练:
   - 用户反馈自动收集
   - 定期运行RL训练: python sales_rag_deepretrieval_trainer.py --mode continuous
   - 监控效果: 访问 /chat/enhancement_stats

"""
