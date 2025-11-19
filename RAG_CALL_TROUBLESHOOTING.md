# RAG调用故障排查指南

## 🔴 当前问题

训练日志显示：
```
HTTP request exception
request: {...}, response: None
```

这说明RAG服务调用失败，返回了None。

## 🔍 可能的原因

### 1. RAG服务未启动
**检查方法**：
```bash
# 检查RAG服务是否运行
curl http://127.0.0.1:7861/health

# 或者检查进程
ps aux | grep rag
```

**解决方案**：
```bash
# 启动RAG服务
bash vllm_host.sh
```

### 2. 端口冲突或配置错误
**检查方法**：
```bash
# 检查端口是否被占用
netstat -tulpn | grep 7861

# 检查配置文件中的RAG_URL
grep RAG_URL src/config/basic_settings.yaml
```

**预期配置**：
```yaml
RAG_URL: "http://127.0.0.1:7861"
```

### 3. RAG服务返回非200状态码
**可能原因**：
- 请求参数错误
- 服务内部错误
- 超时

**检查日志**：
```bash
# 查看RAG服务日志
tail -f /path/to/rag/service.log
```

### 4. 网络连接问题
**检查方法**：
```bash
# 测试连接
telnet 127.0.0.1 7861

# 或使用curl测试
curl -X POST http://127.0.0.1:7861/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "chengla",
    "contact_id": "Customer_knowledge_17",
    "account_id": "Sale_knowledge_17",
    "kb_name": "chengla",
    "thought_unit": "",
    "score_threshold": 0.9,
    "top_k": 3,
    "context": "测试对话"
  }'
```

### 5. 请求参数问题
**检查清单**：
- [ ] `contact_id` 是 `Customer_knowledge_17`
- [ ] `account_id` 是 `Sale_knowledge_17`
- [ ] `top_k` 参数存在且为整数
- [ ] `context` 不为空
- [ ] `score_threshold` 在合理范围（0-1）

## 🔧 调试步骤

### 步骤1：确认RAG服务状态
```bash
# 检查服务是否运行
curl http://127.0.0.1:7861/health

# 预期返回
{"status": "ok"}
```

### 步骤2：手动测试RAG调用
```bash
# 使用训练日志中的实际参数测试
curl -X POST http://127.0.0.1:7861/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "chengla",
    "contact_id": "Customer_knowledge_17",
    "account_id": "Sale_knowledge_17",
    "kb_name": "chengla",
    "thought_unit": "",
    "score_threshold": 0.9,
    "top_k": 3,
    "context": "\n[客户][2025-10-09 18:01:41]: 我已经添加了你，现在我们可以开始聊天了。\n[销售][2025-10-09 18:01:47]:  同学你好❤"
  }'
```

### 步骤3：检查训练日志
查看更详细的错误信息：
```bash
# 查看完整日志
tail -f training.log | grep -A 10 "HTTP request exception"
```

现在日志会显示：
- `[HTTP] Sending POST to {url}` - 实际请求的URL
- `[HTTP] Response status: {code}` - HTTP状态码
- `[HTTP] Exception type: {type}, message: {msg}` - 异常类型和消息

### 步骤4：使用测试脚本
```bash
# 运行RAG调用测试
python scripts/test_rag_call.py
```

## 📊 常见错误和解决方案

### 错误1：Connection refused
**原因**：RAG服务未启动
**解决**：
```bash
bash vllm_host.sh
```

### 错误2：404 Not Found
**原因**：端点路径错误
**检查**：
- 32B应该调用 `/rag/chat`
- 8B应该调用 `/rag/chat_8b`

### 错误3：500 Internal Server Error
**原因**：RAG服务内部错误
**解决**：
1. 检查RAG服务日志
2. 验证参数格式
3. 重启RAG服务

### 错误4：Timeout
**原因**：请求超时
**解决**：
1. 检查RAG服务负载
2. 增加超时时间
3. 检查网络连接

## 🎯 快速修复

### 如果RAG服务未启动
```bash
# 启动RAG服务
cd /path/to/rag/service
bash vllm_host.sh

# 等待服务启动（通常需要1-2分钟）
sleep 60

# 验证服务
curl http://127.0.0.1:7861/health
```

### 如果参数错误
检查 `src/pipeline.py` 中的配置：
```python
rag = RagChater(
    tenant_id="chengla",
    contact_id="Customer_knowledge_17",  # 确认正确
    account_id="Sale_knowledge_17",      # 确认正确
    message_id="chengla_query_rl_message_id"
)
```

### 如果端口冲突
```bash
# 查找占用端口的进程
lsof -i :7861

# 杀死进程
kill -9 <PID>

# 重新启动RAG服务
bash vllm_host.sh
```

## 📝 日志分析

### 正常的日志应该是：
```
[HTTP] Sending POST to http://127.0.0.1:7861/rag/chat
[HTTP] Response status: 200
request: {...}, response: [{"content": "...", "score": 0.95}]
get rag chat results success
```

### 异常的日志：
```
[HTTP] Sending POST to http://127.0.0.1:7861/rag/chat
[HTTP] Exception type: ConnectError, message: Connection refused
HTTP request exception
request: {...}, response: None
get rag chat results failed
```

## 🚀 下一步

1. **确认RAG服务运行**
   ```bash
   curl http://127.0.0.1:7861/health
   ```

2. **手动测试调用**
   ```bash
   python scripts/test_rag_call.py
   ```

3. **查看详细日志**
   ```bash
   tail -f training.log | grep -E "\[HTTP\]|RAG"
   ```

4. **重新启动训练**
   ```bash
   cd verl_code
   bash your_training_script.sh
   ```

## 📞 需要检查的配置文件

1. `src/config/basic_settings.yaml` - RAG_URL配置
2. `src/pipeline.py` - RAG调用参数
3. `src/core/rag_chater.py` - RAG客户端实现
4. `vllm_host.sh` - RAG服务启动脚本

---

**提示**：现在HTTP工具已添加详细日志，重新运行训练会看到更多诊断信息！
