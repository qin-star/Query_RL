# DeepRetrieval × LangChain-Chatchat 快速开始指南

## 📋 概述

本指南帮助你快速完成DeepRetrieval与LangChain-Chatchat的集成,实现基于强化学习的智能查询重写功能。

**核心价值**:
- ✅ 通过强化学习自动优化查询质量
- ✅ 提升检索召回率和准确性
- ✅ 减少多轮澄清对话
- ✅ 提高首次回答准确率

---

## 🎯 数据集分析

### 训练数据集 (five_deal_answer_res.csv)

**数据量**: 359条  
**领域**: 女博士保健品FAQ  
**结构**:
```
- query: 用户问题
- res_queries: 改写后的查询
- answer: 标准答案
```

**数据分布**:
- 胶原蛋白相关: ~45%
- 备孕/孕期相关: ~15%
- 服用方法: ~25%
- 效果咨询: ~15%

### 测试数据集 (女博士-日常跟进数据集.xlsx)

**数据量**: 155条  
**类型**: 真实客服对话记录  
**特点**: 包含多轮对话上下文

---

## 🚀 快速开始(5步部署)

### 步骤1: 环境准备 (10分钟)

```bash
# 1. 创建Python环境
conda create -n deepretrieval python=3.9 -y
conda activate deepretrieval

# 2. 安装DeepRetrieval依赖
cd code
pip install -r requirements.txt
pip install -e .

# 3. 安装额外依赖
pip install vllm==0.6.3 pandas openpyxl
```

### 步骤2: 数据准备 (5分钟)

```bash
# 转换数据为DeepRetrieval训练格式
python prepare_training_data.py
```

**输出**: 
- `data/wuboshi_faq/processed/train.jsonl` (287条)
- `data/wuboshi_faq/processed/dev.jsonl` (72条)
- `data/wuboshi_faq/processed/corpus.jsonl` (359条)

### 步骤3: 训练Query重写模型 (1-2小时)

```bash
# 启动训练(需要GPU)
cd code
sh scripts/train/wuboshi_faq.sh

# 或手动运行
python -m verl.trainer.main_ppo \
    --config verl/trainer/config/wuboshi_config.yaml \
    --output_dir ../outputs/wuboshi_$(date +%Y%m%d)
```

**训练参数**:
- 基座模型: deepseek-llm-7b-chat
- 训练轮数: 5 epochs
- 批次大小: 4
- 学习率: 5e-7
- 预计时间: 1-2小时 (单卡RTX 3090)

### 步骤4: 部署模型 (5分钟)

```bash
# 启动vLLM服务
vllm serve outputs/wuboshi_YYYYMMDD/checkpoint-final \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048
```

**验证部署**:
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "query-rewrite",
    "messages": [{"role": "user", "content": "胶原蛋白怎么吃"}],
    "max_tokens": 256
  }'
```

### 步骤5: 集成到LangChain-Chatchat (15分钟)

#### 5.1 上传知识库

```bash
# 启动LangChain-Chatchat
cd /path/to/Langchain-Chatchat
python startup.py -a

# 在另一个终端上传数据
cd /path/to/DeepRetrieval-main
python upload_to_langchain.py
```

#### 5.2 集成查询重写模块

修改 `Langchain-Chatchat/server/knowledge_base/kb_service/base.py`:

```python
# 在文件开头添加
from langchain_query_rewriter import QueryRewriter

# 在 KBService.__init__ 中添加
self.query_rewriter = QueryRewriter(
    api_url="http://localhost:8001/v1/chat/completions"
)

# 在 search_docs 方法中添加
def search_docs(self, query: str, top_k: int = 10, **kwargs):
    # 查询重写
    rewritten_query = self.query_rewriter.rewrite(
        query=query,
        context=kwargs.get("history", "")
    )
    
    logger.info(f"Query重写: {query} -> {rewritten_query}")
    
    # 使用重写后的query检索
    docs = self.do_search(
        query=rewritten_query,
        top_k=top_k
    )
    
    return docs
```

#### 5.3 复制查询重写器

```bash
# 将查询重写器复制到LangChain-Chatchat项目
cp langchain_query_rewriter.py /path/to/Langchain-Chatchat/server/
```

#### 5.4 重启服务

```bash
# 重启LangChain-Chatchat
cd /path/to/Langchain-Chatchat
pkill -f "python startup.py"
python startup.py -a
```

---

## 🧪 测试集成效果

```bash
# 运行集成测试
python test_integration.py
```

**测试内容**:
1. 查询重写功能测试
2. 知识库检索测试
3. 完整对话测试
4. A/B对比测试

**预期效果**:
- Recall@10: 65% → 85% (提升20%)
- MRR: 0.6 → 0.8 (提升33%)
- 用户满意度: 提升20%

---

## 📁 项目文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `DeepRetrieval_LangChain_Integration_Plan.md` | 完整设计方案文档 |
| `prepare_training_data.py` | 数据准备脚本 |
| `langchain_query_rewriter.py` | 查询重写模块 |
| `upload_to_langchain.py` | 知识库上传工具 |
| `test_integration.py` | 集成测试脚本 |

### 数据文件

```
code/data/
├── five_deal_answer_res.csv          # 原始训练数据
└── 女博士-日常跟进数据集.xlsx        # 真实对话数据

data/wuboshi_faq/processed/
├── train.jsonl                       # 训练集
├── dev.jsonl                         # 验证集
├── corpus.jsonl                      # 文档库
└── stats.json                        # 数据统计
```

### 模型输出

```
outputs/wuboshi_YYYYMMDD_HHMMSS/
├── checkpoint-100/
├── checkpoint-200/
└── checkpoint-final/                 # 最终模型
```

---

## 🔧 常见问题

### Q1: 训练时显存不足?

**解决方案**:
```bash
# 减小批次大小
--ppo.batch_size 2 \
--ppo.gradient_accumulation_steps 16

# 启用梯度检查点
--gradient_checkpointing true

# 使用更小的模型
--model.path "qwen/Qwen2-7B-Instruct"
```

### Q2: vLLM启动失败?

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装vLLM
pip uninstall vllm -y
pip install vllm==0.6.3 --no-cache-dir

# 使用CPU模式(仅测试)
vllm serve <model> --device cpu
```

### Q3: LangChain-Chatchat连接失败?

**解决方案**:
```bash
# 检查服务状态
curl http://localhost:7861/docs

# 查看日志
cd Langchain-Chatchat
tail -f logs/api.log

# 重启服务
python startup.py -a
```

### Q4: 查询重写效果不理想?

**解决方案**:
1. **增加训练数据**: 从对话日志中提取更多样本
2. **调整奖励函数**: 修改 `code/verl/utils/reward_score/wuboshi_reward.py`
3. **使用更大模型**: 换用13B或更大参数的基座模型
4. **延长训练时间**: 增加epoch数或降低学习率

---

## 📊 性能优化建议

### 1. 批量处理

```python
# 批量重写,减少API调用
queries = ["问题1", "问题2", "问题3"]
results = query_rewriter.batch_rewrite(queries)
```

### 2. 启用缓存

```python
# 查询重写器自带LRU缓存
# 缓存大小在初始化时设置
rewriter = QueryRewriter(cache_size=2000)
```

### 3. 异步处理

```python
import asyncio

async def process_queries(queries):
    tasks = [rewriter.rewrite_async(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### 4. 降级策略

```python
# 超时或失败时使用规则重写
result = query_rewriter.rewrite_with_fallback(query)
# result['method'] 可能是: 'model' | 'rule' | 'none'
```

---

## 📈 监控指标

### 关键指标

1. **查询重写指标**
   - 重写成功率
   - 平均延迟
   - 缓存命中率

2. **检索性能指标**
   - Recall@10
   - MRR (Mean Reciprocal Rank)
   - NDCG@10

3. **业务指标**
   - 用户满意度
   - 多轮对话次数
   - 人工介入率

### 监控脚本

```python
# 查看重写日志
tail -f query_rewrite.log

# 分析性能
python -c "
import json
with open('query_rewrite.log') as f:
    logs = [json.loads(line) for line in f]
    avg_latency = sum(l['latency_ms'] for l in logs) / len(logs)
    print(f'平均延迟: {avg_latency:.1f}ms')
"
```

---

## 🎓 进阶使用

### 自定义奖励函数

```python
# code/verl/utils/reward_score/custom_reward.py
class CustomReward:
    def compute_reward(self, original_query, rewritten_query):
        # 实现你的奖励逻辑
        # 例如: 基于用户点击率、转化率等
        pass
```

### 多模型融合

```python
# 使用多个重写模型,选择最佳结果
rewriters = [
    QueryRewriter(api_url="http://localhost:8001"),
    QueryRewriter(api_url="http://localhost:8002")
]

results = [r.rewrite(query) for r in rewriters]
best_rewrite = select_best(results)  # 自定义选择策略
```

### A/B测试

```python
def route_request(user_id, query):
    # 根据user_id分组
    if hash(user_id) % 2 == 0:
        return search_with_rewrite(query)  # A组
    else:
        return search_direct(query)        # B组
```

---

## 🔗 相关资源

- **DeepRetrieval论文**: [链接]
- **LangChain-Chatchat文档**: https://github.com/chatchat-space/Langchain-Chatchat
- **vLLM文档**: https://docs.vllm.ai/
- **WandB监控**: https://wandb.ai/

---

## 📞 支持

如遇到问题:
1. 查看 `DeepRetrieval_LangChain_Integration_Plan.md` 完整文档
2. 检查日志文件定位问题
3. 提交Issue并附带完整错误信息

---

## ✅ 验收清单

完成以下检查确保集成成功:

- [ ] 数据准备完成,生成train/dev/corpus文件
- [ ] 训练完成,模型保存在outputs目录
- [ ] vLLM服务启动,可正常调用API
- [ ] LangChain-Chatchat知识库创建成功
- [ ] 查询重写模块集成到LangChain-Chatchat
- [ ] 集成测试通过,检索性能有提升
- [ ] A/B测试结果满意

---

**祝你集成顺利! 🎉**

如有任何问题,请参考完整设计方案或联系技术支持。

