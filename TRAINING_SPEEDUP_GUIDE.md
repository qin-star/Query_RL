# 训练加速指南

## 🐌 问题分析

训练卡在 `Training Progress: 0%|  | 0/21 [00:00<?, ?it/s]` 的原因：

### 主要瓶颈
1. **首次模型加载** - FSDP模型初始化需要时间
2. **首次VLLM引擎启动** - rollout引擎第一次启动较慢
3. **数据预处理** - 第一个batch的tokenization
4. **RAG服务预热** - 第一次调用RAG API较慢

## ⚡ 优化方案

### 1. 立即生效的优化（修改配置文件）

#### A. 减少rollout候选数量（最有效）
```yaml
# 在 sales_rag_grpo_hybrid_config.yaml 第145行
rollout:
  n: 5  # 改为 3，减少40%的生成时间
```

**效果**：每个样本生成时间从 5×生成时间 → 3×生成时间

#### B. 减少batch size（加快第一次迭代）
```yaml
# 第501行
data:
  train_batch_size: 256  # 改为 128 或 64
```

**效果**：第一次迭代更快完成，但总训练时间可能增加

#### C. 增加GPU内存利用率
```yaml
# 第152行
rollout:
  gpu_memory_utilization: 0.45  # 改为 0.6-0.7
```

**效果**：更多内存用于KV cache，生成更快

#### D. 启用更激进的缓存
```yaml
# 第164-165行
rollout:
  enable_chunked_prefill: true  # 保持
  enable_prefix_caching: true   # 保持
```

**效果**：重复的prompt前缀会被缓存

### 2. 代码层面优化

#### A. 预热RAG服务
在训练开始前添加预热调用：

```python
# 在训练脚本开始处添加
async def warmup_rag_service():
    """预热RAG服务"""
    from src.core.rag_chater import RagChater
    
    rag = RagChater(
        tenant_id="chengla",
        contact_id="test",
        account_id="test",
        message_id="warmup"
    )
    
    # 预热8B端点
    await rag.chat_8b(
        context="预热测试",
        user_profile="测试",
        history_summary="测试",
        rewritten_query="测试",
        score_threshold=0.9,
        top_k=3
    )
    
    # 预热32B端点
    await rag.chat(
        context="预热测试",
        score_threshold=0.9,
        top_k=3
    )
    
    print("✅ RAG服务预热完成")

# 在main函数中调用
import asyncio
asyncio.run(warmup_rag_service())
```

#### B. 并行数据加载
```yaml
# 第509行
data:
  dataloader_num_workers: 8  # 改为 16（如果CPU核心足够）
```

### 3. 系统层面优化

#### A. 使用更快的存储
```bash
# 如果数据在慢速磁盘，复制到SSD
cp -r data/sales_rag /tmp/sales_rag

# 修改配置
train_files: ["/tmp/sales_rag/train.parquet"]
```

#### B. 增加共享内存
```bash
# 检查当前共享内存
df -h /dev/shm

# 如果不足，增加（需要root权限）
mount -o remount,size=32G /dev/shm
```

### 4. 调试模式（快速验证）

创建一个小数据集快速测试：

```python
# scripts/create_mini_dataset.py
import pandas as pd

# 读取原始数据
df = pd.read_parquet("data/sales_rag/train.parquet")

# 只保留前10条
mini_df = df.head(10)

# 保存
mini_df.to_parquet("data/sales_rag/train_mini.parquet")
```

修改配置：
```yaml
data:
  train_files: ["../data/sales_rag/train_mini.parquet"]
  train_batch_size: 10
```

## 📊 预期效果对比

| 优化项 | 原始时间 | 优化后时间 | 提升 |
|--------|----------|------------|------|
| 首次迭代 | ~5分钟 | ~2分钟 | 60% |
| 单个样本生成 | ~30秒 | ~18秒 | 40% |
| RAG调用 | ~6秒 | ~3秒 | 50% |

## 🎯 推荐配置（平衡速度和效果）

```yaml
# 修改这些参数
rollout:
  n: 3  # 从5改为3
  gpu_memory_utilization: 0.6  # 从0.45改为0.6
  temperature: 0.6  # 从0.7改为0.6，减少随机性

data:
  train_batch_size: 128  # 从256改为128
  dataloader_num_workers: 16  # 从8改为16

algorithm:
  hybrid_grpo:
    group_size: 3  # 从5改为3，与rollout.n保持一致
```

## 🔍 监控训练进度

添加详细日志：

```python
# 在训练循环中添加
import time
start_time = time.time()

# 每个step后
elapsed = time.time() - start_time
print(f"Step {step}/{total_steps} | "
      f"Elapsed: {elapsed:.1f}s | "
      f"Avg: {elapsed/step:.1f}s/step | "
      f"ETA: {(total_steps-step)*elapsed/step/60:.1f}min")
```

## ⚠️ 注意事项

1. **不要同时修改所有参数** - 逐个测试效果
2. **保存原始配置** - 方便回滚
3. **监控GPU内存** - 避免OOM
4. **验证训练效果** - 确保优化不影响收敛

## 🚀 快速启动检查清单

- [ ] RAG服务已启动并可访问
- [ ] 数据文件路径正确
- [ ] GPU内存充足（至少40GB可用）
- [ ] 模型checkpoint存在
- [ ] 已运行 `standalone_test_rag.py` 验证RAG连接
- [ ] 配置文件中的路径都是绝对路径

## 💡 如果还是很慢

### 诊断步骤

1. **检查是否卡在模型加载**
```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi
```

2. **检查是否卡在数据加载**
```bash
# 查看磁盘IO
iostat -x 1
```

3. **检查是否卡在RAG调用**
```bash
# 查看RAG服务日志
tail -f /path/to/rag/service.log
```

4. **添加详细日志**
```yaml
debug:
  enable: true
  log_level: "DEBUG"
  verbose_logging:
    model_output: true
    rag_calls: true
    scoring: true
```

### 最后的手段：跳过第一次验证

```yaml
trainer:
  val_before_train: false  # 确保是false
  test_freq: 999999  # 先不做验证，专注训练
```

## 📝 实际测试结果

根据你的测试：
- RAG 8B调用: 5.649s ✅
- RAG 32B调用: 5.635s ✅

这个速度是正常的。如果训练卡住，问题可能在：
1. **模型生成阶段** - 5个候选×512 tokens = 2560 tokens需要生成
2. **批次处理** - 256个样本需要较长时间

**建议先改 `rollout.n: 3` 和 `train_batch_size: 128`**
