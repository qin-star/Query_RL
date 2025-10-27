# SFT训练数据转换工具

## 📁 目录结构

```
sft_training_data/
├── scripts/          # 数据转换脚本
├── tests/           # 测试脚本
├── docs/            # 说明文档
└── data/            # 输出数据（运行后生成）
```

## 🚀 快速开始

### 1. 数据转换

```bash
cd scripts
python convert_to_sft_format_v2.py
```

**输出**：
- `../data/sft/chengla_v2/train_latest.jsonl` - 训练集
- `../data/sft/chengla_v2/val_latest.jsonl` - 验证集
- `../data/sft/chengla_v2/test_latest.jsonl` - 测试集
- `../data/sft/chengla_v2/stats_report.json` - 统计报告

### 2. 运行测试

```bash
cd tests
python test_think_tag.py
```

### 3. 查看文档

所有说明文档都在 `docs/` 目录下。

## 📊 数据格式

### 输入数据

Excel文件：`code/data/橙啦-query_RL_训练集.xlsx`

包含字段：
- 最终传参上下文（对话历史）
- rewritten_query（改写后的query）
- user_profile（用户画像）
- history_summary（历史摘要）

### 输出格式

JSONL格式，每行一个JSON对象：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "系统prompt..."
    },
    {
      "role": "user",
      "content": "用户输入..."
    },
    {
      "role": "assistant",
      "content": "<think>\n\n</think>\n\n{JSON输出}"
    }
  ],
  "metadata": {...}
}
```

## 🎯 特性

- ✅ **多任务输出**：同时生成 user_profile、history_summary、rewritten_query
- ✅ **<think>标记**：保持模型推理能力
- ✅ **数据质量过滤**：自动过滤低质量样本
- ✅ **自动划分**：训练集80% / 验证集10% / 测试集10%

## 📖 详细文档

查看 `docs/` 目录获取详细说明：
- System_Prompt修改说明.md
- Think标记使用说明.md
- 数据格式对比说明.md

## ⚙️ 配置

编辑 `scripts/convert_to_sft_format_v2.py` 中的参数：
- `tenant_id`: 租户ID（默认："chengla"）
- `quality_filter`: 是否进行质量过滤（默认：True）
- `train_ratio`: 训练集比例（默认：0.8）
- `val_ratio`: 验证集比例（默认：0.1）
