# 数据转换脚本

## convert_to_sft_format_v2.py

主要的数据转换脚本，用于将Excel训练数据转换为SFT训练格式。

### 使用方法

```bash
python convert_to_sft_format_v2.py
```

### 功能

1. **读取Excel数据**：从 `../../code/data/橙啦-query_RL_训练集.xlsx` 读取
2. **数据清洗**：过滤空值和低质量样本
3. **格式转换**：转换为JSONL格式
4. **添加<think>标记**：保持模型推理能力
5. **数据划分**：自动划分训练/验证/测试集

### 输出位置

```
../data/sft/chengla_v2/
├── train_latest.jsonl
├── val_latest.jsonl
├── test_latest.jsonl
├── stats_report.json
└── sample_examples.json
```

### 自定义配置

修改脚本中的参数：

```python
converter = SFTDataConverterV2(tenant_id="chengla")

samples = converter.convert_excel_to_jsonl(
    excel_path=r"../../code/data/橙啦-query_RL_训练集.xlsx",
    output_dir="../data/sft/chengla_v2",
    quality_filter=True  # 设置为False跳过质量过滤
)
```
