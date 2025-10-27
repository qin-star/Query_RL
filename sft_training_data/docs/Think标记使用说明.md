# <think> 标记使用说明

## 📋 功能说明

已在SFT训练数据中添加`<think></think>`标记，用于保持模型的推理能力。

### 为什么需要<think>标记？

在大语言模型微调中，如果直接让模型输出最终答案，可能会：
- ❌ 丧失推理过程
- ❌ 降低复杂问题的解决能力
- ❌ 无法展示思维链（Chain of Thought）

通过添加`<think></think>`标记：
- ✅ 为模型预留"思考空间"
- ✅ 保持推理能力
- ✅ 训练后可以看到模型的思考过程

---

## 🎯 修改内容

### 修改前的输出格式

```json
{
  "user_profile": "用户为备考公务员的应届生",
  "history_summary": "用户正在了解课程信息",
  "rewritten_query": "公务员考试培训课程的价格是多少？"
}
```

### 修改后的输出格式

```
<think>

</think>

{
  "user_profile": "用户为备考公务员的应届生",
  "history_summary": "用户正在了解课程信息",
  "rewritten_query": "公务员考试培训课程的价格是多少？"
}
```

---

## 📝 完整训练样本示例

```json
{
  "messages": [
    {
      "role": "system",
      "content": "## 背景\n你是教培行业的对话理解与用户洞察专家..."
    },
    {
      "role": "user",
      "content": "请基于以下信息，依次完成三个任务：\n\n## 输入信息\n\n- 历史对话内容：\n[销售][...]...\n\n- 用户当前输入：\n我想了解课程价格\n\n请按照要求输出JSON格式的分析结果。"
    },
    {
      "role": "assistant",
      "content": "<think>\n\n</think>\n\n{\n  \"user_profile\": \"用户为备考公务员的应届生\",\n  \"history_summary\": \"用户正在了解课程信息\",\n  \"rewritten_query\": \"公务员考试培训课程的价格是多少？\"\n}"
    }
  ]
}
```

---

## 🔧 代码修改

### 修改的函数：`convert_to_messages_format`

```python
# 助手输出：JSON格式
assistant_output = {
    "user_profile": user_profile.strip() if pd.notna(user_profile) else "",
    "history_summary": history_summary.strip() if pd.notna(history_summary) else "",
    "rewritten_query": rewritten_query.strip() if pd.notna(rewritten_query) else ""
}

# 转换为JSON字符串（格式化输出）
json_output = json.dumps(assistant_output, ensure_ascii=False, indent=2)

# ⭐ 在输出前添加<think>标记，保持模型推理能力
assistant_content = f"<think>\n\n</think>\n\n{json_output}"
```

### 修改的函数：`generate_stats_report`

```python
# 解析assistant的JSON输出（需要去掉<think>标记）
assistant_content = sample['messages'][2]['content']

# ⭐ 提取JSON部分（去掉<think>\n\n</think>\n\n前缀）
if assistant_content.startswith('<think>\n\n</think>\n\n'):
    json_str = assistant_content.replace('<think>\n\n</think>\n\n', '', 1)
else:
    json_str = assistant_content

output_json = json.loads(json_str)
```

---

## 🚀 训练效果预期

### 训练前（无<think>标记）

```
输入：请分析对话...
输出：{"user_profile": "...", ...}
```
❌ 模型直接输出答案，无推理过程

### 训练后（有<think>标记）

```
输入：请分析对话...
输出：<think>
用户最后提问是关于课程价格，之前讨论了报名事宜...
</think>

{"user_profile": "...", ...}
```
✅ 模型先思考再输出，保持推理能力

---

## 🎓 推理时的使用

### 标准推理模式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model = AutoModelForCausalLM.from_pretrained("output/chengla_v2/final")
tokenizer = AutoTokenizer.from_pretrained("output/chengla_v2/final")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 解析响应
if '<think>' in response:
    # 提取思考过程
    think_start = response.find('<think>')
    think_end = response.find('</think>')
    thinking_process = response[think_start+7:think_end].strip()
    
    # 提取JSON结果
    json_start = think_end + 9  # '</think>\n\n' 的长度
    json_str = response[json_start:].strip()
    
    result = json.loads(json_str)
    
    print("🧠 思考过程：")
    print(thinking_process)
    print("\n📊 分析结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    # 如果没有<think>标记（fallback）
    result = json.loads(response)
```

### 快速模式（忽略思考过程）

```python
# 如果只需要最终结果
response = model.generate(...)
response_text = tokenizer.decode(...)

# 提取JSON部分
if response_text.startswith('<think>\n\n</think>\n\n'):
    json_str = response_text.replace('<think>\n\n</think>\n\n', '', 1)
else:
    json_str = response_text

result = json.loads(json_str)
```

---

## ⚙️ 训练配置建议

### ms-swift训练命令（更新）

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B-Instruct \
    --train_type lora \
    --dataset data/sft/chengla_v2/train_latest.jsonl \
    --val_dataset data/sft/chengla_v2/val_latest.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --max_length 2048 \
    --output_dir output/chengla_v2 \
    --warmup_ratio 0.1
```

**关键参数**：
- `max_length=2048`：需要足够长以容纳`<think>`标记的内容
- 建议在验证集上观察模型是否学会使用`<think>`标记

---

## 🔍 验证测试

### 测试脚本已创建

```bash
python test_think_tag.py
```

**测试内容**：
1. ✅ 验证`<think>`标记是否正确添加
2. ✅ 验证JSON格式是否正确
3. ✅ 验证解析逻辑是否正常
4. ✅ 展示完整的messages结构

---

## 📊 对比：有无<think>标记的区别

| 特性 | 无<think>标记 | 有<think>标记 |
|------|--------------|--------------|
| **输出长度** | 较短 | 稍长 |
| **推理能力** | 可能退化 | 保持良好 |
| **复杂问题** | 表现一般 | 表现更好 |
| **可解释性** | 黑盒 | 可见思考过程 |
| **训练成本** | 略低 | 略高（token增加） |
| **推理成本** | 略低 | 略高（生成更长） |

---

## ⚠️ 注意事项

### 1. Token消耗

添加`<think>`标记后，每个样本会增加约20个token：
- 训练数据：474条 × 20 tokens = ~10,000 extra tokens
- 对总体训练成本影响较小

### 2. 推理时的处理

推理时需要正确解析带`<think>`标记的输出：
```python
# 错误示例
result = json.loads(response)  # ❌ 会报错，因为包含<think>

# 正确示例
json_str = response.replace('<think>\n\n</think>\n\n', '', 1)
result = json.loads(json_str)  # ✅ 正确
```

### 3. 空的<think>标记

当前实现中，`<think></think>`之间是空的，这是正常的：
- 训练时：模型学习在这里"思考"
- 推理时：模型会在这里生成思考内容

如果希望在训练数据中也包含思考内容，需要：
1. 人工标注或使用GPT生成思考过程
2. 修改数据生成脚本添加思考内容

---

## 🎯 最佳实践建议

### 训练阶段

1. **从空<think>开始**：先用空的`<think></think>`训练
2. **观察效果**：验证模型是否学会在这里生成内容
3. **迭代优化**：如果效果不佳，考虑添加示例思考内容

### 推理阶段

1. **保留思考过程**：便于调试和理解
2. **可选优化**：生产环境可以设置`stop_tokens=['</think>']`提前停止，节省成本
3. **监控质量**：定期检查思考过程是否有意义

---

## ✅ 修改完成清单

- ✅ 修改`convert_to_messages_format`函数，添加`<think>`标记
- ✅ 修改`generate_stats_report`函数，正确解析带标记的输出
- ✅ 创建测试脚本`test_think_tag.py`
- ✅ 创建说明文档（本文档）

---

## 🚀 下一步

现在可以运行转换脚本生成训练数据：

```bash
python convert_to_sft_format_v2.py
```

生成的训练数据将包含`<think></think>`标记，可以直接用于模型训练！

