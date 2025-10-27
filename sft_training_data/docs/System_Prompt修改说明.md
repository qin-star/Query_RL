# System Prompt 修改后的脚本调整说明

## 📊 问题分析

你修改后的system prompt包含了**结构化输入要求**，期望：
- **历史对话内容**（`{{history_chat}}`）
- **用户当前输入**（`{{query}}`）  
- **可选的思路提示**（`{{thought}}`）

但原脚本将**整个对话历史作为一个整体**传入，没有做拆分。

---

## ✅ 已完成的修改

### 1. **修改了 `convert_to_messages_format` 函数**

**修改前**：
```python
# 用户输入：只提供对话历史
user_content = f"""请分析以下销售-客户对话历史，生成用户画像、历史摘要和改写问题：

对话历史：
{context}  # 整个对话历史作为一个整体

请以JSON格式输出分析结果。"""
```

**修改后**：
```python
# 拆分对话历史：提取最后一条客户消息作为当前query
lines = context.strip().split('\n')

# 从后往前找最后一条[客户]消息
current_query = ""
history_chat = ""

for i in range(len(lines) - 1, -1, -1):
    line = lines[i].strip()
    if line.startswith('[客户]'):
        # 找到最后一条客户消息
        if '：' in line:
            current_query = line.split('：', 1)[1].strip()
        else:
            current_query = line
        
        # 其余部分作为历史对话
        history_chat = '\n'.join(lines[:i]) if i > 0 else ""
        break

# 构建用户输入 - 匹配system prompt的结构
user_content = f"""请基于以下信息，依次完成三个任务：

## 输入信息

- 历史对话内容：
{history_chat if history_chat else "（无历史对话）"}

- 用户当前输入：
{current_query}

请按照要求输出JSON格式的分析结果。"""
```

**关键改动**：
1. ✅ 提取最后一条`[客户]`消息作为`current_query`
2. ✅ 将之前的对话作为`history_chat`
3. ✅ 按照system prompt的结构组织输入

---

## 🔍 System Prompt 中的注意点

### 1. **占位符语法说明**

你的system prompt中使用了Jinja2风格的占位符：
```
{{history_chat}}
{{query}}
{{thought}}
```

**说明**：这些只是**示例占位符**，用于说明输入格式。在实际使用时：
- 我们用Python的f-string来填充这些内容
- 不需要真的使用Jinja2模板引擎

### 2. **可选字段 `thought`**

System prompt中提到了可选的`{{thought}}`字段：
```
{% if thought %}
- 销售思路提示（可选）：
"{{thought}}"
{% endif %}
```

**当前处理**：
- 你的数据集中**没有**这个字段
- 当前脚本**没有**传入thought
- 如果未来需要，可以在user_content中添加

---

## 📝 训练数据格式示例

### 完整的训练样本结构

```json
{
  "messages": [
    {
      "role": "system",
      "content": "## 背景\n你是教培行业的对话理解与用户洞察专家...\n\n（完整system prompt）"
    },
    {
      "role": "user",
      "content": "请基于以下信息，依次完成三个任务：\n\n## 输入信息\n\n- 历史对话内容：\n[销售][2025-03-03 20:47:47]：麻烦咱们孩子加老师微信...\n[客户][2025-03-03 20:48:25]：和孩子还没联系上\n...\n\n- 用户当前输入：\n以后孩子听课在这里吗？\n\n请按照要求输出JSON格式的分析结果。"
    },
    {
      "role": "assistant",
      "content": "{\n  \"user_profile\": \"用户为家长，孩子在大学读书，正在为孩子报名公考培训课程\",\n  \"history_summary\": \"用户正在为孩子报名课程，关心课程学习方式和平台\",\n  \"rewritten_query\": \"橙啦公考课程的学习平台在哪里？如何听课？\"\n}"
    }
  ],
  "metadata": {
    "source": "chengla_rl_dataset",
    "tenant_id": "chengla",
    "sample_id": "chengla_v2_1",
    "original_query": "以后孩子听课在这里吗？",
    "task_type": "multi_output"
  }
}
```

---

## ✅ 验证测试

已创建测试脚本 `test_data_format.py`，验证结果：

### 测试1：正常对话拆分
```
【原始对话】
[销售][...]：麻烦咱们孩子加老师微信...
[客户][...]：和孩子还没联系上
...
[客户][...]：以后孩子听课在这里吗？

【拆分结果】
✅ 历史对话：6行对话历史
✅ 当前query：以后孩子听课在这里吗？
```

### 测试2：边缘情况
- ✅ 只有一条客户消息 → 历史为空，query正确提取
- ✅ 多条交替对话 → 正确拆分历史和当前query

---

## 🚀 使用方法

### 1. 运行转换脚本

```bash
python convert_to_sft_format_v2.py
```

**输出**：
```
data/sft/chengla_v2/
├── train_latest.jsonl      # 训练集
├── val_latest.jsonl        # 验证集
├── test_latest.jsonl       # 测试集
├── stats_report.json       # 统计报告
└── sample_examples.json    # 样本示例
```

### 2. 检查生成的数据

```bash
# 查看样本示例
cat data/sft/chengla_v2/sample_examples.json

# 查看统计报告
cat data/sft/chengla_v2/stats_report.json

# 查看第一条训练数据
head -n 1 data/sft/chengla_v2/train_latest.jsonl | python -m json.tool
```

---

## 🎯 后续建议

### 1. **是否需要添加 `thought` 字段？**

如果你的实际应用中需要"销售思路提示"，可以：

**选项A：忽略thought字段**
- 当前数据集没有这个字段
- System prompt中标记为可选
- 训练时模型会学习在没有thought的情况下工作

**选项B：添加thought字段**
- 如果未来有这个数据，修改`convert_to_messages_format`函数
- 在user_content中添加thought部分

### 2. **System Prompt 优化建议**

当前system prompt很详细（~100行），可以考虑：

**优化1：简化版本**（用于快速验证）
```python
system_prompt_simple = """你是教培行业对话分析专家。

任务：分析对话历史，输出JSON格式：
{
  "user_profile": "用户画像",
  "history_summary": "历史摘要",
  "rewritten_query": "改写问题"
}

输入：
- 历史对话内容
- 用户当前输入

要求：保持自然，不编造信息。"""
```

**优化2：去掉占位符语法**

将system prompt中的`{{history_chat}}`、`{{query}}`等改为直接说明：
```
## 输入信息说明
你将收到两部分信息：
1. 历史对话内容
2. 用户当前输入
```

### 3. **数据质量检查**

转换后建议检查：
```python
# 检查脚本
import json

with open('data/sft/chengla_v2/train_latest.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5:  # 只检查前5条
            break
        
        sample = json.loads(line)
        
        # 检查1：user content是否包含"历史对话内容"和"用户当前输入"
        user_msg = sample['messages'][1]['content']
        assert '历史对话内容' in user_msg
        assert '用户当前输入' in user_msg
        
        # 检查2：assistant输出是否为合法JSON
        assistant_msg = sample['messages'][2]['content']
        output = json.loads(assistant_msg)
        assert 'user_profile' in output
        assert 'history_summary' in output
        assert 'rewritten_query' in output
        
        print(f"✅ 样本 {i+1} 格式正确")
```

---

## 📌 总结

### 修改清单

| 项目 | 状态 | 说明 |
|------|------|------|
| 拆分对话历史和当前query | ✅ 已完成 | `convert_to_messages_format`函数已修改 |
| 匹配system prompt结构 | ✅ 已完成 | user_content格式已调整 |
| 测试验证 | ✅ 已完成 | 创建了`test_data_format.py` |
| 文档说明 | ✅ 已完成 | 本文档 |

### 可选优化

| 项目 | 优先级 | 说明 |
|------|--------|------|
| 添加thought字段支持 | 低 | 当前数据集无此字段 |
| 简化system prompt | 中 | 如果模型学习困难可考虑 |
| 添加数据质量检查 | 高 | 训练前建议执行 |

---

## ❓ FAQ

### Q1: System prompt太长会有问题吗？

**A**: 不会，但可能：
- 增加token消耗
- 影响模型学习效率（如果规则太复杂）

**建议**：先用当前版本训练，如果效果不好再简化。

### Q2: 为什么要拆分历史对话和当前query？

**A**: 
1. **匹配system prompt的结构**：你的prompt明确要求分开输入
2. **更清晰的语义**：帮助模型理解哪个是需要改写的问题
3. **更好的训练效果**：结构化输入有助于模型学习

### Q3: 如果对话历史中有多条客户消息怎么办？

**A**: 
- 当前逻辑：取**最后一条**`[客户]`消息作为current_query
- 其余所有内容（包括之前的客户消息）作为history_chat
- 这符合多轮对话的场景

---

## 🎉 准备开始训练

当前脚本已经完全适配你修改后的system prompt！

**下一步**：
```bash
# 1. 运行转换脚本
python convert_to_sft_format_v2.py

# 2. 检查生成的数据
python -c "import json; print(json.dumps(json.loads(open('data/sft/chengla_v2/sample_examples.json').read())[0], ensure_ascii=False, indent=2))"

# 3. 开始训练（参考MD文档中的ms-swift命令）
```


