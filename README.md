# Sales-RAG Query改写RL训练方案（精简版）

> 基于Qwen-8B的两阶段训练：SFT知识蒸馏 + RL竞争优化
>
> **版本**: v2.0 | **状态**: 已优化精简 | **更新**: 2025-01-20

---

## 📋 方案概述

### 核心思路

先在 Qwen-8B 小模型上尝试整个训练流程，随后再将方案替换为对 Qwen-32B 的训练。若发现在 Qwen-8B 的训练效果足以满足业务要求，则直接部署上线，充分发挥小模型的优势Qwen-8B 初步尝试：
```
Qwen-8B 初步尝试：
    阶段1: SFT知识蒸馏
        Qwen-32B (Teacher) → 改写数据 → Qwen-8B (Student) SFT训练
    阶段2: RL竞争优化  
        Qwen-8B ↔ Qwen-32B (双模型竞争) + GPT-5评分 → PPO/GRPO优化
--------------------
Qwen-32B 进阶尝试：
    阶段1: SFT知识蒸馏
        GPT-5/DeepSeek V3.1 (Teacher) → 生成改写数据 → Qwen-32B (Student) SFT训练  
    阶段2: RL竞争优化  
        Qwen-32B (训练中) ↔ GPT-5/DeepSeek V3.1 (Baseline) + 实时RAG检索 → PPO优化
```    

### 技术栈

- **基座模型**: Qwen3-8B-Instruct
- **教师模型**: Qwen-32B (现有部署)
- **评分模型**: GPT-5 (API调用)
- **RL算法**: PPO (Proximal Policy Optimization)
- **训练框架**: VERL

---

## 1️⃣ SFT训练数据准备

### 1.1 数据来源

使用BVT测试集批量测试RAG框架，收集32B改写结果：

```python
{
    "original_query": "胶原蛋白怎么吃",
    "rewritten_query": "胶原蛋白肽 服用方法 推荐用量",
    "user_profile": "25-35岁女性，关注抗衰老",
    "history_summary": "近期咨询过多次胶原蛋白产品",
    "top1_score": 0.87,
    "recall_count": 5
}
```

### 1.2 数据转换

**步骤1**: 批量RAG测试 → 保存到 `test_sft.xlsx`

**步骤2**: 质量筛选（top1_score > 0.6）

**步骤3**: 转换为JSONL训练格式

```python
# convert_to_sft_format.py
converter = TestToSFTConverter(tenant_id="fivedoctors")

samples = converter.convert_excel_to_jsonl(
    excel_path="data/test_sft_fivedoctors.xlsx",
    output_jsonl="data/sft/fivedoctors/all_samples.jsonl",
    quality_threshold=0.6
)

# 划分数据集
converter.split_train_val_test(
    jsonl_path="data/sft/fivedoctors/all_samples.jsonl"
)
```

**预期数据规模**:

- 测试集: 500-1000条
- 筛选后: 300-800条
- 训练集: 240-640条
- 验证集: 30-80条

### 1.3 SFT训练

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset data/sft/fivedoctors/train_latest.jsonl \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir outputs/sft/fivedoctors
```

---

## 2️⃣ GPT-5评分器设计

### 2.1 核心原则

**严格可复核**: 仅依据检索结果及排序变化，不引入外部知识或主观推测

**对比式评分**: 同时评估32B和8B两个方案，直接给出better判断

### 2.2 评分维度（4个维度，0-10分）

| 维度         | 权重 | 说明               |
| ------------ | ---- | ------------------ |
| 质量提升度   | 40%  | 改进程度、噪声抑制 |
| 相关性准确性 | 20%  | 语义匹配度         |
| 信息完整性   | 20%  | 关键信息覆盖       |
| 检索有效性   | 20%  | 召回结果可用性     |

### 2.3 评分Prompt

```python

你是一名严格、可复核的RAG改写评估专家。
请仅依据提供的检索结果及排序变化进行对比分析，不得引入任何外部知识、常识或主观推测。
你的任务是客观、量化地判断两种方案（qwen3-32b 与 qwen3-8b）在改写查询、用户画像、历史消息总结效果上的优劣，

客户与销售的聊天记录
{{ history_chat }}

【输入】
方案一（qwen3-32b方案）：

用户画像: {{ user_profile }}
改写查询: {{ rewritten_query }}
历史消息总结: {{ history_summary }}
RAG 召回结果: {{ rag_recall }}
方案二（qwen3-8b方案）：

用户画像: {{ user_profile_8B}}
改写查询: {{ rewritten_query_8B}}
历史消息总结: {{ history_summary_8B}}
RAG 召回结果: {{ rag_recall_8B }}
其中  
用户画像、改写查询、历史消息总结均是 32B 或 8B 模型的输出结果。  
RAG 召回结果是使用 rewritten_query 查询 RAG 系统的结果。

【评估维度（共四项）】  
（评分均为整数0–10，禁止输出小数）

1️⃣ 质量提升度  
衡量方案生成的三部分结果质量提升程度。  
10：显著改进，高相关内容系统性前置，噪声明显抑制。  
7–9：改进明显，大部分高相关内容前置，少量噪声残留。  
4–6：改进有限，排序混乱或噪声较多。  
0–3：几乎无改进，高相关内容被噪声掩盖。  

2️⃣ 相关性准确性  
衡量方案生成三部分结果与客户和销售历史对话是否准确。  
10：与问题高度匹配，语义理解精准。  
7–9：匹配良好，意图理解基本准确。  
4–6：相关性一般，存在语义偏差。  
0–3：相关性差，理解错误。  

3️⃣ 信息完整性  
衡量方案生成三部分结果与客户和销售历史对话是否覆盖回答问题所需的全部关键信息。  
判断标准应基于方案生成三部分结果是否覆盖回答问题所需的关键知识点、逻辑链条或证据类型，而非文本长度或表述丰富度。  
10：完整覆盖所有关键信息，无缺失。  
7–9：覆盖主要信息，次要信息略有缺失。  
4–6：关键信息不全，影响回答质量。  
0–3：缺失严重，无法支持有效回答。  

4️⃣ 检索有效性  
衡量使用改写后的 rewritten_query 查询的 RAG 召回结果能否更好地回复用户问题。  
RAG 召回结果中可能包含以下情况：
- "lack of knowledge"：当前改写未能找到知识，若另一方案可找到知识，则该方案应得更高分。
- "no knowledge required"：需判断当前用户问题是否真的无需知识支撑，若是则判断准确的方案得高分
评分标准：  
10：改写对 RAG 召回完全覆盖、回答用户问题。  
7–9：改写产生的 RAG 召回部分覆盖用户问题。  
4–6：改写产生的 RAG 召回结果存在无关信息。  
0–3：召回缺失严重，无法支持回答。  

【评分说明】  
- 四项指标评分逻辑应保持一致性（例如，若排序质量显著提升，应与相关性提升保持一致）。  
- 若存在矛盾或不确定性，请在理由中说明。  
- 禁止拒绝评分或输出“无法判断”。  

【分析与理由结构】  
请在 "reason" 中简要说明以下四部分（不少于4句）：  
1. 改进度分析：描述 32b 与 8b 在生成三部分结果的具体表现。  
2. 相关性对比：说明两者最终结果与问题的语义匹配差异。  
3. 完整性检查：指出哪一方案信息更完整或存在缺口。  
4. 优劣结论：明确指出哪一方案更优及理由。  

【判定规则】  
- 若两方案四项评分平均差 ≤ 1 且差异不显著，输出 "same"。  
- 若两方案均表现差、三项均低分，输出 "both bad"。  
- 其余情况必须在 "better_solution" 中明确选择 "32b" 或 "8b"。  
【加权与总分】

加权比例：质量提升度 40%，相关性准确性 20%，信息完整性 20%，检索有效性 20%
总分 = 加权平均后四舍五入至整数。
所有评分必须为整数（0–10），不得包含小数或非数字。
请确保三个维度的评分逻辑一致：若排序质量显著提升，应与相关性提升保持一致性；若存在矛盾，请在理由中说明。
若任一项证据不足，仍需在理由中说明不确定性并合理估算，禁止拒绝评分。
【分析与理由结构】
请在 "reason" 中简要说明以下四部分（不少于4句）：

改进度分析：描述 32b 与 8b 在生成三部分结果的具体表现。
相关性对比：说明两者最终结果与问题的语义匹配差异。
完整性检查：指出哪一方案信息更完整或存在缺口。
优劣结论：明确指出哪一方案更优及理由。
【判定规则】

若两方案总分差 ≤ 2 且差异不显著，输出 "same"。
若两方案均表现差、三项均低分，输出 "both bad"。
其余情况必须在 "better_solution" 中明确选择 "32b" 或 "8b"。
禁止输出“无法判断”或含糊结果。
【输出格式（严格JSON）】
请严格按照以下格式输出，不得添加任何多余文本或解释：

输出需要有 4 个，better、reason、score、brief
better 是指更好的模型，32b 或 8b 或 same 或 both bad
reason 包含四项指标分析、对比说明和结论，逻辑清晰，不少于4句
score 分别输出 32b 和 8b 的 scores 和 sum，scores 对应四个指标的分数，sum 对应总分
brief 简要描述输出更优方案胜选的理由与更差方案落选的理由
{
  "better": "32b 或 8b 或 same 或 both bad",
  "reason": "包含四项指标分析、对比说明和结论，逻辑清晰，不少于4句。",
  "score": {
    "32b": {
      scores: [1,5,2,4],
      sum: 32b 总分
    }
    "8b": {
      scores: [10,8,4,1],
      sum: 8b 总分
    }
  },
  "brief": "更优方案胜选的理由与更差方案落选的理由"
}

请开始评估并严格按照JSON格式输出结果。
```

### 2.4 实现代码

```python
class GPT5QueryRewriteScorer:
    """严格可复核版GPT-5评分器"""
  
    def __init__(self, api_key, model="gpt-5", temperature=0.1):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.weights = [0.4, 0.2, 0.2, 0.2]  # 4个维度权重
  
    async def score_comparative(
        self,
        history_chat: str,
        # 32B方案
        user_profile_32b: str,
        rewritten_query_32b: str,
        history_summary_32b: str,
        rag_recall_32b: List[Dict],
        # 8B方案
        user_profile_8b: str,
        rewritten_query_8b: str,
        history_summary_8b: str,
        rag_recall_8b: List[Dict]
    ) -> Dict:
        """对比评分两个方案，返回详细评分结果"""
        # 构建prompt并调用GPT-5
        # 返回: {better, reason, score, brief}
```

---

## 3️⃣ Reward函数设计

### 3.1 评分到Reward的映射

```python
class RewardCalculator:
    def compute_reward(self, gpt5_result: Dict) -> float:
        # 1. 提取分数
        sum_8b = gpt5_result["score"]["8b"]["sum"]    # [0, 100]
        sum_32b = gpt5_result["score"]["32b"]["sum"]  # [0, 100]
        better = gpt5_result["better"]
  
        # 2. 计算分数差距（归一化）
        score_diff = (sum_8b - sum_32b) / 100.0
  
        # 3. 平滑映射
        base_reward = np.tanh(score_diff * 2)
  
        # 4. 根据获胜情况调整
        if better == "8b":
            reward = base_reward + 0.2
        elif better == "32b":
            reward = base_reward - 0.2
        elif better == "same":
            reward = base_reward * 0.5
        elif better == "both bad":
            reward = -0.5
        else:
            reward = base_reward
  
        # 5. 截断到[-1, 1]
        return np.clip(reward, -1.0, 1.0)
```

### 3.2 Reward示例

| 场景       | sum_32b | sum_8b | better   | reward |
| ---------- | ------- | ------ | -------- | ------ |
| 8B显著胜出 | 65      | 88     | 8b       | +0.63  |
| 8B小幅胜出 | 72      | 78     | 8b       | +0.32  |
| 平局       | 75      | 76     | same     | +0.01  |
| 32B胜出    | 82      | 70     | 32b      | -0.44  |
| 双方都差   | 45      | 42     | both bad | -0.50  |

---

## 4️⃣ VERL框架集成

### 4.1 VERL简介

**VERL**：火山引擎开源的大模型RL训练框架

**核心特点**:

- 支持大规模LLM的PPO训练
- 高效的分布式训练（多GPU）
- 灵活的Reward函数接口
- 自动处理经验回放和参数更新

### 4.2 自定义Reward函数

```python
from verl.utils.reward_score import RewardFunction

class QueryRewriteRewardFunction(RewardFunction):
    """VERL标准Reward函数"""
  
    def __init__(self, gpt5_api_key, qwen32b_api, rag_api):
        super().__init__()
        self.gpt5_scorer = GPT5QueryRewriteScorer(api_key=gpt5_api_key)
        self.reward_calculator = RewardCalculator()
        self.qwen32b_api = qwen32b_api
        self.rag_api = rag_api
  
    async def __call__(self, prompts: List[str], outputs: List[str]) -> List[float]:
        """
        VERL调用接口
  
        Args:
            prompts: List[history_chat]
            outputs: List[rewrite_8b]
  
        Returns:
            rewards: List[float]
        """
        rewards = []
  
        for prompt, output_8b in zip(prompts, outputs):
            # 1. 调用32B生成baseline
            output_32b = await self._generate_32b(prompt)
  
            # 2. 并行RAG检索
            recall_8b, recall_32b = await self._parallel_rag(output_8b, output_32b)
  
            # 3. GPT-5评分
            gpt5_result = await self.gpt5_scorer.score_comparative(...)
  
            # 4. 计算reward
            reward = self.reward_calculator.compute_reward(gpt5_result)
            rewards.append(reward)
  
        return rewards
```

### 4.3 VERL训练配置

```python
verl_config = {
    "actor_model": {
        "path": "outputs/sft/Qwen-8B-sft", # 替换成实际路径
        "dtype": "bfloat16"
    },
    "critic_model": {
        "path": "outputs/sft/Qwen-8B-sft", # 替换成实际路径
        "dtype": "bfloat16"
    },
    "ppo": {
        "learning_rate": 1e-6,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "lambda_": 0.95,
        "ppo_epochs": 4,
        "batch_size": 8
    },
    "training": {
        "num_epochs": 10,
        "save_steps": 500,
        "logging_steps": 10
    }
}
```

#### 📌 Actor与Critic模型说明

**Q: 为什么配置路径相同？**

虽然配置路径相同，但实际运行时会创建**两个独立的模型实例**：

```python
# VERL内部实现（简化）
actor_model = AutoModelForCausalLM.from_pretrained(path)   # 独立副本A
critic_model = AutoModelForCausalLM.from_pretrained(path)  # 独立副本B
critic_model = add_value_head(critic_model)                # 添加价值头

# 两个对象，参数独立更新
id(actor_model) != id(critic_model)  # True
```

**模型结构对比**:

| 组件               | Actor (策略网络)     | Critic (价值网络)          |
| ------------------ | -------------------- | -------------------------- |
| **Backbone** | Qwen-8B (32层)       | Qwen-8B (32层)             |
| **输出层**   | LM Head → token概率 | Value Head → V(state)标量 |
| **作用**     | 生成改写query        | 预测状态价值               |
| **训练目标** | 最大化reward         | 准确预测reward             |
| **参数更新** | ✅ 每步更新          | ✅ 每步更新                |

**训练模式（非推理部署）**:

```python
# 训练时：参数可更新
actor_model.train()              # 启用dropout
actor_model.requires_grad = True # 允许梯度计算
optimizer.step()                 # W_new = W_old - lr * ∂L/∂W

# 推理时：参数冻结
actor_model.eval()
with torch.no_grad():
    output = model.generate(...)
```

**参数更新路径**:

```
Rollout → Reward(GPT-5) → Advantage计算 → PPO Loss → 
loss.backward() → optimizer.step() → 模型参数更新 ✅
```

### 4.4 训练启动

```python
class VERLQueryRewriteTrainer:
    def __init__(self, config):
        self.actor_model = AutoModelForCausalLM.from_pretrained(...)
        self.reward_fn = QueryRewriteRewardFunction(...)
        self.trainer = PPOTrainer(
            model=self.actor_model,
            reward_fn=self.reward_fn,
            config=config["ppo"]
        )
  
    def train(self, train_dataset):
        for epoch in range(num_epochs):
            # VERL自动处理：
            # 1. Rollout（生成改写）
            # 2. 调用reward_fn获取rewards
            # 3. 计算Advantage
            # 4. PPO参数更新
            metrics = self.trainer.train_epoch(train_dataset)
  
            wandb.log({
                "avg_reward": metrics["avg_reward"],
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"]
            })
```

### 4.5 参数更新路径

```
1. GPT-5评分 → gpt5_result: {better, score}

2. Reward计算 → reward = compute_reward(gpt5_result)
   └─▶ 返回给VERL: List[float]

3. VERL接收 → rollout_buffer.rewards = [r1, r2, ...]

4. 计算Advantage → advantages = GAE(rewards, values)

5. 计算PPO Loss → total_loss = policy_loss + value_loss

6. 反向传播 → total_loss.backward()
   └─▶ 计算梯度: ∂L/∂W

7. 参数更新 ✅ → optimizer.step()
   └─▶ W_new = W_old - lr * ∂L/∂W
   └─▶ Qwen-8B所有层参数更新

8. 下一个训练step → 使用更新后的模型继续
```

---

## 5️⃣ 完整训练流程

### 5.1 阶段1：SFT训练

```
测试集 → 批量RAG测试 → test_sft.xlsx → 
质量筛选(>0.6) → JSONL格式 → Qwen-8B SFT → 初版模型
```

### 5.2 阶段2：RL训练

```
每个训练step:

1. 输入: history_chat
   ↓
2. 并行生成:
   ├─ 8B: user_profile_8b + rewrite_8b + history_summary_8b
   └─ 32B: user_profile_32b + rewrite_32b + history_summary_32b
   ↓
3. 并行RAG检索:
   ├─ RAG(rewrite_8b) → rag_recall_8b
   └─ RAG(rewrite_32b) → rag_recall_32b
   ↓
4. GPT-5对比评分:
   输入: history_chat + 两个方案完整数据
   输出: {better, reason, score, brief}
   ↓
5. 计算Reward:
   score_diff = (sum_8b - sum_32b) / 100
   reward = tanh(score_diff * 2) + better调整
   ↓
6. PPO更新8B参数:
   Advantage = reward - V(state)
   Loss = -min(ratio*adv, clip(ratio)*adv)
   optimizer.step() → 参数更新
   ↓
7. 下一批样本...
```

### 5.3 RAG API修改

```python
@router.post("/api/chat/general_rag")
async def general_rag_endpoint(
    query: str,
    rewritten_query: Optional[str] = None,  # 新增
    ...
):
    if rewritten_query:
        new_query = rewritten_query
    else:
        new_query = await rewrite_query_by_model(...)
  
    search_res = await rag_workflow(new_query, ...)
    return {"data": {"rewritten_query": new_query, "recall": search_res}}
```

---

## 7️⃣ 关键技术要点

### PPO损失函数

```
Policy Loss = -min(ratio * Advantage, clip(ratio, 0.8, 1.2) * Advantage)
Value Loss = (Reward - V(state))²
Total Loss = Policy Loss + 0.5 * Value Loss - 0.01 * Entropy
```

### 超参数配置

| 参数          | 值   | 说明         |
| ------------- | ---- | ------------ |
| learning_rate | 1e-6 | 8B模型学习率 |
| clip_range    | 0.2  | PPO clip范围 |
| batch_size    | 8-16 | 每批样本数   |
| ppo_epochs    | 4    | 每批更新次数 |

### 训练监控

```python
{
    "avg_reward": 0.0 → 0.2,
    "8b_win_rate": 30% → 65%,
    "better_8b": 逐步增加,
    "better_32b": 逐步减少,
    "avg_score_8b": 逐步提升
}
```

---

## 8️⃣ 预期效果

| 指标            | Baseline (32B) | SFT (8B) | RL (8B) |
| --------------- | -------------- | -------- | ------- |
| 改写质量评分    | 4.2/5          | 3.8/5    | 4.5/5   |
| 检索Top-1准确率 | 78%            | 72%      | 85%     |
| 推理延迟        | 850ms          | 320ms    | 350ms   |
| 成本/1000次     | $2.50 | $0.80  | $0.85    |         |

**核心目标**: 8B模型成本降低70%，检索效果超越32B（85% vs 78%）

---

## 9️⃣ 预期训练曲线

```
Epoch 1-2:  avg_reward: -0.1 → 0.0,  8b_win_rate: 20% → 35%
Epoch 3-5:  avg_reward:  0.0 → 0.1,  8b_win_rate: 35% → 50%
Epoch 6-8:  avg_reward:  0.1 → 0.15, 8b_win_rate: 50% → 60%
Epoch 9-10: avg_reward:  0.15 → 0.2, 8b_win_rate: 60% → 65%
```

**最终目标**:

- 8B胜率 ≥ 60%
- 平均reward ≥ 0.15
- 平局率 ≤ 20%
- 双差率 ≤ 5%

---

## 🔟 常见问题与解决

### Q1: GPT-5评分不稳定？

- 设置temperature=0.1
- 使用response_format={"type": "json_object"}
- 添加重试机制（3次）
- 缓存评分结果

### Q2: Reward波动太大？

- Advantage归一化：`(adv - mean) / std`
- 使用moving average平滑
- 检查GPT-5评分合理性
- 调整better奖励系数

### Q3: 8B模型不学习？

- 检查learning_rate
- 调整clip_range (0.2 → 0.3)
- 增加batch_size (8 → 16)
- 检查Critic训练状态

### Q4: RAG API调用慢？

- 增加并发数：max_concurrent=10
- 启用结果缓存
- 减少batch_size
- 优化RAG响应时间

---

## 🚀 快速开始

### 1. 准备SFT训练数据

```bash
# 批量测试RAG
python batch_test_rag.py --tenant fivedoctors

# 转换为训练格式
python convert_to_sft_format.py \
    --input test_sft_fivedoctors.xlsx \
    --output data/sft/fivedoctors/ \
    --quality_threshold 0.6
```

### 2. SFT训练

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset data/sft/fivedoctors/train_latest.jsonl \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir outputs/sft/fivedoctors
```

### 3. RL训练

```bash
# 启动RAG服务
python startup.py -a

# 启动RL训练（新终端）
python train_with_verl.py \
    --config verl_config.yaml \
    --tenant fivedoctors \
    --num_epochs 10
```

### 4. 监控训练

```bash
wandb login
# 访问 https://wandb.ai 查看训练曲线
```
