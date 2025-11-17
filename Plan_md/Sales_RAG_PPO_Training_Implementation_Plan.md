# Sales-RAG Query改写PPO训练实施方案

> 基于VERL框架的Qwen-8B模型强化学习训练方案

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO训练主流程 (VERL)                      │
├─────────────────────────────────────────────────────────────┤
│  1. 数据加载 (Excel → DataProto)                             │
│  2. Actor(8B) Rollout生成改写                                │
│  3. Reward计算：                                             │
│     ├─ 调用Qwen-32B生成baseline                             │
│     ├─ 并行RAG检索 (8B vs 32B)                              │
│     ├─ GPT-5对比评分 (4维度)                                │
│     └─ 转换为Reward [-1, 1]                                 │
│  4. PPO参数更新 (GAE + Policy Loss)                         │
│  5. 保存checkpoint & 验证                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心发现：复用现有Prompt

### 2.1 现有Prompt位置

**路径**: `sales-rag/libs/chatchat-server/chatchat/settings.py`

**Prompt配置**:
```python
jiaopei_general_rag: dict = {
    "query_rewrite_without_portrait": """
    ## 输出要求
    {
      "user_profile": "用户画像",
      "history_summary": "历史摘要", 
      "rewritten_query": "改写查询"
    }
    """
}
```

**调用函数**: `rewrite_query_by_model()` in `general_rag_utils.py`

### 2.2 结论

✅ **直接复用现有Prompt**，无需修改  
✅ 训练目标：让8B学习模仿32B的输出格式和质量

---

## 三、数据准备

### 3.1 Excel格式要求

| 历史对话 | 当前问题 |
|---------|---------|
| 完整对话历史 | 用户当前提问 |

### 3.2 数据预处理脚本

```python
# scripts/prepare_training_data.py
import pandas as pd
import asyncio
from general_rag_utils import rewrite_query_by_model

async def generate_training_data(excel_path, output_path):
    df = pd.read_excel(excel_path)
    training_data = []
    
    for idx, row in df.iterrows():
        history_chat = row['历史对话']
        original_query = row['当前问题']
        
        # 调用现有API生成训练标签
        user_profile, history_summary, rewritten_query = \
            await rewrite_query_by_model(
                query=original_query,
                history=history_chat,
                thought="",
                tenant_id="chengla"
            )
        
        training_data.append({
            "history_chat": history_chat,
            "original_query": original_query,
            "user_profile": user_profile,
            "history_summary": history_summary,
            "rewritten_query": rewritten_query,
            "data_source": "sales_rag_chenglao"
        })
    
    # 保存为JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

---

## 四、模块设计

### 4.1 目录结构

```
code/verl/
├── data/
│   └── sales_rag_dataset.py          # 数据加载
├── utils/
│   ├── apis/
│   │   └── sales_rag_api.py          # API封装
│   └── reward_score/
│       └── sales_rag.py              # Reward函数
├── trainer/
│   ├── config/
│   │   └── sales_rag_ppo.yaml        # 训练配置
│   └── main_ppo.py                   # 主训练脚本（修改）
```

### 4.2 核心模块

#### **1) GPT-5评分器**

```python
# code/verl/utils/reward_score/sales_rag.py

class GPT5QueryRewriteScorer:
    def __init__(self, api_key, model="gpt-4o", temperature=0.1):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.weights = [0.4, 0.2, 0.2, 0.2]  # 质量、相关性、完整性、检索
    
    async def score_comparative(
        self,
        history_chat: str,
        user_profile_32b: str,
        rewritten_query_32b: str,
        history_summary_32b: str,
        rag_recall_32b: List[Dict],
        user_profile_8b: str,
        rewritten_query_8b: str,
        history_summary_8b: str,
        rag_recall_8b: List[Dict]
    ) -> Dict:
        """
        返回: {
            "better": "32b" | "8b" | "same" | "both bad",
            "score": {
                "32b": {"scores": [8,7,6,7], "sum": 70},
                "8b": {"scores": [9,8,7,8], "sum": 80}
            },
            "reason": "详细分析",
            "brief": "简要说明"
        }
        """
        # 构建prompt（使用规划文档中的4维度评分）
        # 调用GPT-5
        # 解析JSON返回
```

#### **2) Reward计算器**

```python
class RewardCalculator:
    def compute_reward(self, gpt5_result: Dict) -> float:
        sum_8b = gpt5_result["score"]["8b"]["sum"]
        sum_32b = gpt5_result["score"]["32b"]["sum"]
        better = gpt5_result["better"]
        
        # 归一化分数差
        score_diff = (sum_8b - sum_32b) / 100.0
        base_reward = np.tanh(score_diff * 2)
        
        # better调整
        if better == "8b":
            reward = base_reward + 0.2
        elif better == "32b":
            reward = base_reward - 0.2
        elif better == "same":
            reward = base_reward * 0.5
        elif better == "both bad":
            reward = -0.5
        
        return np.clip(reward, -1.0, 1.0)
```

#### **3) VERL标准接口**

```python
def compute_score(
    solution_str: str,           # Actor生成的完整输出
    ground_truth: Dict,          # 包含history_chat等信息
    qwen32b_api,                 # 32B API
    sales_rag_api,               # RAG API
    gpt5_scorer                  # GPT-5评分器
) -> float:
    """
    VERL框架调用的标准Reward接口
    
    流程：
    1. 解析solution_str提取8B的三字段
    2. 调用32B生成baseline
    3. 并行RAG检索
    4. GPT-5对比评分
    5. 计算reward
    
    Returns: reward [-1, 1]
    """
    # 1. 解析8B输出
    output_8b = parse_json_to_dict(solution_str)
    
    # 2. 调用32B（复用现有API）
    user_profile_32b, history_summary_32b, rewritten_query_32b = \
        await rewrite_query_by_model(
            query=ground_truth["original_query"],
            history=ground_truth["history_chat"],
            thought="",
            tenant_id="chengla"
        )
    
    # 3. 并行RAG检索
    rag_recall_8b = await sales_rag_api.retrieve(output_8b["rewritten_query"])
    rag_recall_32b = await sales_rag_api.retrieve(rewritten_query_32b)
    
    # 4. GPT-5评分
    gpt5_result = await gpt5_scorer.score_comparative(
        history_chat=ground_truth["history_chat"],
        user_profile_32b=user_profile_32b,
        rewritten_query_32b=rewritten_query_32b,
        history_summary_32b=history_summary_32b,
        rag_recall_32b=rag_recall_32b,
        user_profile_8b=output_8b["user_profile"],
        rewritten_query_8b=output_8b["rewritten_query"],
        history_summary_8b=output_8b["history_summary"],
        rag_recall_8b=rag_recall_8b
    )
    
    # 5. 计算reward
    reward_calculator = RewardCalculator()
    return reward_calculator.compute_reward(gpt5_result)
```

#### **4) API封装**

```python
# code/verl/utils/apis/sales_rag_api.py

class SalesRAGAPI:
    """封装现有RAG API"""
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def retrieve(self, rewritten_query: str, tenant_id: str = "chengla") -> List[Dict]:
        """调用 /rag/chat 接口"""
        # 构建请求
        # 返回召回结果
```

---

## 五、main_ppo.py修改

### 5.1 添加reward函数选择

```python
# code/verl/trainer/main_ppo.py

def _select_rm_score_fn(data_source):
    # ... 现有代码 ...
    elif 'sales_rag' in data_source:
        from verl.utils.reward_score import sales_rag
        return sales_rag.compute_score
    else:
        raise NotImplementedError
```

### 5.2 初始化APIs

```python
class RewardManager():
    def __init__(self, tokenizer, num_examine, config):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        
        # 现有代码...
        
        # 初始化Sales-RAG APIs
        if hasattr(config, 'sales_rag'):
            from verl.utils.apis.sales_rag_api import SalesRAGAPI
            from verl.utils.reward_score.sales_rag import GPT5QueryRewriteScorer
            
            self.sales_rag_api = SalesRAGAPI(config.sales_rag.rag_api_url)
            self.gpt5_scorer = GPT5QueryRewriteScorer(config.sales_rag.gpt5_api_key)
```

### 5.3 调用Reward函数

```python
def __call__(self, data: DataProto):
    # ... 现有代码 ...
    
    for i in range(len(data)):
        # ...
        
        if 'sales_rag' in data_source:
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                qwen32b_api=None,  # 直接调用rewrite_query_by_model
                sales_rag_api=self.sales_rag_api,
                gpt5_scorer=self.gpt5_scorer
            )
        else:
            score = compute_score_fn(...)
```

---

## 六、训练配置

```yaml
# code/verl/trainer/config/sales_rag_ppo.yaml

data:
  train_files:
    - path: "RL_trainning_data/chenglao_training.jsonl"
      data_source: "sales_rag_chenglao"

actor_rollout_ref:
  model:
    path: "outputs/sft/Qwen-8B-sft"
    dtype: "bfloat16"

critic:
  model:
    path: "outputs/sft/Qwen-8B-sft"
    dtype: "bfloat16"

ppo:
  learning_rate: 1.0e-6
  clip_range: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  gamma: 0.99
  lambda_: 0.95
  ppo_epochs: 4
  batch_size: 8

trainer:
  total_epochs: 10
  save_steps: 500
  n_gpus_per_node: 1
  nnodes: 1

reward_model:
  enable: false  # 使用函数式Reward

sales_rag:
  rag_api_url: "http://localhost:8000"
  gpt5_api_key: "sk-xxx"
  tenant_id: "chengla"
```

---

## 七、PPO算法原理（简化）

### 7.1 数据流

```
Rollout阶段:
  保存 prompts, responses, log_probs, values
    ↓
Reward计算:
  decode(responses) → GPT-5评分 → reward
    ↓
PPO更新:
  GAE → advantages
  Policy Loss = -E[log_prob * advantage]
  loss.backward() → 参数更新
```

### 7.2 关键点

- **responses已保存在rollout_buffer**，无需重复传递
- **log_probs记录生成概率**，用于计算policy loss
- **梯度反向传播**自动计算每个参数的调整方向
- **VERL框架自动处理**所有复杂计算

---

## 八、训练启动

```bash
# 1. 数据准备
python scripts/prepare_training_data.py \
    --input RL_trainning_data/橙啦-query_RL_训练集.xlsx \
    --output RL_trainning_data/chenglao_training.jsonl

# 2. SFT训练（如已完成则跳过）
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset RL_trainning_data/chenglao_training.jsonl \
    --num_train_epochs 3 \
    --output_dir outputs/sft/Qwen-8B-sft

# 3. 启动RAG服务
cd sales-rag
python startup.py -a

# 4. 启动PPO训练
cd code/verl/trainer
python main_ppo.py --config config/sales_rag_ppo.yaml
```

---

## 九、预期效果

| 指标 | Baseline (32B) | SFT (8B) | RL (8B) |
|------|----------------|----------|---------|
| 改写质量评分 | 4.2/5 | 3.8/5 | 4.5/5 |
| 检索Top-1准确率 | 78% | 72% | 85% |
| 推理延迟 | 850ms | 320ms | 350ms |
| 成本/1000次 | $2.50 | $0.80 | $0.85 |

**目标**: 8B成本降低70%，效果超越32B

---

## 十、待确认事项

1. Excel列名确认：`历史对话`、`当前问题`
2. RAG API URL
3. GPT-5 API Key
4. SFT模型路径

---

## 附录：文件清单

### 新建文件
```
code/verl/data/sales_rag_dataset.py
code/verl/utils/apis/sales_rag_api.py
code/verl/utils/reward_score/sales_rag.py
code/verl/trainer/config/sales_rag_ppo.yaml
scripts/prepare_training_data.py
```

### 修改文件
```
code/verl/trainer/main_ppo.py
  - _select_rm_score_fn()
  - RewardManager.__init__()
  - RewardManager.__call__()
```

