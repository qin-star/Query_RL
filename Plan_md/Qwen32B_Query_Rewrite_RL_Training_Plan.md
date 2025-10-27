# Qwen-32B Query改写RL训练方案

> 基于GPT-5知识蒸馏 + RL竞争优化，训练高性能的Qwen-32B Query改写模型
>
> （本方案是基于8B模型调通的基础上的升级版本）

---

## 📋 方案概述

### 核心思路

```
阶段1: SFT知识蒸馏
GPT-5/DeepSeek V3.1 (Teacher) → 生成改写数据 → Qwen-32B (Student) SFT训练

阶段2: RL竞争优化  
Qwen-32B (训练中) ↔ GPT-5/DeepSeek V3.1 (Baseline) + 实时RAG检索 → PPO优化
```

### 为什么选择GPT-5或其他更高级的模型作为Teacher？

**相比使用现有的Qwen-32B作为Teacher：**

| 对比项   | 现有Qwen-32B    | GPT-5/DeepSeek V3.1     |
| -------- | --------------- | ----------------------- |
| 改写质量 | 较好，但有局限  | 顶尖水平                |
| 领域适配 | 需要prompt调优  | 少样本学习能力强        |
| 天花板   | 只能学到32B水平 | 可以逼近甚至超越Teacher |
| 成本     | API调用成本低   | API调用成本较高         |
| 训练效果 | 提升有限        | 显著提升                |

**推荐方案**：使用**GPT-5**作为Teacher（DeepSeek V3.1作为备选）

### 技术栈

- **基座模型**: Qwen3-32B-Instruct
- **教师模型**: GPT-5 (API调用)
- **评分模型**: GPT-5 (API调用，用于RL奖励计算)
- **RL算法**: PPO (Proximal Policy Optimization)
- **训练框架**: ms-swift + VERL

---

## 1️⃣ 数据准备（基于测试集）

### 1.1 核心流程

先换成高阶模型进行Query改写和用户画像等信息的提前，构建datasheet用于Qwen-32B的SFT

```
测试集Excel → 批量调用GPT-5改写 → 保存结果到test_sft.xlsx → 
质量筛选(基于检索效果) → 转换为JSONL训练格式 → SFT训练
```

### 1.2 批量调用GPT-5生成训练数据

```python
# generate_training_data.py

import asyncio
import pandas as pd
from openai import AsyncOpenAI
from typing import List, Dict
import json

class GPT5DataGenerator:
    """使用GPT-5生成Query改写训练数据"""
  
    def __init__(
        self,
        api_key: str,
        tenant_id: str = "fivedoctors",
        model: str = "gpt-5"
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.tenant_id = tenant_id
        self.model = model
  
        # 领域专用的系统prompt
        self.system_prompts = {
            "fivedoctors": """你是保健品领域的专业Query改写专家。
改写要求：
1. 提取核心产品关键词（胶原蛋白肽、虾青素等）
2. 明确查询意图（功效、用法、禁忌、适用人群等）
3. 补充专业术语，提升检索精度
4. 保持简洁，避免冗余
5. 完整保留用户原始意图""",

            "chengla": """你是教育培训领域的专业Query改写专家。
改写要求：
1. 识别课程类型和科目
2. 明确学习阶段和需求
3. 提取关键知识点
4. 保持教育领域专业性"""
        }
  
    async def rewrite_query(
        self,
        query: str,
        user_profile: str = "",
        history_summary: str = ""
    ) -> str:
        """调用GPT-5改写单个query"""
  
        # 构建用户prompt
        user_content = f"原始查询: {query}"
  
        if user_profile:
            user_content += f"\n用户画像: {user_profile}"
  
        if history_summary:
            user_content += f"\n历史摘要: {history_summary}"
  
        user_content += "\n\n请改写这个查询，使其更适合知识库检索。只返回改写后的查询，不要其他内容。"
  
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompts.get(self.tenant_id, "")
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                temperature=0.3,  # 较低温度保证稳定性
                max_tokens=200
            )
      
            return response.choices[0].message.content.strip()
  
        except Exception as e:
            print(f"GPT-5调用失败: {e}")
            return query  # 失败时返回原query
  
    async def batch_generate_from_excel(
        self,
        excel_path: str,
        output_path: str = "data/gpt5_rewrites.xlsx",
        query_column: str = "问题",
        max_concurrent: int = 10
    ) -> pd.DataFrame:
        """从Excel批量生成改写数据"""
  
        # 读取测试集
        df = pd.read_excel(excel_path)
        print(f"📚 读取测试集: {len(df)} 条")
  
        # 准备任务
        semaphore = asyncio.Semaphore(max_concurrent)
  
        async def rewrite_with_limit(row):
            async with semaphore:
                query = row[query_column]
                user_profile = row.get("用户画像", "")
                history_summary = row.get("历史摘要", "")
          
                rewritten = await self.rewrite_query(
                    query, user_profile, history_summary
                )
          
                return {
                    "original_query": query,
                    "rewritten_query": rewritten,
                    "user_profile": user_profile,
                    "history_summary": history_summary
                }
  
        # 批量执行
        print("🚀 开始批量生成...")
        tasks = [rewrite_with_limit(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
  
        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_excel(output_path, index=False)
  
        print(f"✅ 生成完成: {output_path}")
        print(f"   成功生成: {len(results)} 条")
  
        return result_df


# 使用示例
async def main():
    generator = GPT5DataGenerator(
        api_key="your-api-key",
        tenant_id="fivedoctors"
    )
  
    # 批量生成
    result_df = await generator.batch_generate_from_excel(
        excel_path="sales-rag/Test-jq-only/Test_data/女博士测试集.xlsx",
        output_path="data/gpt5_rewrites_fivedoctors.xlsx"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 1.3 调用RAG验证检索效果

生成改写后，需要验证检索效果，筛选高质量样本：

```python
# validate_with_rag.py

import asyncio
import pandas as pd
import httpx
from typing import List, Dict

async def validate_rewrites_with_rag(
    input_excel: str,
    output_excel: str,
    rag_api_url: str = "http://localhost:8000/api/chat/general_rag",
    tenant_id: str = "fivedoctors"
):
    """调用RAG验证改写效果"""
  
    df = pd.read_excel(input_excel)
    results = []
  
    async with httpx.AsyncClient(timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = {
                "query": row["original_query"],
                "rewritten_query": row["rewritten_query"],
                "tenant_id": tenant_id,
                "kb_name": "default",
                "top_k": 5,
                "score_threshold": 0.5
            }
      
            try:
                response = await client.post(rag_api_url, json=payload)
                result = response.json()
          
                data = result.get("data", {})
                recall = data.get("recall", [])
          
                # 计算检索质量指标
                top1_score = recall[0].get("reranker_score", 0) if recall else 0
                recall_count = len(recall)
          
                results.append({
                    **row.to_dict(),
                    "recall_count": recall_count,
                    "top1_score": top1_score,
                    "success": True
                })
      
            except Exception as e:
                print(f"❌ 检索失败: {row['original_query']} - {e}")
                results.append({
                    **row.to_dict(),
                    "success": False
                })
  
    # 保存验证结果
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_excel, index=False)
  
    # 统计
    success_df = result_df[result_df["success"] == True]
    print(f"\n✅ 验证完成:")
    print(f"   成功率: {len(success_df)}/{len(df)} = {len(success_df)/len(df)*100:.1f}%")
    print(f"   平均Top1分数: {success_df['top1_score'].mean():.3f}")
  
    return result_df
```

### 1.4 转换为训练格式

```python
# convert_to_training_format.py

import pandas as pd
import json
from pathlib import Path

def convert_to_sft_format(
    excel_path: str,
    output_dir: str,
    quality_threshold: float = 0.6
):
    """转换为SFT训练格式"""
  
    df = pd.read_excel(excel_path)
  
    # 质量筛选
    df = df[
        (df["success"] == True) &
        (df["top1_score"] >= quality_threshold) &
        (df["rewritten_query"].notna())
    ]
  
    print(f"✅ 质量筛选: {len(df)} 条样本 (threshold={quality_threshold})")
  
    # 转换为对话格式
    samples = []
    for _, row in df.iterrows():
        messages = [
            {
                "role": "system",
                "content": "你是保健品领域的Query改写专家..."
            },
            {
                "role": "user",
                "content": f"原始查询: {row['original_query']}\n\n请改写。"
            },
            {
                "role": "assistant",
                "content": row["rewritten_query"]
            }
        ]
  
        samples.append({
            "messages": messages,
            "metadata": {
                "top1_score": float(row["top1_score"]),
                "source": "gpt5_teacher"
            }
        })
  
    # 划分数据集
    import random
    random.shuffle(samples)
  
    n = len(samples)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
  
    splits = {
        "train": samples[:train_size],
        "val": samples[train_size:train_size + val_size],
        "test": samples[train_size + val_size:]
    }
  
    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
  
    for split_name, split_data in splits.items():
        file_path = output_path / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in split_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"   {split_name}: {len(split_data)} 条 → {file_path}")
```

### 1.5 快速执行流程

```bash
# 步骤1: 使用GPT-5批量生成改写（约30分钟，视API速度）
python generate_training_data.py

# 步骤2: 调用RAG验证检索效果（约20分钟）
python validate_with_rag.py

# 步骤3: 转换为训练格式（1分钟）
python convert_to_training_format.py

# 数据准备完成！进入SFT训练阶段
```

**预期数据规模**：

- 原始测试集: 500-1000条
- GPT-5生成成功: 480-980条
- 检索验证通过: 400-900条
- 质量筛选后(top1_score>0.6): 300-700条
- 最终训练集: 240-560条

---

## 2️⃣ SFT训练（基于ms-swift）

### 2.1 为什么使用ms-swift？

- ✅ 支持Qwen系列开箱即用
- ✅ 自动配置LoRA、DeepSpeed
- ✅ 训练简单，一条命令搞定
- ✅ 支持多GPU分布式训练

### 2.2 训练脚本

```bash
# train_sft_qwen32b.sh

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4张GPU
export NPROC_PER_NODE=4

# 模型和数据路径
MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
DATASET_PATH="data/sft_fivedoctors/train.jsonl"
OUTPUT_DIR="output/sft_qwen32b_fivedoctors"

# LoRA训练（推荐，显存占用低）
swift sft \
    --model ${MODEL_PATH} \
    --train_type lora \
    --dataset ${DATASET_PATH} \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4

echo "✅ SFT训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
```

### 2.3 资源需求

**Qwen-32B LoRA训练**：

- GPU: 4x A100 (80GB) 或 4x H100
- 显存: 每卡约50-60GB
- 训练时间: 2-3天（300-500条样本，3 epochs）

**全量微调**（不推荐）：

- GPU: 8x A100 (80GB)
- 显存: 每卡约70GB
- 训练时间: 4-5天

---

## 3️⃣ RL训练（Qwen-32B vs GPT-5）

### 3.1 核心思路

```
每个训练步骤：
1. 输入原始query
2. Qwen-32B生成改写 → 实时调用RAG检索 → 获得检索结果A
3. GPT-5生成改写 → 实时调用RAG检索 → 获得检索结果B
4. 比较A和B：
   - GPT-5评分：评估改写质量
   - 检索效果：比较reranker_score
5. 计算奖励：
   - 如果32B检索效果 > GPT-5：正奖励（鼓励）
   - 如果32B检索效果 < GPT-5：负奖励（惩罚）
6. PPO更新Qwen-32B参数
```

### 3.2 GPT-5评分模型

```python
# gpt5_scorer.py（与之前相同）

from openai import OpenAI

class GPT5Scorer:
    """GPT-5评分器"""
  
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
  
    def score_rewrite(
        self,
        original: str,
        rewritten: str,
        context: dict = None
    ) -> dict:
        """评分改写质量"""
  
        # （详细实现见前文）
        # 返回：{"综合得分": 4.5, "评分理由": "..."}
        ...
```

### 3.3 Reward函数设计

```python
# reward_function.py

import numpy as np
from gpt5_scorer import GPT5Scorer

class RewardFunction:
    """多维度奖励函数"""
  
    def __init__(self, gpt5_api_key: str):
        self.gpt5_scorer = GPT5Scorer(api_key=gpt5_api_key)
  
        # 奖励权重
        self.weights = {
            "gpt5_score": 0.35,         # GPT-5改写质量评分
            "retrieval_quality": 0.40,  # 检索效果（reranker_score）
            "relative_win": 0.25        # 相对GPT-5的胜率
        }
  
    def compute_reward(
        self,
        original_query: str,
        qwen32b_rewrite: str,
        gpt5_rewrite: str,
        context: dict,
        qwen32b_retrieval: list,
        gpt5_retrieval: list
    ) -> float:
        """计算综合奖励"""
  
        # 1. GPT-5评分奖励
        gpt5_score_32b = self.gpt5_scorer.score_rewrite(
            original_query, qwen32b_rewrite, context
        )["综合得分"] / 5.0  # 归一化到[0,1]
  
        gpt5_score_gpt5 = self.gpt5_scorer.score_rewrite(
            original_query, gpt5_rewrite, context
        )["综合得分"] / 5.0
  
        score_diff = gpt5_score_32b - gpt5_score_gpt5
        score_reward = np.tanh(score_diff * 3)  # 缩放到[-1,1]
  
        # 2. 检索质量奖励
        retrieval_32b = self._calc_retrieval_quality(qwen32b_retrieval)
        retrieval_gpt5 = self._calc_retrieval_quality(gpt5_retrieval)
  
        retrieval_diff = retrieval_32b - retrieval_gpt5
        retrieval_reward = np.tanh(retrieval_diff * 3)
  
        # 3. 胜率奖励（是否超越GPT-5）
        if retrieval_32b > retrieval_gpt5:
            win_reward = 0.5  # 32B赢了
        elif retrieval_32b < retrieval_gpt5:
            win_reward = -0.3  # 32B输了
        else:
            win_reward = 0.0  # 平局
  
        # 加权求和
        total_reward = (
            self.weights["gpt5_score"] * score_reward +
            self.weights["retrieval_quality"] * retrieval_reward +
            self.weights["relative_win"] * win_reward
        )
  
        return np.clip(total_reward, -1.0, 1.0)
  
    def _calc_retrieval_quality(self, results: list) -> float:
        """计算检索质量"""
        if not results:
            return 0.0
  
        # Top1分数
        top1 = results[0].get("reranker_score", 0) if results else 0
  
        # Top3平均分数
        top3_scores = [r.get("reranker_score", 0) for r in results[:3]]
        avg_top3 = np.mean(top3_scores) if top3_scores else 0
  
        # 综合质量分数
        return 0.6 * top1 + 0.4 * avg_top3
```

### 3.4 RL训练主流程

```python
# rl_trainer.py

import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import httpx
from reward_function import RewardFunction

class Qwen32BRLTrainer:
    """Qwen-32B RL训练器"""
  
    def __init__(
        self,
        qwen32b_sft_path: str,  # SFT训练后的模型
        gpt5_api_key: str,
        rag_api_url: str = "http://localhost:8000/api/chat/general_rag",
        tenant_id: str = "fivedoctors"
    ):
        self.tenant_id = tenant_id
        self.rag_api_url = rag_api_url
  
        # 加载Qwen-32B模型（Policy Model）
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            qwen32b_sft_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen32b_sft_path)
  
        # GPT-5 API（用于生成baseline改写和评分）
        from openai import AsyncOpenAI
        self.gpt5_client = AsyncOpenAI(api_key=gpt5_api_key)
  
        # 奖励函数
        self.reward_fn = RewardFunction(gpt5_api_key=gpt5_api_key)
  
    async def generate_training_episode(
        self,
        original_query: str,
        context: dict
    ) -> dict:
        """生成一个训练episode"""
  
        # 1. Qwen-32B生成改写
        qwen32b_rewrite = await self._generate_qwen32b(original_query, context)
  
        # 2. GPT-5生成改写（baseline）
        gpt5_rewrite = await self._generate_gpt5(original_query, context)
  
        # 3. 并行调用RAG检索
        qwen32b_results, gpt5_results = await asyncio.gather(
            self._call_rag(qwen32b_rewrite, original_query),
            self._call_rag(gpt5_rewrite, original_query)
        )
  
        # 4. 计算奖励
        reward = self.reward_fn.compute_reward(
            original_query=original_query,
            qwen32b_rewrite=qwen32b_rewrite,
            gpt5_rewrite=gpt5_rewrite,
            context=context,
            qwen32b_retrieval=qwen32b_results,
            gpt5_retrieval=gpt5_results
        )
  
        return {
            "original_query": original_query,
            "qwen32b_rewrite": qwen32b_rewrite,
            "gpt5_rewrite": gpt5_rewrite,
            "reward": reward,
            "qwen32b_top1": qwen32b_results[0].get("reranker_score", 0) if qwen32b_results else 0,
            "gpt5_top1": gpt5_results[0].get("reranker_score", 0) if gpt5_results else 0
        }
  
    async def _generate_qwen32b(self, query: str, context: dict) -> str:
        """Qwen-32B生成改写"""
  
        prompt = f"原始查询: {query}\n\n请改写。"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.policy_model.device)
  
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
  
        rewrite = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
  
        return rewrite.strip()
  
    async def _generate_gpt5(self, query: str, context: dict) -> str:
        """GPT-5生成改写（baseline）"""
  
        response = await self.gpt5_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "你是Query改写专家..."},
                {"role": "user", "content": f"原始查询: {query}\n\n请改写。"}
            ],
            temperature=0.3,
            max_tokens=150
        )
  
        return response.choices[0].message.content.strip()
  
    async def _call_rag(self, rewritten_query: str, original_query: str) -> list:
        """调用RAG API获取检索结果"""
  
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.rag_api_url,
                json={
                    "query": original_query,
                    "rewritten_query": rewritten_query,
                    "tenant_id": self.tenant_id,
                    "kb_name": "default",
                    "top_k": 5
                }
            )
      
            result = response.json()
            return result.get("data", {}).get("recall", [])
```

### 3.5 RAG API修改

需要在 `general_rag_routes.py`添加参数支持外部传入改写：

```python
# sales-rag/libs/chatchat-server/chatchat/server/api_server/general_rag_routes.py

@router.post("/api/chat/general_rag")
async def general_rag_endpoint(
    query: str,
    tenant_id: str,
    rewritten_query: Optional[str] = None,  # 🆕 新增参数
    ...
):
    # 如果提供了改写query，直接使用
    if rewritten_query:
        new_query = rewritten_query
    else:
        # 使用原有改写逻辑
        new_query = await rewrite_query_by_model(...)
  
    # 后续检索流程不变
    ...
```

### 3.6 从Reward到参数更新：PPO算法详解

这是RL训练的核心！让我详细解释GPT-5评分生成的reward如何更新Qwen-32B的参数。

#### 3.6.1 PPO算法原理

**基本流程**：

```
1. 收集轨迹(Trajectory)
   - 当前Qwen-32B生成改写
   - 调用RAG获取检索结果
   - GPT-5评分 → 计算reward

2. 计算优势函数(Advantage)
   - 估计状态价值 V(s)
   - 计算 Advantage = Reward - V(s)

3. 策略梯度更新
   - 计算策略比率 ratio = π_new / π_old
   - 计算 PPO loss（带clip）
   - 反向传播更新参数

4. 价值函数更新
   - 更新 V(s) 使其更准确估计未来回报
```

#### 3.6.2 详细数学推导

**Step 1: 收集经验**

对于每个query，我们收集一个完整的trajectory：

```python
trajectory = {
    "state": original_query,              # 状态（原始query）
    "action": qwen32b_rewrite,           # 动作（32B生成的改写）
    "reward": reward_from_gpt5_and_rag,  # 奖励（GPT-5评分+检索效果）
    "log_prob": log_prob_of_action,      # 当前策略下动作的对数概率
}
```

**Step 2: 计算优势函数(Advantage)**

优势函数告诉我们：**这个动作比平均水平好多少**

```python
# 价值函数估计：这个状态下期望的累积回报
V(state) = critic_model(state)  # 使用critic网络估计

# 优势函数：实际reward - 期望reward
Advantage = Reward - V(state)

# 如果 Advantage > 0：这个动作比期望好 → 增加这个动作的概率
# 如果 Advantage < 0：这个动作比期望差 → 降低这个动作的概率
```

**Step 3: 计算策略比率**

PPO的核心：比较新旧策略

```python
# 旧策略：当前的Qwen-32B
log_prob_old = log P_old(qwen32b_rewrite | original_query)

# 新策略：更新一步后的Qwen-32B  
log_prob_new = log P_new(qwen32b_rewrite | original_query)

# 策略比率
ratio = exp(log_prob_new - log_prob_old) = P_new / P_old
```

**Step 4: PPO损失函数**

```python
# 基础策略梯度
surrogate_loss = ratio * Advantage

# PPO clip：防止更新太激进
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε通常为0.2
clipped_loss = clipped_ratio * Advantage

# 最终loss：取两者最小值（保守更新）
policy_loss = -min(surrogate_loss, clipped_loss)

# 为什么是负号？因为我们要最大化reward，但优化器是minimize loss
```

**Step 5: 价值函数损失**

```python
# Critic网络要准确预测回报
value_loss = (Reward - V(state))^2
```

**Step 6: 总损失**

```python
total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

# entropy_bonus: 鼓励探索，避免策略过早收敛
```

#### 3.6.3 参数更新流程

```python
# 伪代码：完整的PPO更新步骤

class PPOTrainer:
    def __init__(self):
        self.actor = Qwen32B_Model()      # 策略网络（生成改写）
        self.critic = Value_Network()     # 价值网络（估计V(s)）
        self.optimizer_actor = Adam(self.actor.parameters(), lr=1e-5)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=1e-4)
  
    def update(self, trajectories):
        """使用收集的trajectories更新模型"""
    
        # 1. 提取数据
        states = [t["state"] for t in trajectories]
        actions = [t["action"] for t in trajectories]
        rewards = [t["reward"] for t in trajectories]
        old_log_probs = [t["log_prob"] for t in trajectories]
    
        # 2. 计算优势函数
        with torch.no_grad():
            values = self.critic(states)  # V(s)
            advantages = rewards - values  # Advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 归一化
    
        # 3. PPO更新（多个epoch）
        for epoch in range(4):  # PPO通常更新4次
            # 3.1 前向传播
            new_log_probs = self.actor.get_log_prob(states, actions)
            new_values = self.critic(states)
        
            # 3.2 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
        
            # 3.3 计算PPO loss
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
            # 3.4 计算value loss
            value_loss = 0.5 * (rewards - new_values).pow(2).mean()
        
            # 3.5 计算entropy（鼓励探索）
            entropy = self.actor.get_entropy(states)
            entropy_loss = -0.01 * entropy.mean()
        
            # 3.6 总损失
            total_loss = policy_loss + value_loss + entropy_loss
        
            # 4. 反向传播
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
        
            # 5. 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        
            # 6. 更新参数
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item()
        }
```

#### 3.6.4 具体示例

假设我们有一个training episode：

```python
# 原始query
original_query = "胶原蛋白怎么吃"

# Qwen-32B生成改写
qwen32b_rewrite = "胶原蛋白肽 服用方法 推荐用量"

# GPT-5评分 + 检索效果 → reward
reward = 0.65  # 正奖励，说明32B表现不错

# 当前策略下，这个改写的log概率
log_prob_old = -2.3  # 对数概率（负数）

# --- PPO更新过程 ---

# 1. Critic估计状态价值
V_state = 0.5  # Critic认为这个query平均能得0.5的reward

# 2. 计算Advantage
Advantage = reward - V_state = 0.65 - 0.5 = 0.15  # 正数！比期望好

# 3. 更新后的策略
log_prob_new = -2.1  # 更新后，这个改写的概率增加了

# 4. 计算ratio
ratio = exp(-2.1 - (-2.3)) = exp(0.2) = 1.22

# 5. PPO loss
surrogate1 = 1.22 * 0.15 = 0.183
clipped_ratio = min(max(1.22, 0.8), 1.2) = 1.2
surrogate2 = 1.2 * 0.15 = 0.18
policy_loss = -min(0.183, 0.18) = -0.18  # 负数 → 梯度上升 → 增加概率

# 6. 反向传播更新参数
# 结果：下次遇到类似query，更可能生成类似的好改写
```

如果reward是负数（32B表现差）：

```python
reward = -0.3  # 负奖励
Advantage = -0.3 - 0.5 = -0.8  # 负数！比期望差

# PPO会降低这个动作的概率
# 下次遇到类似query，会尝试生成不同的改写
```

#### 3.6.5 VERL框架实现

在实际代码中，VERL框架帮我们处理了大部分细节：

```python
# 使用VERL的PPO Trainer

from verl.trainer.ppo import PPOTrainer
from verl.utils.reward_score import RewardFunction

# 1. 配置PPO参数
ppo_config = {
    "ppo_epochs": 4,           # 每批数据更新4次
    "clip_range": 0.2,         # clip范围 [0.8, 1.2]
    "value_loss_coef": 0.5,    # value loss权重
    "entropy_coef": 0.01,      # entropy权重
    "max_grad_norm": 0.5,      # 梯度裁剪阈值
    "learning_rate": 1e-5,     # 学习率
    "gamma": 0.99,             # 未来奖励折扣因子
    "gae_lambda": 0.95         # GAE优势估计参数
}

# 2. 创建trainer
trainer = PPOTrainer(
    actor_model=qwen32b_sft_model,      # 待训练的32B模型
    critic_model=None,                   # VERL自动创建critic
    reward_fn=MultiDimensionalReward(),  # 我们的reward函数
    config=ppo_config
)

# 3. 训练循环
for epoch in range(num_epochs):
    # 收集经验（批量生成改写+获取reward）
    trajectories = await collect_trajectories(
        num_samples=256,
        actor_model=trainer.actor
    )
  
    # PPO更新（VERL自动处理所有细节）
    metrics = trainer.update(trajectories)
  
    # 监控
    print(f"Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"Value Loss: {metrics['value_loss']:.4f}")
    print(f"Avg Reward: {metrics['avg_reward']:.4f}")
```

#### 3.6.6 关键超参数

| 参数                  | 值    | 说明                       |
| --------------------- | ----- | -------------------------- |
| learning_rate         | 1e-5  | Actor学习率（小心调整！）  |
| critic_learning_rate  | 1e-4  | Critic学习率（通常>actor） |
| ppo_epochs            | 4     | 每批数据更新次数           |
| clip_range            | 0.2   | PPO clip范围               |
| batch_size            | 32-64 | 每批样本数                 |
| gradient_accumulation | 4-8   | 梯度累积步数               |
| max_grad_norm         | 0.5   | 梯度裁剪阈值               |

#### 3.6.7 训练监控要点

```python
# 健康的训练应该看到：
wandb.log({
    "avg_reward": 0.3 → 0.5 → 0.65,      # 逐步提升
    "policy_loss": -0.2 → -0.15,         # 逐渐减小（绝对值）
    "value_loss": 0.5 → 0.3 → 0.15,      # 逐渐减小
    "qwen32b_win_rate": 0.3 → 0.5 → 0.7,  # 胜率提升
    "clip_fraction": 0.1-0.3              # 10-30%的样本被clip（正常）
})

# ⚠️ 异常情况：
# - reward下降：可能学习率太大
# - clip_fraction > 0.5：更新太激进，降低学习率
# - value_loss不降：Critic训练有问题
```

### 3.7 训练启动

```bash
# 1. 启动RAG服务（一个终端）
cd sales-rag
python startup.py -a

# 2. 启动RL训练（另一个终端）
cd code
python -m verl.trainer.main_ppo \
    --config config/qwen32b_rl_config.yaml \
    --model_path output/sft_qwen32b_fivedoctors \
    --rag_api_url http://localhost:8000/api/chat/general_rag \
    --gpt5_api_key sk-xxx \
    --max_concurrent_rag_calls 10 \
    --num_epochs 10

# 训练监控
wandb login
# 在 https://wandb.ai 查看训练曲线
```

---

## 4️⃣ 关键优化策略

### 4.1 成本控制

**GPT-5 API调用成本优化**：

1. **缓存机制**：相同query缓存改写结果
2. **采样策略**：不是每个样本都调用GPT-5评分（采样30%）
3. **批量调用**：使用batch API（如果支持）

**预期成本**：

- SFT数据生成：500条 × $0.01 = $5
- RL训练（10 epochs，1000条）：30%采样 = 3000次调用 × $0.01 = $30
- **总计**：约$35-50

### 4.2 训练效率

**并发控制**：

- RAG API并发：10-20个请求
- GPT-5 API并发：5-10个请求（避免限流）

**缓存策略**：

- 检索结果缓存：命中率30-40%
- GPT-5改写缓存：命中率50-60%（baseline稳定）

### 4.3 监控指标

```python
# WandB监控
wandb.log({
    # 核心指标
    "avg_reward": avg_reward,
    "qwen32b_win_rate": wins / total,  # 32B胜率
    "qwen32b_avg_top1": avg_32b_top1,
    "gpt5_avg_top1": avg_gpt5_top1,
  
    # API调用
    "rag_api_calls": total_calls,
    "gpt5_api_calls": gpt5_calls,
    "cache_hit_rate": cache_hits / total_calls,
  
    # 训练进度
    "policy_loss": policy_loss,
    "value_loss": value_loss
})
```

---

## 5️⃣ 完整训练流程

```bash
# 总时间：约7-10天

# 步骤1: 数据准备（1天）
python generate_training_data.py      # GPT-5生成改写（30分钟）
python validate_with_rag.py          # RAG验证（20分钟）
python convert_to_training_format.py # 格式转换（1分钟）

# 步骤2: SFT训练（3-4天）
bash train_sft_qwen32b.sh

# 步骤3: RL训练（3-5天）
# 终端1: 启动RAG
cd sales-rag && python startup.py -a

# 终端2: 启动RL训练
python rl_train.py

# 步骤4: 评估与部署（1天）
python evaluate_model.py
bash deploy_model.sh
```

---

## 6️⃣ 预期效果

| 指标            | GPT-5 (Teacher) | Qwen-32B (SFT后) | Qwen-32B (RL后) |
| --------------- | --------------- | ---------------- | --------------- |
| 改写质量评分    | 4.8/5           | 4.3/5            | 4.6/5           |
| 检索Top-1准确率 | 85%             | 78%              | 88%             |
| 推理延迟        | 2000ms          | 850ms            | 900ms           |
| 成本/1000次     | $25 | $2.50     | $2.80            |                 |

**核心目标**：

- ✅ 改写质量接近GPT-5（4.6 vs 4.8）
- ✅ 检索效果超越GPT-5（88% vs 85%）
- ✅ 成本降低90%（$2.80 vs $25）
- ✅ 延迟降低55%（900ms vs 2000ms）

---

## 7️⃣ 常见问题

**Q1: 为什么不用DeepSeek V3.1做Teacher？**

- A: GPT-5和DeepSeek V3.1都可以，建议先测试改写质量，选择更好的

**Q2: Qwen-32B能超越GPT-5吗？**

- A: 在特定领域（如保健品）通过RL训练，有可能在检索效果上超越GPT-5

**Q3: 4张A100够训练吗？**

- A: LoRA训练够用，全量微调需要8张A100

**Q4: RL训练会不会过拟合？**

- A: 通过验证集监控，设置early stopping，避免过拟合

**Q5: GPT-5 API成本太高怎么办？**

- A: 使用缓存、采样、或切换到DeepSeek V3.1（成本更低）

---

## 📚 参考资料

- **ms-swift文档**: https://swift.readthedocs.io/zh-cn/latest/
- **Qwen2.5最佳实践**: https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3最佳实践.html
- **VERL框架**: https://github.com/volcengine/verl
- **PPO算法**: https://arxiv.org/abs/1707.06347
