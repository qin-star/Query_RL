# Sales-RAG Query改写RL训练方案

> 基于Qwen-8B的两阶段训练：SFT知识蒸馏 + RL竞争优化

---

## 📋 方案概述

### 核心思路

```
阶段1: SFT知识蒸馏
Qwen-32B (Teacher) → 改写数据 → Qwen-8B (Student) SFT训练

阶段2: RL竞争优化  
Qwen-8B ↔ Qwen-32B (双模型竞争) + DeepSeek评分 → PPO/GRPO优化
```

### 技术栈

- **基座模型**: Qwen2.5-8B-Instruct
- **教师模型**: Qwen-32B (现有部署)
- **评分模型**: DeepSeek-V2.5 (fine-tuned)
- **RL算法**: PPO (Proximal Policy Optimization)
- **训练框架**: VERL (参考DeepRetrieval)

---

## 1️⃣ 训练数据集设计

### 1.1 数据来源

#### 来源1: 线上真实Query日志

从sales-rag系统的日志中提取真实用户query和32B改写结果：

```python
# 数据提取脚本示例
import json
from datetime import datetime, timedelta
from chatchat.server.tools.reflux_logger import get_logs_by_timerange

def extract_query_rewrite_data(
    tenant_id: str,
    start_date: datetime,
    end_date: datetime,
    min_samples: int = 10000
):
    """
    从reflux日志中提取query改写数据
    
    Returns:
        [
            {
                "tenant_id": "fivedoctors",
                "original_query": "胶原蛋白怎么吃",
                "rewritten_query": "胶原蛋白肽 服用方法 推荐用量 适用人群",
                "user_profile": "25-35岁女性，关注抗衰老...",
                "history_summary": "咨询过多次美容产品...",
                "history_context": "用户: 我想了解保健品\n助手: ...",
                "timestamp": "2025-01-15 10:23:45",
                "call_name": "rewrite_query_by_model"
            }
        ]
    """
    
    logs = get_logs_by_timerange(
        start_date=start_date,
        end_date=end_date,
        call_name="rewrite_query_by_model",
        tenant_id=tenant_id
    )
    
    dataset = []
    for log in logs:
        payload = log.get("payload", {})
        response = log.get("response", {})
        
        # 提取关键信息
        item = {
            "tenant_id": tenant_id,
            "original_query": payload.get("query", ""),
            "rewritten_query": response.get("rewritten_query", ""),
            "user_profile": response.get("user_profile", ""),
            "history_summary": response.get("history_summary", ""),
            "history_context": payload.get("history", ""),
            "thought_unit": payload.get("thought", ""),
            "timestamp": log.get("timestamp", "")
        }
        
        # 数据质量过滤
        if (item["original_query"] and 
            item["rewritten_query"] and 
            item["original_query"] != item["rewritten_query"]):
            dataset.append(item)
    
    return dataset[:min_samples]
```

#### 来源2: 测试集标注数据

从现有测试数据集中提取高质量样本：

```python
def extract_from_test_data(tenant_id: str):
    """
    从测试集中提取标注数据
    
    测试集路径:
    - sales-rag/Test-jq-only/Test_data/女博士测试集.xlsx
    - sales-rag/Test-jq-only/Test_data/橙啦合并测试集.xlsx
    """
    
    import pandas as pd
    
    # fivedoctors数据
    if tenant_id == "fivedoctors":
        df = pd.read_excel("Test-jq-only/Test_data/女博士测试集.xlsx")
        
        dataset = []
        for _, row in df.iterrows():
            # 假设测试集有"问题"和"期望改写"列
            if pd.notna(row.get("问题")) and pd.notna(row.get("期望改写")):
                dataset.append({
                    "tenant_id": "fivedoctors",
                    "original_query": row["问题"],
                    "rewritten_query": row["期望改写"],
                    "user_profile": row.get("用户画像", ""),
                    "history_summary": row.get("历史摘要", ""),
                    "source": "test_set"
                })
        
        return dataset
    
    # chengla数据
    elif tenant_id == "chengla":
        df = pd.read_excel("Test-jq-only/Test_data/橙啦合并测试集.xlsx")
        # 类似处理...
        
    return []
```

#### 来源3: 人工标注高质量数据

针对关键场景人工标注：

```python
# 人工标注模板
annotation_template = {
    "tenant_id": "fivedoctors",
    "original_query": "早上还是晚上喝好",
    "rewritten_query": "胶原蛋白肽 最佳服用时间 早晨空腹 vs 睡前服用",
    "annotation": {
        "改写质量": 5,  # 1-5分
        "领域适配": 5,
        "意图保持": 4,
        "可检索性": 5
    },
    "annotator": "expert_1",
    "timestamp": "2025-01-20 14:30:00",
    "source": "manual_annotation"
}
```

### 1.2 数据结构定义

#### SFT训练数据格式

```json
{
  "dataset_name": "sales_rag_query_rewrite_sft",
  "version": "1.0.0",
  "tenant_id": "fivedoctors",
  "created_at": "2025-01-20T10:00:00Z",
  "total_samples": 15000,
  
  "samples": [
    {
      "sample_id": "fivedr_001",
      "original_query": "胶原蛋白怎么吃",
      "rewritten_query": "胶原蛋白肽 服用方法 推荐用量 适用人群",
      
      "context": {
        "user_profile": "25-35岁女性，关注抗衰老和皮肤健康",
        "history_summary": "近期咨询过多次胶原蛋白产品，关心效果和使用方法",
        "history_context": "用户: 我想了解保健品\n助手: 好的，我来为您介绍...\n用户: 胶原蛋白怎么吃",
        "thought_unit": "用户可能想了解具体的服用指南"
      },
      
      "metadata": {
        "source": "production_log",
        "timestamp": "2025-01-15T10:23:45Z",
        "model": "Qwen-32B",
        "quality_score": 4.5
      }
    }
  ]
}
```

#### RL训练数据格式

```json
{
  "dataset_name": "sales_rag_query_rewrite_rl",
  "version": "1.0.0",
  "tenant_id": "fivedoctors",
  
  "samples": [
    {
      "sample_id": "fivedr_rl_001",
      "original_query": "孕妇能喝吗",
      "context": {
        "user_profile": "备孕期女性，28岁",
        "history_summary": "咨询过胶原蛋白产品",
        "history_context": "用户: 胶原蛋白效果怎么样\n助手: ...\n用户: 孕妇能喝吗"
      },
      
      "candidates": {
        "qwen_8b_rewrite": "胶原蛋白肽 孕妇禁忌 孕期服用安全性",
        "qwen_32b_rewrite": "孕妇 备孕期 胶原蛋白肽 服用禁忌 注意事项"
      },
      
      "retrieval_results": {
        "qwen_8b_results": [
          {
            "content": "孕妇及备孕期女性不建议服用胶原蛋白肽...",
            "score": 0.87,
            "reranker_score": 0.92
          }
        ],
        "qwen_32b_results": [
          {
            "content": "备孕期间建议停用胶原蛋白肽补充剂...",
            "score": 0.91,
            "reranker_score": 0.95
          }
        ]
      },
      
      "evaluation": {
        "deepseek_scores": {
          "qwen_8b": {
            "改写质量": 4.2,
            "领域适配": 4.0,
            "意图保持": 4.5,
            "可检索性": 4.3,
            "综合得分": 4.25
          },
          "qwen_32b": {
            "改写质量": 4.7,
            "领域适配": 4.8,
            "意图保持": 4.9,
            "可检索性": 4.6,
            "综合得分": 4.75
          }
        },
        "retrieval_metrics": {
          "qwen_8b": {
            "top1_score": 0.87,
            "avg_top3_score": 0.84,
            "recall_at_3": 1.0
          },
          "qwen_32b": {
            "top1_score": 0.91,
            "avg_top3_score": 0.89,
            "recall_at_3": 1.0
          }
        },
        "winner": "qwen_32b",
        "margin": 0.50
      },
      
      "reward": 0.15  # 基于winner margin计算
    }
  ]
}
```

### 1.3 数据收集脚本

完整的数据收集pipeline：

```python
# data_collection_pipeline.py

import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import List, Dict

class QueryRewriteDataCollector:
    """Query改写数据收集器"""
    
    def __init__(self, output_dir: str = "data/query_rewrite_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def collect_all_data(
        self,
        tenant_ids: List[str] = ["fivedoctors", "chengla", "zlkt"],
        days_back: int = 30
    ):
        """收集所有训练数据"""
        
        all_data = {}
        
        for tenant_id in tenant_ids:
            print(f"收集 {tenant_id} 的数据...")
            
            # 1. 收集线上日志数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            log_data = extract_query_rewrite_data(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date,
                min_samples=10000
            )
            
            # 2. 收集测试集数据
            test_data = extract_from_test_data(tenant_id)
            
            # 3. 合并数据
            combined_data = log_data + test_data
            
            # 4. 数据清洗和质量评估
            cleaned_data = self.clean_data(combined_data)
            
            # 5. 数据增强
            augmented_data = self.augment_data(cleaned_data, tenant_id)
            
            all_data[tenant_id] = augmented_data
            
            print(f"  - 日志数据: {len(log_data)} 条")
            print(f"  - 测试数据: {len(test_data)} 条")
            print(f"  - 清洗后: {len(cleaned_data)} 条")
            print(f"  - 增强后: {len(augmented_data)} 条")
        
        # 6. 保存数据
        self.save_datasets(all_data)
        
        return all_data
    
    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """数据清洗"""
        
        cleaned = []
        
        for item in data:
            # 1. 去重
            if self._is_duplicate(item, cleaned):
                continue
            
            # 2. 质量检查
            if not self._quality_check(item):
                continue
            
            # 3. 规范化
            normalized_item = self._normalize(item)
            
            cleaned.append(normalized_item)
        
        return cleaned
    
    def _quality_check(self, item: Dict) -> bool:
        """数据质量检查"""
        
        original = item.get("original_query", "")
        rewritten = item.get("rewritten_query", "")
        
        # 基本检查
        if not original or not rewritten:
            return False
        
        # 长度检查
        if len(original) < 2 or len(original) > 200:
            return False
        
        if len(rewritten) < 2 or len(rewritten) > 500:
            return False
        
        # 相似度检查（避免改写前后完全一致）
        if original.strip() == rewritten.strip():
            return False
        
        # 过度改写检查（改写后不应该过长）
        if len(rewritten) > len(original) * 5:
            return False
        
        return True
    
    def augment_data(self, data: List[Dict], tenant_id: str) -> List[Dict]:
        """数据增强"""
        
        augmented = data.copy()
        
        # 同义词替换增强
        for item in data[:len(data)//3]:  # 对1/3数据进行增强
            aug_item = self._synonym_augmentation(item, tenant_id)
            if aug_item:
                augmented.append(aug_item)
        
        return augmented
    
    def _synonym_augmentation(self, item: Dict, tenant_id: str) -> Dict:
        """同义词替换增强"""
        
        # 针对不同租户的同义词库
        synonyms = {
            "fivedoctors": {
                "怎么": ["如何", "怎样"],
                "吃": ["服用", "使用"],
                "效果": ["作用", "功效"],
            },
            "chengla": {
                "学习": ["备考", "复习"],
                "课程": ["课堂", "培训"],
            }
        }
        
        tenant_synonyms = synonyms.get(tenant_id, {})
        
        original = item["original_query"]
        rewritten = item["rewritten_query"]
        
        # 随机替换
        import random
        for word, syns in tenant_synonyms.items():
            if word in original and random.random() < 0.3:
                syn = random.choice(syns)
                original = original.replace(word, syn, 1)
                rewritten = rewritten.replace(word, syn, 1)
        
        if original != item["original_query"]:
            aug_item = item.copy()
            aug_item["original_query"] = original
            aug_item["rewritten_query"] = rewritten
            aug_item["metadata"]["augmented"] = True
            return aug_item
        
        return None
    
    def save_datasets(self, all_data: Dict[str, List[Dict]]):
        """保存数据集"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for tenant_id, data in all_data.items():
            # 划分训练集、验证集、测试集
            train_size = int(len(data) * 0.8)
            val_size = int(len(data) * 0.1)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            # SFT格式数据
            sft_dir = self.output_dir / "sft" / tenant_id
            sft_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_jsonl(train_data, sft_dir / f"train_{timestamp}.jsonl")
            self._save_jsonl(val_data, sft_dir / f"val_{timestamp}.jsonl")
            self._save_jsonl(test_data, sft_dir / f"test_{timestamp}.jsonl")
            
            # 创建符号链接到latest
            for split in ["train", "val", "test"]:
                latest_link = sft_dir / f"{split}_latest.jsonl"
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(f"{split}_{timestamp}.jsonl")
            
            # 生成统计报告
            self._generate_stats_report(tenant_id, {
                "train": train_data,
                "val": val_data,
                "test": test_data
            })
    
    def _save_jsonl(self, data: List[Dict], filepath: Path):
        """保存为JSONL格式"""
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def _generate_stats_report(self, tenant_id: str, splits: Dict):
        """生成统计报告"""
        
        report = {
            "tenant_id": tenant_id,
            "timestamp": datetime.now().isoformat(),
            "splits": {}
        }
        
        for split_name, split_data in splits.items():
            report["splits"][split_name] = {
                "total_samples": len(split_data),
                "avg_original_length": sum(len(d["original_query"]) for d in split_data) / len(split_data),
                "avg_rewritten_length": sum(len(d["rewritten_query"]) for d in split_data) / len(split_data),
                "has_user_profile": sum(1 for d in split_data if d.get("context", {}).get("user_profile")),
                "has_history": sum(1 for d in split_data if d.get("context", {}).get("history_context")),
            }
        
        report_path = self.output_dir / "sft" / tenant_id / "stats_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n统计报告已保存: {report_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))


# 使用示例
async def main():
    collector = QueryRewriteDataCollector()
    
    data = await collector.collect_all_data(
        tenant_ids=["fivedoctors", "chengla"],
        days_back=60
    )
    
    print("\n数据收集完成！")
    print(f"输出目录: {collector.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 2️⃣ SFT训练详细步骤

### 2.1 环境准备

```bash
# 1. 创建训练环境
conda create -n query_rewrite_sft python=3.10
conda activate query_rewrite_sft

# 2. 安装依赖
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0
pip install datasets==2.16.0
pip install accelerate==0.25.0
pip install deepspeed==0.12.6
pip install wandb

# 3. 下载基座模型
# Qwen2.5-8B-Instruct
huggingface-cli download Qwen/Qwen2.5-8B-Instruct --local-dir models/Qwen2.5-8B-Instruct
```

### 2.2 数据转换为训练格式

```python
# convert_to_sft_format.py

from datasets import Dataset
import json
from typing import List, Dict

def convert_to_instruction_format(
    data: List[Dict],
    tenant_id: str
) -> List[Dict]:
    """
    将数据转换为instruction-following格式
    
    格式参考Qwen的对话模板
    """
    
    converted = []
    
    # 针对不同租户的系统提示
    system_prompts = {
        "fivedoctors": """你是一个专业的保健品知识库查询优化专家。
你的任务是将用户的口语化问题改写为更适合知识库检索的专业查询。

改写要求:
1. 提取核心产品关键词 (胶原蛋白肽、富铁软糖、虾青素等)
2. 明确查询意图 (功效、用法、禁忌、成分、适用人群等)
3. 补充必要的专业术语
4. 保持查询简洁性，避免冗余
5. 保留用户原始意图""",
        
        "chengla": """你是一个专业的教育培训知识库查询优化专家。
你的任务是将学员的问题改写为更适合检索的查询。

改写要求:
1. 识别课程类型和科目
2. 明确学习阶段和需求
3. 提取关键知识点
4. 保持教育领域专业性""",
    }
    
    system_prompt = system_prompts.get(tenant_id, system_prompts["fivedoctors"])
    
    for item in data:
        # 构建输入
        user_input = f"""请优化以下查询:

原始查询: {item['original_query']}"""
        
        # 如果有用户画像和历史
        context = item.get("context", {})
        if context.get("user_profile"):
            user_input += f"\n用户画像: {context['user_profile']}"
        
        if context.get("history_summary"):
            user_input += f"\n历史摘要: {context['history_summary']}"
        
        if context.get("history_context"):
            user_input += f"\n对话上下文:\n{context['history_context'][-500:]}"  # 只保留最近500字符
        
        # 构建对话格式
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": item['rewritten_query']}
            ],
            "metadata": item.get("metadata", {})
        }
        
        converted.append(conversation)
    
    return converted


def create_sft_dataset(
    tenant_id: str,
    data_dir: str = "data/query_rewrite_training/sft"
):
    """创建SFT训练数据集"""
    
    import os
    from pathlib import Path
    
    data_path = Path(data_dir) / tenant_id
    
    # 读取数据
    splits = {}
    for split in ["train", "val", "test"]:
        filepath = data_path / f"{split}_latest.jsonl"
        
        if not filepath.exists():
            print(f"警告: {filepath} 不存在")
            continue
        
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        # 转换格式
        converted = convert_to_instruction_format(data, tenant_id)
        
        # 创建Dataset对象
        splits[split] = Dataset.from_list(converted)
    
    return splits


# 使用示例
if __name__ == "__main__":
    datasets = create_sft_dataset("fivedoctors")
    
    print("训练集样本数:", len(datasets["train"]))
    print("验证集样本数:", len(datasets["val"]))
    print("\n样本示例:")
    print(datasets["train"][0])
```

### 2.3 SFT训练脚本

```python
# train_sft.py

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import wandb

class QueryRewriteSFTTrainer:
    """Query改写SFT训练器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-8B-Instruct",
        tenant_id: str = "fivedoctors",
        output_dir: str = "outputs/sft"
    ):
        self.model_name = model_name
        self.tenant_id = tenant_id
        self.output_dir = f"{output_dir}/{tenant_id}"
        
        # 初始化wandb
        wandb.init(
            project="sales-rag-query-rewrite",
            name=f"sft_{tenant_id}",
            config={
                "model": model_name,
                "tenant_id": tenant_id,
                "task": "query_rewrite_sft"
            }
        )
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 添加pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 启用梯度检查点以节省内存
        self.model.gradient_checkpointing_enable()
    
    def preprocess_function(self, examples):
        """数据预处理"""
        
        inputs = []
        targets = []
        
        for messages in examples["messages"]:
            # 应用Qwen的对话模板
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            inputs.append(formatted)
            
            # 提取assistant的回复作为target
            assistant_msg = [m for m in messages if m["role"] == "assistant"][0]
            targets.append(assistant_msg["content"])
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 4
    ):
        """执行SFT训练"""
        
        # 预处理数据
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            
            # 优化器配置
            optim="adamw_torch",
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # 日志和保存
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            save_total_limit=3,
            
            # 评估
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # 混合精度
            bf16=True,
            
            # DeepSpeed
            deepspeed="configs/ds_config_zero2.json",
            
            # Wandb
            report_to="wandb",
            run_name=f"sft_{self.tenant_id}"
        )
        
        # 数据collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        print(f"开始SFT训练 - {self.tenant_id}")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(eval_dataset)}")
        
        trainer.train()
        
        # 保存最终模型
        final_model_path = f"{self.output_dir}/final"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print(f"训练完成！模型保存在: {final_model_path}")
        
        return trainer


# 使用示例
if __name__ == "__main__":
    from convert_to_sft_format import create_sft_dataset
    
    # 1. 加载数据
    datasets = create_sft_dataset("fivedoctors")
    
    # 2. 创建训练器
    trainer = QueryRewriteSFTTrainer(
        model_name="Qwen/Qwen2.5-8B-Instruct",
        tenant_id="fivedoctors"
    )
    
    # 3. 开始训练
    trainer.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )
```

### 2.4 DeepSpeed配置

```json
// configs/ds_config_zero2.json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

### 2.5 训练启动脚本

```bash
#!/bin/bash
# scripts/train_sft.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="sales-rag-query-rewrite"

TENANT_ID="fivedoctors"
MODEL_NAME="Qwen/Qwen2.5-8B-Instruct"
OUTPUT_DIR="outputs/sft/${TENANT_ID}"
NUM_GPUS=4

echo "=========================================="
echo "SFT Training - ${TENANT_ID}"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# 启动分布式训练
deepspeed --num_gpus=${NUM_GPUS} train_sft.py \
    --model_name ${MODEL_NAME} \
    --tenant_id ${TENANT_ID} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4

echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}/final"
```

---

## 3️⃣ RL训练详细步骤

### 3.1 评分模型训练

首先需要训练DeepSeek评分模型：

```python
# train_scorer.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

class QueryRewriteScorerTrainer:
    """Query改写评分模型训练器"""
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V2.5",
        num_labels: int = 5  # 5个维度评分
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16
        )
    
    def prepare_scorer_data(self, annotation_file: str):
        """准备评分训练数据
        
        数据格式:
        {
            "original_query": "胶原蛋白怎么吃",
            "rewritten_query": "胶原蛋白肽 服用方法 推荐用量",
            "scores": {
                "改写质量": 5,
                "领域适配": 4,
                "意图保持": 5,
                "可检索性": 4
            }
        }
        """
        
        data = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                
                # 构建输入文本
                text = f"""原始查询: {item['original_query']}
改写查询: {item['rewritten_query']}

请对改写质量进行评分 (1-5分):
1. 改写质量
2. 领域适配
3. 意图保持
4. 可检索性"""
                
                # 标签 (归一化到0-1)
                scores = item["scores"]
                labels = [
                    scores["改写质量"] / 5.0,
                    scores["领域适配"] / 5.0,
                    scores["意图保持"] / 5.0,
                    scores["可检索性"] / 5.0,
                    sum(scores.values()) / (5.0 * len(scores))  # 综合得分
                ]
                
                data.append({
                    "text": text,
                    "labels": labels
                })
        
        return Dataset.from_list(data)
    
    def train(self, train_dataset, eval_dataset):
        """训练评分模型"""
        
        training_args = TrainingArguments(
            output_dir="outputs/scorer",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=1e-5,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            logging_steps=50,
            bf16=True,
            report_to="wandb"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        trainer.save_model("outputs/scorer/final")
```

### 3.2 Reward函数设计

这是RL训练的核心：

```python
# rl_reward_function.py

import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MultiDimensionalReward:
    """多维度奖励函数"""
    
    def __init__(self):
        # 加载评分模型
        self.scorer_model = AutoModelForSequenceClassification.from_pretrained(
            "outputs/scorer/final"
        )
        self.scorer_tokenizer = AutoTokenizer.from_pretrained(
            "outputs/scorer/final"
        )
        self.scorer_model.eval()
        
        # 权重配置
        self.weights = {
            "deepseek_score": 0.4,      # DeepSeek评分权重
            "retrieval_quality": 0.35,   # 检索质量权重
            "relative_improvement": 0.25 # 相对32B的提升权重
        }
    
    def compute_reward(
        self,
        original_query: str,
        qwen8b_rewrite: str,
        qwen32b_rewrite: str,
        context: Dict,
        retrieval_results: Dict
    ) -> float:
        """
        计算综合奖励
        
        Returns:
            reward: float, 范围 [-1, 1]
        """
        
        # 1. DeepSeek评分奖励
        deepseek_reward = self._compute_deepseek_reward(
            original_query,
            qwen8b_rewrite,
            qwen32b_rewrite
        )
        
        # 2. 检索质量奖励
        retrieval_reward = self._compute_retrieval_reward(
            retrieval_results
        )
        
        # 3. 相对提升奖励 (8B vs 32B)
        improvement_reward = self._compute_improvement_reward(
            deepseek_reward,
            retrieval_reward,
            baseline="qwen32b"
        )
        
        # 加权求和
        total_reward = (
            self.weights["deepseek_score"] * deepseek_reward +
            self.weights["retrieval_quality"] * retrieval_reward +
            self.weights["relative_improvement"] * improvement_reward
        )
        
        # 归一化到[-1, 1]
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        return total_reward
    
    def _compute_deepseek_reward(
        self,
        original: str,
        rewrite_8b: str,
        rewrite_32b: str
    ) -> float:
        """DeepSeek评分奖励"""
        
        # 评分8B的改写
        score_8b = self._get_deepseek_score(original, rewrite_8b)
        
        # 评分32B的改写
        score_32b = self._get_deepseek_score(original, rewrite_32b)
        
        # 计算相对奖励
        # 如果8B > 32B，给正奖励
        # 如果8B < 32B，给负奖励
        # 使用tanh函数平滑
        diff = score_8b - score_32b
        reward = np.tanh(diff * 2)  # 放大差异
        
        return reward
    
    def _get_deepseek_score(self, original: str, rewritten: str) -> float:
        """使用DeepSeek模型评分"""
        
        text = f"""原始查询: {original}
改写查询: {rewritten}

请对改写质量进行评分 (1-5分):
1. 改写质量
2. 领域适配
3. 意图保持
4. 可检索性"""
        
        inputs = self.scorer_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.scorer_model(**inputs)
            scores = torch.sigmoid(outputs.logits[0])  # [5] -> [0, 1]
        
        # 综合得分 (最后一个输出是综合分)
        综合得分 = scores[-1].item()
        
        return 综合得分
    
    def _compute_retrieval_reward(self, retrieval_results: Dict) -> float:
        """检索质量奖励
        
        基于检索结果的相关性分数计算
        """
        
        results_8b = retrieval_results.get("qwen_8b_results", [])
        results_32b = retrieval_results.get("qwen_32b_results", [])
        
        # 计算8B和32B的检索质量
        quality_8b = self._calculate_retrieval_quality(results_8b)
        quality_32b = self._calculate_retrieval_quality(results_32b)
        
        # 相对奖励
        diff = quality_8b - quality_32b
        reward = np.tanh(diff * 3)  # 检索质量差异更重要
        
        return reward
    
    def _calculate_retrieval_quality(self, results: List[Dict]) -> float:
        """计算检索质量分数"""
        
        if not results:
            return 0.0
        
        # 考虑多个因素
        # 1. Top-1分数
        top1_score = results[0].get("reranker_score", results[0].get("score", 0))
        
        # 2. Top-3平均分数
        top3_scores = [r.get("reranker_score", r.get("score", 0)) for r in results[:3]]
        avg_top3 = np.mean(top3_scores) if top3_scores else 0
        
        # 3. 分数衰减 (检查结果的质量分布)
        if len(results) >= 2:
            score_gap = results[0].get("reranker_score", 0) - results[1].get("reranker_score", 0)
            gap_reward = np.clip(score_gap, 0, 0.2) * 2  # 归一化到[0, 0.4]
        else:
            gap_reward = 0
        
        # 综合质量分数
        quality = 0.5 * top1_score + 0.3 * avg_top3 + 0.2 * gap_reward
        
        return quality
    
    def _compute_improvement_reward(
        self,
        deepseek_reward: float,
        retrieval_reward: float,
        baseline: str = "qwen32b"
    ) -> float:
        """相对提升奖励
        
        鼓励8B模型超越32B baseline
        """
        
        # 综合改进度
        improvement = (deepseek_reward + retrieval_reward) / 2
        
        # 如果有显著提升，给额外奖励
        if improvement > 0.3:
            bonus = 0.5  # 显著超越32B
        elif improvement > 0.1:
            bonus = 0.2  # 略微超越32B
        elif improvement > -0.1:
            bonus = 0.0  # 持平
        else:
            bonus = -0.3  # 明显不如32B，惩罚
        
        return bonus


class AdaptiveRewardShaping:
    """自适应奖励塑形
    
    根据训练进度动态调整奖励
    """
    
    def __init__(self):
        self.training_step = 0
        self.reward_history = []
    
    def shape_reward(self, base_reward: float) -> float:
        """奖励塑形"""
        
        self.training_step += 1
        self.reward_history.append(base_reward)
        
        # 早期训练：放大奖励，鼓励探索
        if self.training_step < 1000:
            shaped_reward = base_reward * 1.5
        
        # 中期训练：正常奖励
        elif self.training_step < 5000:
            shaped_reward = base_reward
        
        # 后期训练：更精细的奖励
        else:
            # 计算最近100步的平均奖励
            recent_avg = np.mean(self.reward_history[-100:])
            
            # 如果当前奖励显著高于平均，放大奖励
            if base_reward > recent_avg + 0.2:
                shaped_reward = base_reward * 1.3
            else:
                shaped_reward = base_reward
        
        return np.clip(shaped_reward, -1.0, 1.0)
```

### 3.3 RL训练主流程

```python
# train_rl.py

import asyncio
import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

# 导入VERL框架 (参考DeepRetrieval)
from verl.trainer import PPOTrainer
from verl.utils.reward_score import RewardFunction

class QueryRewriteRLTrainer:
    """Query改写RL训练器"""
    
    def __init__(
        self,
        qwen8b_model_path: str,  # SFT训练后的8B模型
        qwen32b_api_url: str,     # 32B模型API
        tenant_id: str = "fivedoctors"
    ):
        self.tenant_id = tenant_id
        
        # 加载8B模型 (policy model)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            qwen8b_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(qwen8b_model_path)
        
        # 32B模型API
        self.qwen32b_api = qwen32b_api_url
        
        # 奖励函数
        self.reward_function = MultiDimensionalReward()
        self.reward_shaper = AdaptiveRewardShaping()
        
        # 初始化wandb
        wandb.init(
            project="sales-rag-query-rewrite",
            name=f"rl_{tenant_id}",
            config={
                "tenant_id": tenant_id,
                "task": "query_rewrite_rl",
                "algorithm": "PPO"
            }
        )
    
    async def generate_training_episode(
        self,
        original_query: str,
        context: Dict
    ) -> Dict:
        """生成一个训练episode
        
        Returns:
            {
                "original_query": str,
                "qwen8b_rewrite": str,
                "qwen32b_rewrite": str,
                "retrieval_results": {...},
                "reward": float
            }
        """
        
        # 1. 8B模型生成改写
        qwen8b_rewrite = await self._generate_rewrite_8b(
            original_query, context
        )
        
        # 2. 32B模型生成改写 (作为baseline)
        qwen32b_rewrite = await self._generate_rewrite_32b(
            original_query, context
        )
        
        # 3. 并行检索两个改写的结果
        retrieval_results = await self._parallel_retrieval(
            qwen8b_rewrite,
            qwen32b_rewrite
        )
        
        # 4. 计算奖励
        base_reward = self.reward_function.compute_reward(
            original_query=original_query,
            qwen8b_rewrite=qwen8b_rewrite,
            qwen32b_rewrite=qwen32b_rewrite,
            context=context,
            retrieval_results=retrieval_results
        )
        
        # 5. 奖励塑形
        shaped_reward = self.reward_shaper.shape_reward(base_reward)
        
        return {
            "original_query": original_query,
            "context": context,
            "qwen8b_rewrite": qwen8b_rewrite,
            "qwen32b_rewrite": qwen32b_rewrite,
            "retrieval_results": retrieval_results,
            "base_reward": base_reward,
            "shaped_reward": shaped_reward
        }
    
    async def _generate_rewrite_8b(
        self,
        query: str,
        context: Dict
    ) -> str:
        """使用8B模型生成改写"""
        
        # 构建输入
        prompt = self._build_prompt(query, context)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.policy_model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        rewrite = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return rewrite.strip()
    
    async def _generate_rewrite_32b(
        self,
        query: str,
        context: Dict
    ) -> str:
        """调用32B模型API生成改写"""
        
        import httpx
        
        # 调用现有的rewrite_query_by_model逻辑
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.qwen32b_api}/api/rewrite_query",
                json={
                    "query": query,
                    "history": context.get("history_context", ""),
                    "thought": context.get("thought_unit", ""),
                    "tenant_id": self.tenant_id,
                    "user_profile": context.get("user_profile", ""),
                    "history_summary": context.get("history_summary", "")
                }
            )
            
            result = response.json()
            return result.get("rewritten_query", query)
    
    async def _parallel_retrieval(
        self,
        rewrite_8b: str,
        rewrite_32b: str
    ) -> Dict:
        """并行检索两个改写的结果"""
        
        from chatchat.server.chat.kb_chat_v2 import kb_chat_v2
        from chatchat.server.reranker.reranker import LangchainReranker
        
        # 并行检索
        results_8b_task = asyncio.create_task(
            self._retrieve_and_rerank(rewrite_8b)
        )
        results_32b_task = asyncio.create_task(
            self._retrieve_and_rerank(rewrite_32b)
        )
        
        results_8b = await results_8b_task
        results_32b = await results_32b_task
        
        return {
            "qwen_8b_results": results_8b,
            "qwen_32b_results": results_32b
        }
    
    async def _retrieve_and_rerank(self, query: str) -> List[Dict]:
        """检索并重排序"""
        
        # 使用现有的检索流程
        from chatchat.server.chat.kb_chat_v2 import kb_chat_v2
        
        # 检索
        docs = kb_chat_v2(
            query=query,
            mode="local_kb",
            kb_name=f"{self.tenant_id}",
            top_k=10,
            score_threshold=0.5
        )
        
        # Rerank (如果是fivedoctors或chengla)
        if self.tenant_id in ["fivedoctors", "chengla"]:
            from chatchat.server.api_server.general_rag_utils import init_reranker, dynamic_elbow_rerank
            from langchain_core.documents import Document
            
            reranker = init_reranker()
            if reranker and docs:
                # 转换为Document格式
                langchain_docs = [
                    Document(
                        page_content=doc.get("content", ""),
                        metadata={k: v for k, v in doc.items() if k != "content"}
                    )
                    for doc in docs
                ]
                
                # Rerank
                reranked_docs = reranker.compress_documents(langchain_docs, query)
                
                # 动态截断
                final_docs = dynamic_elbow_rerank(reranked_docs, query)
                
                # 转换回dict格式
                docs = [
                    {
                        **doc.metadata,
                        "content": doc.page_content,
                        "reranker_score": doc.metadata.get("relevance_score", 0)
                    }
                    for doc in final_docs
                ]
        
        return docs
    
    def train_with_ppo(
        self,
        train_data_path: str,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-6
    ):
        """使用PPO算法训练"""
        
        # PPO训练配置
        ppo_config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "clip_range": 0.2,
            "vf_coef": 0.1,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "lambda_": 0.95,
        }
        
        # 创建PPO Trainer (使用VERL框架)
        from verl.trainer.ppo import PPOTrainer as VERLPPOTrainer
        
        trainer = VERLPPOTrainer(
            model=self.policy_model,
            tokenizer=self.tokenizer,
            reward_fn=self.reward_function,
            config=ppo_config,
            output_dir=f"outputs/rl/{self.tenant_id}"
        )
        
        # 加载训练数据
        import json
        train_queries = []
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                train_queries.append(json.loads(line))
        
        print(f"加载了 {len(train_queries)} 条训练数据")
        
        # 训练循环
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            epoch_rewards = []
            
            for batch_idx in range(0, len(train_queries), batch_size):
                batch = train_queries[batch_idx:batch_idx + batch_size]
                
                # 生成episodes
                episodes = []
                for item in batch:
                    episode = asyncio.run(
                        self.generate_training_episode(
                            original_query=item["original_query"],
                            context=item.get("context", {})
                        )
                    )
                    episodes.append(episode)
                
                # PPO更新
                metrics = trainer.step(episodes)
                
                # 记录奖励
                batch_reward = np.mean([ep["shaped_reward"] for ep in episodes])
                epoch_rewards.append(batch_reward)
                
                # 日志
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_queries)}, "
                          f"Avg Reward: {batch_reward:.4f}, "
                          f"Policy Loss: {metrics['policy_loss']:.4f}")
                    
                    wandb.log({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "avg_reward": batch_reward,
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"]
                    })
            
            # Epoch总结
            avg_epoch_reward = np.mean(epoch_rewards)
            print(f"Epoch {epoch + 1} 平均奖励: {avg_epoch_reward:.4f}")
            
            # 保存checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"outputs/rl/{self.tenant_id}/checkpoint_epoch{epoch + 1}"
                self.policy_model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                print(f"Checkpoint已保存: {checkpoint_path}")
        
        # 保存最终模型
        final_path = f"outputs/rl/{self.tenant_id}/final"
        self.policy_model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"最终模型已保存: {final_path}")


# 使用示例
if __name__ == "__main__":
    trainer = QueryRewriteRLTrainer(
        qwen8b_model_path="outputs/sft/fivedoctors/final",
        qwen32b_api_url="http://localhost:7861",
        tenant_id="fivedoctors"
    )
    
    trainer.train_with_ppo(
        train_data_path="data/query_rewrite_training/sft/fivedoctors/train_latest.jsonl",
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-6
    )
```

### 3.4 RL训练启动脚本

```bash
#!/bin/bash
# scripts/train_rl.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="sales-rag-query-rewrite"

TENANT_ID="fivedoctors"
SFT_MODEL_PATH="outputs/sft/${TENANT_ID}/final"
QWEN32B_API="http://localhost:7861"
OUTPUT_DIR="outputs/rl/${TENANT_ID}"

echo "=========================================="
echo "RL Training - ${TENANT_ID}"
echo "=========================================="
echo "SFT Model: ${SFT_MODEL_PATH}"
echo "32B API: ${QWEN32B_API}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# 检查SFT模型是否存在
if [ ! -d "${SFT_MODEL_PATH}" ]; then
    echo "错误: SFT模型不存在: ${SFT_MODEL_PATH}"
    echo "请先运行 SFT 训练"
    exit 1
fi

# 启动RL训练
python train_rl.py \
    --qwen8b_model_path ${SFT_MODEL_PATH} \
    --qwen32b_api_url ${QWEN32B_API} \
    --tenant_id ${TENANT_ID} \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 1e-6

echo "RL训练完成！"
echo "模型保存在: ${OUTPUT_DIR}/final"
```

---

## 4️⃣ 完整训练流程总结

```
Step 1: 数据收集 (1-2天)
├─ 从生产日志提取32B改写数据
├─ 整理测试集数据
├─ 人工标注高质量样本 (500+条)
└─ 生成训练/验证/测试集

Step 2: 评分模型训练 (1天)
├─ 准备评分标注数据
├─ 训练DeepSeek评分模型
└─ 评估评分模型性能

Step 3: SFT训练 (2-3天)
├─ 数据格式转换
├─ Qwen-8B SFT训练 (3 epochs)
├─ 验证SFT效果
└─ 部署SFT模型

Step 4: RL训练 (5-7天)
├─ 双模型部署 (8B + 32B)
├─ PPO训练循环 (10 epochs)
├─ 持续监控奖励曲线
└─ 模型性能评估

Step 5: 效果验证 (3-5天)
├─ A/B测试部署
├─ 业务指标监控
├─ 用户反馈收集
└─ 迭代优化
```

**总计**: 约2-3周完成完整训练流程

---

## 📊 预期效果

| 指标 | Baseline (32B) | SFT (8B) | RL (8B) |
|------|----------------|----------|---------|
| 改写质量评分 | 4.2/5 | 3.8/5 | 4.5/5 |
| 检索Top-1准确率 | 78% | 72% | 85% |
| 推理延迟 | 850ms | 320ms | 350ms |
| 成本/1000次 | $2.50 | $0.80 | $0.85 |

通过RL训练，预期8B模型能在保持低成本的同时，在改写质量和检索效果上**超越32B baseline**！
