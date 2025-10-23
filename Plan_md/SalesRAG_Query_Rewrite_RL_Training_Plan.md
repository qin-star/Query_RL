# Sales-RAG Query改写RL训练方案

> 基于Qwen-8B的两阶段训练：SFT知识蒸馏 + RL竞争优化

---

## 📋 方案概述

### 核心思路

```
阶段1: SFT知识蒸馏
Qwen-32B (Teacher) → 改写数据 → Qwen-8B (Student) SFT训练

阶段2: RL竞争优化  
Qwen-8B ↔ Qwen-32B (双模型竞争) + GPT-5评分 → PPO/GRPO优化
```

### 技术栈

- **基座模型**: Qwen2.5-8B-Instruct
- **教师模型**: Qwen-32B (现有部署)
- **评分模型**: GPT-5 (API调用)
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
        "gpt5_scores": {
          "qwen_8b": {
            "改写质量": 4.2,
            "领域适配": 4.0,
            "意图保持": 4.5,
            "可检索性": 4.3,
            "综合得分": 4.25,
            "评分理由": "改写质量较好，保留了原意"
          },
          "qwen_32b": {
            "改写质量": 4.7,
            "领域适配": 4.8,
            "意图保持": 4.9,
            "可检索性": 4.6,
            "综合得分": 4.75,
            "评分理由": "改写质量优秀，领域适配性强"
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

### 1.4 基于测试集生成SFT训练数据（实战方案）

#### 核心思路

在实际业务中，我们已经有现成的测试集和RAG框架，可以直接利用：

1. **批量测试RAG框架**：使用测试集批量调用现有RAG系统，收集32B改写结果和检索效果
2. **保存为Excel**：将测试结果保存到 `test_sft.xlsx`，包含完整的query、改写、检索结果
3. **转换为训练数据**：将Excel转换为SFT训练格式（JSONL）
4. **质量筛选**：根据检索效果筛选高质量样本用于训练

这种方式的优势：

- ✅ **真实数据**：来自实际业务场景的测试集
- ✅ **快速获取**：无需等待线上日志积累
- ✅ **质量可控**：测试集通常经过人工审核
- ✅ **包含检索反馈**：同时获得改写结果和检索效果

#### 步骤1：批量测试RAG框架

通过BVT数据集进行RAG批量化测试，保存下面几类INFO用于SFT训练集

```python
 "data": {  
	"user_profile": user_profileor"",   
	 "history_summary": history_summaryor"",   
	 "rewritten_query": new_query, 
	 "recall": search_res1    },
```

#### 步骤2：从test_sft.xlsx转换为SFT训练数据

构建json格式的训练集

```python
# convert_test_to_sft.py

import pandas as pd
import json
from typing import List, Dict
from pathlib import Path

class TestToSFTConverter:
    """将测试结果转换为SFT训练数据"""
  
    def __init__(self, tenant_id: str = "fivedoctors"):
        self.tenant_id = tenant_id
    
        # 系统prompt（与前面定义一致）
        self.system_prompts = {
            "fivedoctors": """你是一个专业的保健品知识库查询优化专家...""",
            "chengla": """你是一个专业的教育培训知识库查询优化专家..."""
        }
  
    def convert_excel_to_jsonl(
        self,
        excel_path: str,
        output_jsonl: str,
        quality_threshold: float = 0.6
    ):
        """将Excel转换为JSONL训练格式
    
        Args:
            excel_path: test_sft.xlsx路径
            output_jsonl: 输出的JSONL文件路径
            quality_threshold: 质量阈值（基于top1_score筛选）
        """
    
        # 读取Excel
        df = pd.read_excel(excel_path)
        print(f"📚 读取测试结果: {len(df)} 条")
    
        # 质量筛选
        # 1. 只保留成功的测试
        df = df[df['success'] == True]
    
        # 2. 筛选检索效果好的样本（top1_score > threshold）
        df = df[df['top1_score'] >= quality_threshold]
    
        # 3. 确保改写不为空且与原query不同
        df = df[
            (df['rewritten_query'].notna()) &
            (df['rewritten_query'] != df['original_query'])
        ]
    
        print(f"✅ 质量筛选后: {len(df)} 条 (保留率: {len(df)/len(df)*100:.1f}%)")
    
        # 转换为训练格式
        training_samples = []
        system_prompt = self.system_prompts.get(self.tenant_id, "")
    
        for _, row in df.iterrows():
            # 构建用户输入
            user_content = f"""原始查询: {row['original_query']}"""
        
            # 添加上下文信息（如果有）
            if pd.notna(row.get('user_profile')) and row['user_profile']:
                user_content += f"\n\n用户画像: {row['user_profile']}"
        
            if pd.notna(row.get('history_summary')) and row['history_summary']:
                user_content += f"\n\n历史摘要: {row['history_summary']}"
        
            user_content += "\n\n请改写这个查询，使其更适合知识库检索。"
        
            # 构建对话
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": row['rewritten_query']
                    }
                ],
                "metadata": {
                    "source": "test_set",
                    "tenant_id": self.tenant_id,
                    "top1_score": float(row['top1_score']),
                    "recall_count": int(row['recall_count'])
                }
            }
        
            training_samples.append(sample)
    
        # 保存为JSONL
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
        print(f"✅ 训练数据已保存: {output_jsonl}")
        print(f"   总样本数: {len(training_samples)}")
    
        return training_samples
  
    def split_train_val_test(
        self,
        jsonl_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ):
        """划分训练集、验证集、测试集"""
    
        # 读取所有样本
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
    
        # 打乱
        import random
        random.shuffle(samples)
    
        # 划分
        n = len(samples)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
    
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
    
        # 保存
        base_dir = Path(jsonl_path).parent
    
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
        for split_name, split_data in splits.items():
            output_file = base_dir / f"{split_name}_latest.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"  - {split_name}: {len(split_data)} 条 → {output_file}")
    
        return splits


# 使用示例
if __name__ == "__main__":
    converter = TestToSFTConverter(tenant_id="fivedoctors")
  
    # 转换
    samples = converter.convert_excel_to_jsonl(
        excel_path="data/test_sft_fivedoctors.xlsx",
        output_jsonl="data/sft/fivedoctors/all_samples.jsonl",
        quality_threshold=0.6
    )
  
    # 划分数据集
    converter.split_train_val_test(
        jsonl_path="data/sft/fivedoctors/all_samples.jsonl"
    )
  
    print("\n✨ SFT数据准备完成！")
```

#### 步骤3：快速启动SFT训练

准备好数据后，可以直接开始训练：

基于ms-swift训练框架

https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html

```bash
# 显存占用：22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' \
              'swift/self-cognition:qwen3#600' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

**预期数据规模**：

- 测试集: 500-1000条
- 质量筛选后: 300-800条 (top1_score > 0.6)
- 训练集: 240-640条
- 验证集: 30-80条
- 测试集: 30-80条

#### test_sft.xlsx 字段说明

| 字段            | 说明                 | 示例                           |
| --------------- | -------------------- | ------------------------------ |
| original_query  | 原始用户query        | "胶原蛋白怎么吃"               |
| user_profile    | 用户画像             | "25-35岁女性，关注抗衰老"      |
| history_summary | 历史摘要             | "近期咨询过多次胶原蛋白产品"   |
| rewritten_query | 32B改写的query       | "胶原蛋白肽 服用方法 推荐用量" |
| recall_results  | 检索结果列表（JSON） | [...]                          |
| recall_count    | 召回文档数           | 5                              |
| top1_score      | Top1文档reranker分数 | 0.87                           |
| avg_top3_score  | Top3平均分数         | 0.82                           |
| success         | 测试是否成功         | True                           |

---

## 3️⃣ RL训练详细步骤

### 3.1 GPT-5评分模型配置

#### 为什么使用GPT-5作为评分模型？

相比训练专门的评分模型，直接使用GPT-5 API有以下优势：

**✅ 优势**

1. **零训练成本**: 无需准备大量标注数据和GPU资源训练评分模型
2. **高质量评分**: GPT-5具备强大的语义理解能力，评分更准确和一致
3. **灵活可调**: 通过prompt工程即可快速调整评分标准，无需重新训练
4. **快速上线**: 省去评分模型训练的1-2天时间，加速整体训练流程
5. **可解释性**: GPT-5可以提供评分理由，便于理解和调试

**⚠️ 注意事项**

- API调用成本: 需要考虑GPT-5 API调用费用（可通过批量调用和缓存优化）
- 调用延迟: RL训练中需要大量评分，建议使用异步批量调用
- 稳定性: 设置低temperature（0.1）保证评分的稳定性和一致性

#### GPT-5评分器实现

使用GPT-5 API作为评分模型，通过精心设计的prompt进行打分：

```python
# gpt5_scorer.py

import os
import json
from openai import OpenAI
from typing import Dict, List, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class GPT5QueryRewriteScorer:
    """基于GPT-5的Query改写评分器"""
  
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        temperature: float = 0.1,  # 低温度保证评分稳定性
        max_tokens: int = 500
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
  
        # 评分维度定义
        self.dimensions = [
            "改写质量",
            "领域适配",
            "意图保持",
            "可检索性"
        ]
  
    def build_scoring_prompt(
        self,
        original_query: str,
        rewritten_query: str,
        context: Optional[Dict] = None
    ) -> str:
        """构建评分prompt"""
  
        prompt = f"""# Query改写质量评估任务

你是一个专业的Query改写质量评估专家，需要对电商领域的Query改写结果进行多维度评分。

## 原始Query
{original_query}

## 改写Query
{rewritten_query}"""

        # 添加上下文信息（如果有）
        if context:
            if context.get("user_profile"):
                prompt += f"\n\n## 用户画像\n{context['user_profile']}"
            if context.get("history_summary"):
                prompt += f"\n\n## 历史摘要\n{context['history_summary']}"

        prompt += """

## 评分维度说明

请从以下4个维度对改写质量进行评分（每个维度1-5分）：

### 1. 改写质量 (1-5分)
- 5分: 改写后的query极大提升了表达的清晰度和专业性
- 4分: 改写后的query明显优于原query，表达更清晰
- 3分: 改写后的query有所改进，但提升有限
- 2分: 改写后的query与原query差异不大
- 1分: 改写后的query质量下降或偏离原意

### 2. 领域适配 (1-5分)
- 5分: 完美融入领域术语和专业表达，非常符合电商场景
- 4分: 较好地使用了领域相关词汇
- 3分: 基本符合领域特点
- 2分: 领域适配不足
- 1分: 完全不符合领域特点

### 3. 意图保持 (1-5分)
- 5分: 完美保留了用户的原始意图，且表达更清晰
- 4分: 很好地保留了用户意图
- 3分: 基本保留了用户意图
- 2分: 部分偏离了用户意图
- 1分: 严重偏离或完全改变了用户意图

### 4. 可检索性 (1-5分)
- 5分: 改写后的query极大提升了检索相关文档的能力
- 4分: 改写后的query明显更易检索到相关内容
- 3分: 改写后的query检索性有所提升
- 2分: 改写后的query检索性提升不明显
- 1分: 改写后的query反而降低了检索效果

## 输出格式要求

请严格按照以下JSON格式输出评分结果：

```json
{
  "改写质量": <1-5的整数>,
  "领域适配": <1-5的整数>,
  "意图保持": <1-5的整数>,
  "可检索性": <1-5的整数>,
  "综合得分": <四个维度的平均分，保留2位小数>,
  "评分理由": "<简要说明评分依据，50-100字>"
}
```

```python
请开始评分："""        return prompt    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def score_rewrite(
        self,
        original_query: str,
        rewritten_query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """对改写结果进行评分        Returns:
            {
                "改写质量": 4.5,
                "领域适配": 4.0,
                "意图保持": 5.0,
                "可检索性": 4.5,
                "综合得分": 4.5,
                "评分理由": "改写效果很好..."
            }
        """        prompt = self.build_scoring_prompt(original_query, rewritten_query, context)        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的Query改写质量评估专家。请严格按照要求进行评分，并以JSON格式返回结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # 确保返回JSON格式
            )            result_text = response.choices[0].message.content
            scores = json.loads(result_text)            # 验证和归一化得分
            scores = self._validate_scores(scores)            return scores        except Exception as e:
            print(f"GPT-5评分失败: {e}")
            # 返回默认中等分数
            return {
                "改写质量": 3.0,
                "领域适配": 3.0,
                "意图保持": 3.0,
                "可检索性": 3.0,
                "综合得分": 3.0,
                "评分理由": f"评分失败: {str(e)}"
            }    def _validate_scores(self, scores: Dict) -> Dict:
        """验证和归一化评分结果"""        # 确保所有维度都存在
        for dim in self.dimensions:
            if dim not in scores:
                scores[dim] = 3.0
            else:
                # 确保分数在1-5范围内
                scores[dim] = max(1.0, min(5.0, float(scores[dim])))        # 重新计算综合得分
        scores["综合得分"] = sum(scores[dim] for dim in self.dimensions) / len(self.dimensions)
        scores["综合得分"] = round(scores["综合得分"], 2)        # 确保有评分理由
        if "评分理由" not in scores:
            scores["评分理由"] = "基于多维度综合评估"        return scores    async def batch_score(
        self,
        query_pairs: List[Dict],
        max_concurrent: int = 5
    ) -> List[Dict]:
        """批量评分（异步）        Args:
            query_pairs: [
                {
                    "original_query": "...",
                    "rewritten_query": "...",
                    "context": {...}
                }
            ]
        """        semaphore = asyncio.Semaphore(max_concurrent)        async def score_one(pair):
            async with semaphore:
                # 转换为同步调用（在实际使用中可以使用异步HTTP库）
                return await asyncio.to_thread(
                    self.score_rewrite,
                    pair["original_query"],
                    pair["rewritten_query"],
                    pair.get("context")
                )        tasks = [score_one(pair) for pair in query_pairs]
        results = await asyncio.gather(*tasks)  
return results
```

# 使用示例

if __name__ == "__main__":
    scorer = GPT5QueryRewriteScorer()

    # 单个评分
    result = scorer.score_rewrite(
        original_query="胶原蛋白怎么吃",
        rewritten_query="胶原蛋白肽 服用方法 推荐用量 适用人群",
        context={
            "user_profile": "25-35岁女性，关注抗衰老",
            "history_summary": "近期咨询过多次胶原蛋白产品"
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

### 3.2 Reward函数设计

这是RL训练的核心：

```python

# rl_reward_function.py

import numpy as np
from typing import Dict, List, Tuple
from gpt5_scorer import GPT5QueryRewriteScorer

class MultiDimensionalReward:
    """多维度奖励函数"""
  
    def __init__(self, gpt5_api_key: str = None):
        # 初始化GPT-5评分器
        self.gpt5_scorer = GPT5QueryRewriteScorer(api_key=gpt5_api_key)
  
        # 权重配置
        self.weights = {
            "gpt5_score": 0.4,           # GPT-5评分权重
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
  
        # 1. GPT-5评分奖励
        gpt5_reward = self._compute_gpt5_reward(
            original_query,
            qwen8b_rewrite,
            qwen32b_rewrite,
            context
        )
  
        # 2. 检索质量奖励
        retrieval_reward = self._compute_retrieval_reward(
            retrieval_results
        )
  
        # 3. 相对提升奖励 (8B vs 32B)
        improvement_reward = self._compute_improvement_reward(
            gpt5_reward,
            retrieval_reward,
            baseline="qwen32b"
        )
  
        # 加权求和
        total_reward = (
            self.weights["gpt5_score"] * gpt5_reward +
            self.weights["retrieval_quality"] * retrieval_reward +
            self.weights["relative_improvement"] * improvement_reward
        )
  
        # 归一化到[-1, 1]
        total_reward = np.clip(total_reward, -1.0, 1.0)
  
        return total_reward
  
    def _compute_gpt5_reward(
        self,
        original: str,
        rewrite_8b: str,
        rewrite_32b: str,
        context: Dict = None
    ) -> float:
        """GPT-5评分奖励"""
  
        # 评分8B的改写
        score_8b_dict = self.gpt5_scorer.score_rewrite(
            original,
            rewrite_8b,
            context
        )
        score_8b = score_8b_dict["综合得分"] / 5.0  # 归一化到[0, 1]
  
        # 评分32B的改写
        score_32b_dict = self.gpt5_scorer.score_rewrite(
            original,
            rewrite_32b,
            context
        )
        score_32b = score_32b_dict["综合得分"] / 5.0  # 归一化到[0, 1]
  
        # 计算相对奖励
        # 如果8B > 32B，给正奖励
        # 如果8B < 32B，给负奖励
        # 使用tanh函数平滑
        diff = score_8b - score_32b
        reward = np.tanh(diff * 2)  # 放大差异
  
        return reward
  
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
        gpt5_reward: float,
        retrieval_reward: float,
        baseline: str = "qwen32b"
    ) -> float:
        """相对提升奖励
  
        鼓励8B模型超越32B baseline
        """
  
        # 综合改进度
        improvement = (gpt5_reward + retrieval_reward) / 2
  
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

#### 3.2.1 GPT-5评分优化技巧

为了提高RL训练效率和降低成本，这里提供一些实用的优化建议：

**1. 批量异步调用**

```python
# 批量评分优化示例
class BatchedGPT5Scorer:
    """批量GPT-5评分器，优化调用效率"""
  
    def __init__(self, batch_size: int = 10, cache_enabled: bool = True):
        self.scorer = GPT5QueryRewriteScorer()
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.cache = {}  # 评分缓存
  
    async def batch_score_with_cache(
        self,
        query_pairs: List[Dict]
    ) -> List[Dict]:
        """带缓存的批量评分"""
  
        results = []
        to_score = []
        cache_hits = 0
  
        # 检查缓存
        for pair in query_pairs:
            cache_key = self._make_cache_key(pair)
            if self.cache_enabled and cache_key in self.cache:
                results.append(self.cache[cache_key])
                cache_hits += 1
            else:
                to_score.append(pair)
  
        # 批量评分未命中缓存的项
        if to_score:
            new_scores = await self.scorer.batch_score(
                to_score,
                max_concurrent=self.batch_size
            )
      
            # 更新缓存
            for pair, score in zip(to_score, new_scores):
                cache_key = self._make_cache_key(pair)
                self.cache[cache_key] = score
                results.append(score)
  
        print(f"缓存命中率: {cache_hits}/{len(query_pairs)} = {cache_hits/len(query_pairs)*100:.1f}%")
  
        return results
  
    def _make_cache_key(self, pair: Dict) -> str:
        """生成缓存键"""
        return f"{pair['original_query']}|||{pair['rewritten_query']}"
```

**2. 成本控制策略**

```python
class CostAwareScorer:
    """成本感知的评分器"""
  
    def __init__(
        self,
        daily_budget: float = 50.0,  # 每日预算($)
        cost_per_call: float = 0.01   # 每次调用成本估算
    ):
        self.daily_budget = daily_budget
        self.cost_per_call = cost_per_call
        self.daily_calls = 0
        self.total_cost = 0.0
  
    def should_score(self) -> bool:
        """检查是否应该调用评分"""
        estimated_cost = (self.daily_calls + 1) * self.cost_per_call
        return estimated_cost < self.daily_budget
  
    def score_with_fallback(
        self,
        original: str,
        rewritten: str,
        context: Dict = None
    ) -> Dict:
        """带降级策略的评分"""
  
        if self.should_score():
            # 使用GPT-5评分
            score = self.gpt5_scorer.score_rewrite(original, rewritten, context)
            self.daily_calls += 1
            self.total_cost += self.cost_per_call
            return score
        else:
            # 预算用完，降级到规则评分
            print(f"⚠️  达到每日预算限制，使用规则评分")
            return self._rule_based_score(original, rewritten)
  
    def _rule_based_score(self, original: str, rewritten: str) -> Dict:
        """简单的规则评分（降级方案）"""
        # 基于长度、关键词等简单规则
        score = 3.0  # 默认中等分数
  
        if len(rewritten) > len(original) * 1.5:
            score += 0.5  # 改写更详细
  
        if any(kw in rewritten for kw in ["服用", "方法", "推荐"]):
            score += 0.3  # 包含关键术语
  
        return {
            "综合得分": min(5.0, score),
            "评分理由": "规则评分（预算限制）"
        }
```

**3. 评分采样策略**

在RL训练中不是每个样本都需要GPT-5评分，可以采用采样策略：

```python
class SamplingScorer:
    """采样评分器"""
  
    def __init__(self, sampling_rate: float = 0.3):
        self.sampling_rate = sampling_rate
        self.gpt5_scorer = GPT5QueryRewriteScorer()
        self.step = 0
  
    def score_with_sampling(
        self,
        original: str,
        rewritten: str,
        retrieval_score: float,  # 使用检索分数作为代理
        context: Dict = None
    ) -> Dict:
        """基于采样的评分"""
  
        self.step += 1
  
        # 策略1: 固定比例采样
        if np.random.random() < self.sampling_rate:
            return self.gpt5_scorer.score_rewrite(original, rewritten, context)
  
        # 策略2: 对检索分数极端的样本必评（高分和低分都很重要）
        if retrieval_score > 0.9 or retrieval_score < 0.5:
            return self.gpt5_scorer.score_rewrite(original, rewritten, context)
  
        # 策略3: 其他情况使用检索分数作为代理
        proxy_score = retrieval_score * 5.0  # 映射到1-5分
        return {
            "综合得分": proxy_score,
            "评分理由": "基于检索分数的代理评分"
        }
```

**4. prompt缓存和复用**

```python
# 预编译常用的评分prompt模板
SCORING_PROMPT_TEMPLATE = """
# Query改写质量评估任务
...（省略完整prompt）...
"""

# 使用时只需填充变量，减少token使用
def build_efficient_prompt(original: str, rewritten: str) -> str:
    return SCORING_PROMPT_TEMPLATE.format(
        original_query=original,
        rewritten_query=rewritten
    )
```

**预期优化效果**：

- 通过缓存：减少 **40-60%** 的重复评分调用
- 通过采样：降低 **70%** 的评分成本
- 通过批量调用：提高 **3-5倍** 的吞吐量

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
  
        # 3. 并行检索两个改写的结果（实时调用RAG API）
        retrieval_results = await self._parallel_retrieval(
            rewrite_8b=qwen8b_rewrite,
            rewrite_32b=qwen32b_rewrite,
            original_query=original_query,
            context=context
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
        rewrite_32b: str,
        original_query: str = None,
        context: Dict = None
    ) -> Dict:
        """并行检索两个改写的结果
    
        🔥 实时调用RAG框架API进行检索
        """
  
        import httpx
  
        # 并行调用RAG API
        results_8b_task = asyncio.create_task(
            self._call_rag_api(rewrite_8b, original_query, context)
        )
        results_32b_task = asyncio.create_task(
            self._call_rag_api(rewrite_32b, original_query, context)
        )
  
        results_8b = await results_8b_task
        results_32b = await results_32b_task
  
        return {
            "qwen_8b_results": results_8b,
            "qwen_32b_results": results_32b
        }
  
    async def _call_rag_api(
        self,
        rewritten_query: str,
        original_query: str = None,
        context: Dict = None
    ) -> List[Dict]:
        """调用RAG框架API进行实时检索
    
        直接调用general_rag路由，获取真实的检索结果
        """
  
        import httpx
    
        payload = {
            "query": original_query or rewritten_query,  # 原始query
            "tenant_id": self.tenant_id,
            "kb_name": "default",
            "history": context.get("history_context", "") if context else "",
            "top_k": 5,
            "score_threshold": 0.5,
            # 关键：直接传入改写后的query用于检索
            "rewritten_query": rewritten_query
        }
    
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8000/api/chat/general_rag",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
                # 提取检索结果
                data = result.get("data", {})
                recall_results = data.get("recall", [])
            
                return recall_results
    
        except Exception as e:
            print(f"⚠️  RAG API调用失败: {e}")
            return []
  
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

Step 2: 评分模型配置 (半天)
├─ 配置GPT-5 API密钥
├─ 设计评分prompt模板
└─ 测试评分稳定性和准确性

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

---

## 5️⃣ 实时RL训练与RAG集成

### 5.1 为什么需要实时RAG集成？

在RL训练阶段，我们需要实时获取检索结果来计算奖励，这与离线SFT训练不同：

**SFT阶段（离线）**：

- 使用预先收集的训练数据
- 数据中已包含改写结果和检索效果
- 一次性准备，反复使用

**RL阶段（实时）**：

- 8B模型在每个训练步生成**新的**改写query
- 需要**实时调用RAG系统**检索，获取检索效果
- 基于检索效果计算奖励，指导模型学习

### 5.2 实时集成架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        RL训练循环                                   │
│                                                                    │
│  1. 读取训练样本                                                    │
│     ├─ original_query: "胶原蛋白怎么吃"                             │
│     ├─ user_profile                                                │
│     └─ history_context                                             │
│                                                                    │
│  2. 双模型并行生成改写                                               │
│     ├─ 8B生成: "胶原蛋白肽 服用方法 用量"                            │
│     └─ 32B生成: "胶原蛋白肽 服用方法 推荐用量 适用人群"               │
│                                                                    │
│  3. 🔥 实时调用RAG API（并行）                                       │
│     ├─ POST /api/chat/general_rag                                  │
│     │   ├─ query: original_query                                   │
│     │   ├─ rewritten_query: 8B改写                                  │
│     │   └─ tenant_id, kb_name, top_k...                            │
│     │   → 返回: recall_results_8b (包含reranker_score)             │
│     │                                                              │
│     └─ POST /api/chat/general_rag                                  │
│         ├─ query: original_query                                   │
│         ├─ rewritten_query: 32B改写                                 │
│         └─ tenant_id, kb_name, top_k...                            │
│         → 返回: recall_results_32b (包含reranker_score)           │
│                                                                    │
│  4. 计算多维度奖励                                                   │
│     ├─ GPT-5评分: 评估改写质量                                       │
│     ├─ 检索质量: 基于reranker_score                                 │
│     └─ 相对提升: 8B vs 32B                                          │
│                                                                    │
│  5. PPO更新策略                                                      │
│     └─ 根据奖励优化8B模型参数                                         │
│                                                                    │
│  6. 重复直到收敛                                                     │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 关键实现细节

#### 修改general_rag_routes.py支持外部改写

需要在RAG API中添加一个参数，允许外部传入改写后的query：

```python
# general_rag_routes.py

@router.post("/api/chat/general_rag")
async def general_rag_endpoint(
    query: str,
    tenant_id: str,
    kb_name: str = "default",
    history: List[Dict] = [],
    top_k: int = 5,
    score_threshold: float = 0.5,
    rewritten_query: Optional[str] = None  # 🆕 新增参数
):
    """General RAG接口
  
    Args:
        rewritten_query: 可选，外部传入的改写query
                        如果提供，则跳过内部query改写步骤
    """
  
    # 如果外部提供了改写query，直接使用
    if rewritten_query:
        new_query = rewritten_query
        logger.info(f"使用外部改写query: {new_query}")
    else:
        # 否则使用原有的改写逻辑
        new_query = await rewrite_query_by_model(
            query, history, thought_unit, tenant_id, contact_id
        )
  
    # 后续检索流程保持不变
    search_res = await rag_workflow(
        new_query, history_str, tu, tenant_id, contact_id, kb_name,
        top_k, score_threshold
    )
  
    return {
        "data": {
            "user_profile": user_profile or "",
            "history_summary": history_summary or "",
            "rewritten_query": new_query,
            "recall": search_res
        },
        "status": RAGResponseStatus.success
    }
```

#### RL训练中的并发控制

由于每个训练step都需要调用2次RAG API（8B和32B各一次），需要合理控制并发：

```python
# rl_trainer_config.yaml

retrieval:
  rag_api_url: "http://localhost:8000/api/chat/general_rag"
  max_concurrent_requests: 10  # 最大并发请求数
  timeout: 30  # 超时时间(秒)
  retry_times: 2  # 失败重试次数
  
  # 批处理设置
  batch_size: 32  # 每批处理32个样本
  # 每批需要 32 * 2 = 64次RAG调用
  # 如果max_concurrent=10，则需要约6-7轮并发调用
```

### 5.4 性能优化策略

#### 1. 批量处理

```python
async def batch_process_episodes(
    self,
    samples: List[Dict],
    batch_size: int = 32
):
    """批量处理训练样本"""
  
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
    
        # 并行处理整个batch
        episodes = await asyncio.gather(*[
            self.generate_training_episode(
                sample['original_query'],
                sample['context']
            )
            for sample in batch
        ])
    
        # 批量计算奖励和更新模型
        yield episodes
```

#### 2. 结果缓存

对于相同的改写query，可以缓存检索结果：

```python
class RetrievalCache:
    """检索结果缓存"""
  
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
  
    def get(self, query: str, tenant_id: str) -> Optional[List[Dict]]:
        """获取缓存"""
        key = f"{tenant_id}:{query}"
        return self.cache.get(key)
  
    def set(self, query: str, tenant_id: str, results: List[Dict]):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 删除最旧的缓存
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
        key = f"{tenant_id}:{query}"
        self.cache[key] = results
```

**预期缓存效果**：

- 训练初期：缓存命中率低（~5-10%）
- 训练中期：缓存命中率提升（~20-30%）
- 减少30%的RAG API调用

#### 3. 超时和降级策略

```python
async def _call_rag_api_with_fallback(
    self,
    rewritten_query: str,
    original_query: str,
    context: Dict
) -> List[Dict]:
    """带降级的RAG API调用"""
  
    try:
        # 首先尝试调用RAG API
        async with asyncio.timeout(30):  # 30秒超时
            return await self._call_rag_api(
                rewritten_query, original_query, context
            )
  
    except asyncio.TimeoutError:
        logger.warning(f"RAG API超时，使用缓存或默认结果")
    
        # 降级策略1：使用缓存
        cached_result = self.retrieval_cache.get(rewritten_query, self.tenant_id)
        if cached_result:
            return cached_result
    
        # 降级策略2：返回空结果，使用默认奖励
        logger.warning("使用默认奖励")
        return []
  
    except Exception as e:
        logger.error(f"RAG API调用失败: {e}")
        return []
```

### 5.5 训练监控指标

在WandB中实时监控以下指标：

```python
# 每个epoch记录
wandb.log({
    # 奖励相关
    "avg_reward": avg_reward,
    "avg_gpt5_score": avg_gpt5_score,
    "avg_retrieval_quality": avg_retrieval_quality,
  
    # 检索效果
    "8b_avg_top1_score": avg_8b_top1,
    "32b_avg_top1_score": avg_32b_top1,
    "8b_win_rate": 8b_wins / total_samples,  # 8B超过32B的比例
  
    # 性能指标
    "rag_api_calls": total_rag_calls,
    "cache_hit_rate": cache_hits / total_rag_calls,
    "avg_rag_latency": avg_rag_latency,
  
    # 训练指标
    "policy_loss": policy_loss,
    "value_loss": value_loss
})
```

### 5.6 快速启动实时RL训练

```bash
# 1. 确保RAG服务运行
# 在一个终端启动RAG服务
cd sales-rag
python startup.py -a

# 2. 在另一个终端启动RL训练
cd code
python -m verl.trainer.main_ppo \
    --config config/sales_rag_rl_config.yaml \
    --rag_api_url http://localhost:8000/api/chat/general_rag \
    --enable_realtime_retrieval true \
    --max_concurrent_rag_calls 10

# 训练会自动：
# - 从8B模型生成改写
# - 实时调用RAG API获取检索结果
# - 计算奖励并更新模型
```

### 5.7 常见问题

**Q1: RAG API调用太慢，影响训练速度怎么办？**

- A: 使用批量+并发处理，设置合理的 `max_concurrent_requests`
- A: 启用检索结果缓存
- A: 考虑在同一机器上运行RAG服务，减少网络延迟

**Q2: RAG服务偶尔超时或失败怎么办？**

- A: 实现重试机制和降级策略
- A: 超时的样本使用缓存结果或跳过
- A: 记录失败日志，定期分析原因

**Q3: 如何确保8B和32B的改写都使用相同的检索环境？**

- A: 固定 `kb_name`、`top_k`、`score_threshold`等参数
- A: 使用相同的tenant_id和knowledge base
- A: 确保reranker等组件稳定运行

---

## 📊 预期效果

| 指标            | Baseline (32B) | SFT (8B) | RL (8B) |
| --------------- | -------------- | -------- | ------- |
| 改写质量评分    | 4.2/5          | 3.8/5    | 4.5/5   |
| 检索Top-1准确率 | 78%            | 72%      | 85%     |
| 推理延迟        | 850ms          | 320ms    | 350ms   |
| 成本/1000次     | $2.50 | $0.80  | $0.85    |         |

通过RL训练，预期8B模型能在保持低成本的同时，在改写质量和检索效果上**超越32B baseline**！
