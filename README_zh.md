## DeepRetrieval 项目简介（中文说明）

此文档为 DeepRetrieval 项目的中文说明，旨在帮助你快速理解代码结构、主要组件、依赖安装、运行方法以及常见问题排查。项目原文档为英文，本文件为中文导览与快速上手指南。

## 一、项目概述

DeepRetrieval 是一套将强化学习（RL）方法用于训练大语言模型（LLM）进行查询重写和检索增强的研究代码。核心思想是让模型通过与检索器交互并根据检索性能获得奖励，从而生成更能提高检索召回/准确性的查询（query rewriting）。

主要能力：
- 在真实搜索引擎或检索器上优化查询质量
- 支持基于 vLLM 的推理接口用于在线查询改写
- 包含数据下载/预处理、训练、评估与推理脚本

## 二、仓库重要路径说明

（下列路径均相对于仓库根目录）

- `code/`：核心实现目录，包含数据处理、训练框架（基于 verl）、示例、脚本等。
  - `code/src/`：项目源码（如 `query_rewrite.py` 等实用脚本）。
  - `code/data_preprocess/`：数据预处理脚本（各个数据集对应的脚本）。
  - `code/scripts/`：运行训练/评估/上线命令的 shell 脚本（在示例/训练/评估中使用）。
  - `code/docs/`：文档构建相关内容（仅限开发者查看或构建 HTML 文档）。
  - `code/requirements.txt`：训练与运行所需 Python 依赖（核心依赖列表）。
  - `code/setup.py`：可将 `code` 目录安装为包（`pip install -e .`）。

- `query_rewrite.py`（项目根或 `code/src/query_rewrite.py`）：提供了一个通过 vLLM API 调用模型进行查询改写的示例脚本，便于快速尝试查询改写功能。

- `vllm_host.sh`：用于在本地启动 vLLM 推理服务器的脚本（示例）。

## 三、关键文件解读（精要）

- `code/src/query_rewrite.py`
  - 作用：使用 vLLM（或兼容的 Chat API）发送具有特定模版（包含 `<think>` 与 `<answer>` 的结构化提示），接收模型输出，并解析 JSON 中的 `query` 字段作为改写结果。
  - 工作流程：构造 prompt → 请求本地 vLLM API → 从模型输出中抽取 `<answer>` 标签内的 JSON → 打印原始与改写后查询。
  - 常用参数：`--query`（要改写的查询），`--api_url`（vLLM 服务地址，默认 `http://localhost:8000/v1/chat/completions`）。

- `code/requirements.txt`
  - 包含训练/运行所需的依赖，如：`vllm`, `transformers`, `ray`, `wandb`, `flash-attn` 等。建议按需安装并优先使用与 README 中推荐的版本。

- `code/setup.py`
  - 用于将 `code` 目录中的 `verl` 包安装到环境中（`pip install -e .`），有助于运行训练与评估脚本。

## 四、快速安装（在 Windows PowerShell 下示例）

建议使用 Conda 创建隔离环境（Linux/Windows 通用）：

```powershell
# 创建并激活 conda 环境（示例）
conda create -n deepretrieval python=3.9 -y; conda activate deepretrieval

# 进入 code 目录并安装依赖
cd code; pip install -r requirements.txt; pip install -e .

# 安装 vllm（请参考系统 GPU/CPU 环境选择合适版本）
pip install vllm==0.6.3

# 一些可选工具
pip install wandb IPython matplotlib huggingface_hub
```

注意：一些依赖（如 `faiss-gpu`、`flash-attn`）在 Windows 上安装可能更复杂，建议在 Linux/服务器环境按 README 指南安装 GPU 支持的包，或在 Windows 使用 CPU 版本并留意兼容性问题。

## 五、快速运行示例：查询改写（Query Rewriting）

步骤概览：先运行 vLLM 服务，再调用 `query_rewrite.py`。示例（PowerShell）：

```powershell
# 在类 Unix 环境：sh vllm_host.sh
# Windows 下可按需手动启动 vLLM 服务，或在 WSL 中执行 vllm_host.sh

# 使用本地 vLLM 服务改写查询
python code/src/query_rewrite.py --query "Who built DeepRetrieval in 2025?"

# 输出示例：
# Original query: Who built DeepRetrieval in 2025?
# Rewritten query: (The DeepRetrieval system, which was built in 2025)
```

如果你无法或不想本地启动 vLLM，请将 `--api_url` 指向兼容的 Chat API 服务（例如内部 vLLM 或 OpenAI-like 接口），但请确保返回格式包含 `choices[0].message.content` 并且内容中含 `<answer>{...}</answer>` 的 JSON 字段。

## 六、数据准备与预处理

- `code/download_datasets.py`：从 Hugging Face 或指定存储下载预处理/原始数据（支持列出与指定下载）。
- `code/data_preprocess/`：包含针对不同数据集的预处理脚本（例如 `pubmed.py`、`msmarco_beir.py` 等），用于生成训练/评估所需格式。

推荐流程：
1. 在 `code/` 下运行 `python download_datasets.py --list_only --repo_id DeepRetrieval/datasets` 查看可用数据集
2. 使用 `--output_dir` 参数下载特定数据集
3. 运行对应预处理脚本（例如 `python code/data_preprocess/pubmed_32.py`）来生成训练数据

## 七、训练与评估（概要）

- 训练脚本位置：`code/scripts/train/`（包含针对不同任务的 shell 脚本）。
- 评估脚本位置：`code/scripts/eval/`。
- 配置与 trainer 逻辑主要基于 `code/verl/`（项目中集成的 verl 框架），训练过程中会使用 RL 奖励函数来计算检索效果并作为奖励信号。

示例（在 Linux/服务器环境）:

```bash
# 激活 conda 环境
conda activate deepretrieval
# 运行示例训练脚本（以 pubmed 为例）
sh code/scripts/train/pubmed_32.sh
```

## 八、日志与监控

- 项目使用 `wandb` 记录训练日志（如果已配置）。在训练脚本中通常会写入 wandb 项目信息。

## 九、常见问题与排查建议

- vLLM/模型服务无法连接：确认 `--api_url` 正确，模型服务正在运行且返回结构符合预期。
- CUDA/显存错误：检查 CUDA 驱动/模块是否正确加载，或修改训练脚本降低显存使用（如启用 gradient checkpointing）。
- 依赖冲突：按 `code/requirements.txt` 及顶部 README 中建议的版本安装，必要时降级 `numpy` 到 `<2.0.0` 等兼容版本。

## 十、开发者提示

- 若要调试改写策略，可先本地调用 `code/src/query_rewrite.py` 验证 prompt 模版与模型输出格式。
- 若修改 reward 逻辑或 trainer 配置，相关代码主要位于 `code/verl/trainer/` 下。

## 十一、参考与引用

- 请参阅仓库顶层 `README.md` 获取更详细的实验结果、安装说明与论文引用信息（英文）。
- 引用论文：Jiang et al., "DeepRetrieval: Hacking real search engines and retrievers with large language models via reinforcement learning" (arXiv:2503.00223)

---

如果你希望我把 README_zh.md 调整为更精简的快速上手版、或生成一个中文版的开发者文档（按模块分文件），我可以继续拆分并补充具体命令与示例。现在我会把任务标记为已完成并显示接下来的建议步骤。

## 十二、项目框架深入分析与使用说明

下面按照模块化的方式介绍项目各个子系统、它们的职责以及如何常用它们来完成数据准备、训练、评估与推理。

1) 核心代码组织（`code/`）

- `code/verl/`：项目集成并改造的 veRL（Volcano Engine Reinforcement Learning）框架，包含训练器、分布式、模型封装、reward 计算、工具函数等。
  - `code/verl/trainer/`：trainer 入口和训练流程文件。
    - `main_ppo.py`：基于 PPO 的训练入口（强化学习主流程）。
    - `main_generation.py`：用于评估或生成输出的入口脚本（inference/generation）。
    - `main_eval.py`：评估入口（跑评估脚本并计算召回/指标）。
    - `fsdp_sft_trainer.py`：用于 FSDP（分布式）SFT 的训练脚本。
    - `config/`：trainer 相关的 YAML 配置文件（策略、优化器、模型、环境等）。
  - `code/verl/utils/`：大量实用工具（模型加载、分布式、IO、日志、reward、tokenizer 等），在修改训练逻辑或排查错误时会频繁查看。

- `code/src/`：轻量脚本与工具，例如 `query_rewrite.py`，可用于快速测试推理接口或作为 demo（非训练主流程）。

- `code/data_preprocess/`：数据处理脚本。每个数据集通常有对应的处理脚本，输出统一的训练/评估数据格式（例如 JSON/TSV 或特定目录结构），训练脚本会读取这些目录。

- `code/scripts/`：封装常用训练和评估命令的 shell 脚本。它们通常负责设置环境变量、模型与数据路径、以及调用 `python code/verl/trainer/main_*.py` 等入口脚本。

2) 常用运行流程（建议顺序）

- 数据准备：
  1. 下载或准备原始数据：`python code/download_datasets.py --repo_id DeepRetrieval/datasets --output_dir ./data` 或按需下载单个数据集。
  2. 运行对应预处理脚本：例如 `python code/data_preprocess/pubmed_32.py`，生成训练/评估所需文件结构。

- 快速评估/推理（不训练）：
  1. 启动 vLLM 或兼容 Model API（`vllm_host.sh` 或本地部署的 chat completion 接口）。
  2. 使用 `python code/src/query_rewrite.py --query "..."` 进行单条查询改写测试。

- 训练（PPO 强化学习主流程）：
  1. 准备数据与奖励接口（例如检索器或真实搜索引擎 API key）。
  2. 在 `code/scripts/train/` 选择对应的脚本（例如 `pubmed_32.sh`），脚本会设置配置并调用 `main_ppo.py`。
  3. 监控 wandb（如果启用）或本地日志来观察训练曲线。

- 评估：
  1. 使用 `code/scripts/eval/` 下的脚本运行评估（例如 `sh code/scripts/eval/pubmed_32.sh`），或直接调用 `main_eval.py` 并传入 checkpoint。

3) 配置说明（在哪里改什么）

- 模型与训练参数：位于 `code/verl/trainer/config/` 的 YAML 文件中，或训练脚本中通过命令行覆盖（configs 通常描述 lr、batch、模型路径、分布式参数等）。
- 奖励函数：位于 `code/verl/utils/reward_score/`（以及 `reward_score_dense/`），这里实现了对检索结果的打分逻辑。若需要修改 reward 设计，请在此处更改并测试效果。
- 日志与跟踪：跟 wandb 集成的代码通常在 `code/verl/utils/tracking.py` 或 trainer 的 log 初始化处，修改或关闭 wandb 可在训练脚本中调整对应参数。

4) 常见命令示例（Linux/WSL/PowerShell 可参考）

```powershell
# 安装并进入环境（PowerShell）
conda create -n deepretrieval python=3.9 -y; conda activate deepretrieval
cd code; pip install -r requirements.txt; pip install -e .

# 单条查询改写（确保 vLLM 服务已启动）
python code/src/query_rewrite.py --query "diagnosis of diabetes"

# 运行训练脚本（示例，Linux/WSL）
sh code/scripts/train/pubmed_32.sh

# 运行评估脚本（示例）
sh code/scripts/eval/pubmed_32.sh
```

5) 调试建议

- 若报错指向 `verl` 的接口或 utils，优先检查 `code/verl/utils/` 下的实现（例如 `model.py`, `distributed.py` 等）。
- 如果发现训练任务直接 OOM（显存溢出），尝试：
  - 降低单卡 batch_size 或 seq_len
  - 在 config 中开启 gradient checkpointing
  - 使用 FSDP 或增加 GPU 数量

6) 扩展开发建议

- 增加新的数据集：在 `code/data_preprocess/` 添加对应处理脚本，并在 `download_datasets.py` 或文档中说明数据结构。
- 修改 reward：在 `code/verl/utils/reward_score/` 中实现新的 reward 逻辑，同时在 `main_ppo.py` 或对应配置中启用。
- 新模型/推理后端：`code/src/query_rewrite.py` 实现了一个通用的 Chat API 调用结构，你可以基于该模式替换为其他推理后端。

---

我已经把这些扩展内容写入 `README_zh.md`。任务状态接下来我会标记为完成。若你要我把这些章节拆成独立文档文件（例如 `docs/zh/architecture.md`、`docs/zh/usage.md`），我可以继续自动拆分并创建目录结构。 

## 十三、基于本项目进行自定义微调的完整指南

本章节专门为希望基于 DeepRetrieval 项目加入自己代码和数据进行微调的开发者准备。

### 13.1 微调流程概览

```
准备数据 → 编写预处理脚本 → 配置奖励函数 → 创建训练脚本 → 执行训练 → 评估与部署
```

### 13.2 详细步骤说明

#### 步骤 1：准备你的数据集

**目标**：准备符合你业务场景的查询-文档对数据。

**操作**：
1. 收集你的原始数据，通常包含：
   - 查询集合（queries）
   - 文档库（corpus/documents）
   - 相关性标注（可选，用于评估）

2. 数据格式建议：
   ```json
   {
     "queries": [
       {"id": "q1", "text": "你的查询文本"},
       {"id": "q2", "text": "另一个查询"}
     ],
     "corpus": [
       {"id": "doc1", "title": "文档标题", "text": "文档内容"},
       {"id": "doc2", "title": "另一个文档", "text": "内容"}
     ],
     "qrels": {
       "q1": {"doc1": 1, "doc3": 1}
     }
   }
   ```

3. 将数据放置在项目目录下，例如 `data/my_dataset/`

#### 步骤 2：创建数据预处理脚本

**目标**：将原始数据转换为训练所需格式。

**操作**：
1. 参考现有预处理脚本模板：
   ```powershell
   # 查看现有预处理脚本作为参考
   ls code/data_preprocess/
   ```

2. 创建你的预处理脚本 `code/data_preprocess/my_dataset.py`：
   ```python
   import json
   import os
   from pathlib import Path
   
   def preprocess_my_data():
       """
       将你的数据转换为训练格式
       输出格式通常包含：
       - train.jsonl: 训练数据（每行一个query）
       - dev.jsonl: 验证数据
       - corpus.jsonl: 文档库
       """
       input_dir = Path("data/my_dataset/raw")
       output_dir = Path("data/my_dataset/processed")
       output_dir.mkdir(parents=True, exist_ok=True)
       
       # 读取原始数据
       with open(input_dir / "queries.json") as f:
           queries = json.load(f)
       
       # 转换为训练格式
       train_data = []
       for query in queries:
           train_data.append({
               "query_id": query["id"],
               "query": query["text"],
               # 添加其他必要字段
           })
       
       # 保存处理后的数据
       with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
           for item in train_data:
               f.write(json.dumps(item, ensure_ascii=False) + "\n")
       
       print(f"数据预处理完成，输出到: {output_dir}")
   
   if __name__ == "__main__":
       preprocess_my_data()
   ```

3. 运行预处理：
   ```powershell
   python code/data_preprocess/my_dataset.py
   ```

#### 步骤 3：配置或自定义奖励函数

**目标**：定义如何评估查询改写的质量（这是强化学习的核心）。

**操作**：

**方案 A：使用现有奖励函数**
- 项目已包含多种奖励函数（BM25、Dense retrieval 等）
- 位置：`code/verl/utils/reward_score/`
- 如果现有检索器符合需求，可直接使用

**方案 B：自定义奖励函数**
1. 在 `code/verl/utils/reward_score/` 创建 `my_reward.py`：
   ```python
   import numpy as np
   from typing import List, Dict
   
   class MyCustomReward:
       """自定义奖励函数"""
       
       def __init__(self, retriever_config):
           """初始化你的检索器或评分系统"""
           self.retriever = self._init_retriever(retriever_config)
       
       def compute_reward(self, original_query: str, rewritten_query: str, 
                         ground_truth_docs: List[str] = None) -> float:
           """
           计算奖励分数
           
           Args:
               original_query: 原始查询
               rewritten_query: 改写后的查询
               ground_truth_docs: 真实相关文档（可选）
           
           Returns:
               reward: 奖励分数（通常在 0-1 之间）
           """
           # 使用改写查询进行检索
           retrieved_docs = self.retriever.search(rewritten_query, top_k=10)
           
           # 计算检索质量（例如召回率、NDCG等）
           if ground_truth_docs:
               recall = self._calculate_recall(retrieved_docs, ground_truth_docs)
               return recall
           else:
               # 或者使用其他无监督指标
               return self._calculate_diversity_score(retrieved_docs)
       
       def _calculate_recall(self, retrieved, ground_truth):
           """计算召回率"""
           retrieved_ids = set([d['id'] for d in retrieved])
           gt_ids = set(ground_truth)
           return len(retrieved_ids & gt_ids) / len(gt_ids) if gt_ids else 0.0
   ```

2. 在训练配置中引用你的奖励函数（步骤 4 中说明）

#### 步骤 4：创建训练配置文件

**目标**：配置训练超参数、模型路径、数据路径等。

**操作**：
1. 在 `code/verl/trainer/config/` 创建 `my_dataset_config.yaml`：
   ```yaml
   # 模型配置
   model:
     path: "deepseek-ai/deepseek-llm-7b-chat"  # 或你的基座模型
     type: "causal_lm"
   
   # 数据配置
   data:
     train_path: "data/my_dataset/processed/train.jsonl"
     eval_path: "data/my_dataset/processed/dev.jsonl"
     corpus_path: "data/my_dataset/processed/corpus.jsonl"
   
   # PPO训练参数
   ppo:
     learning_rate: 1e-6
     batch_size: 8
     gradient_accumulation_steps: 4
     max_epochs: 3
     warmup_steps: 100
     clip_range: 0.2
   
   # 奖励函数配置
   reward:
     type: "my_custom"  # 或 "bm25", "dense" 等
     config:
       retriever_model: "your-retriever-model"
       top_k: 10
   
   # 生成配置
   generation:
     max_new_tokens: 512
     temperature: 0.7
     top_p: 0.9
   
   # 日志配置
   logging:
     wandb_project: "my-deepretrieval"
     wandb_run_name: "my-dataset-exp1"
     log_interval: 10
   ```

#### 步骤 5：创建训练脚本

**目标**：封装训练命令，方便重复运行。

**操作**：
1. 在 `code/scripts/train/` 创建 `my_dataset.sh`：
   ```bash
   #!/bin/bash
   
   # 设置环境变量
   export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据你的GPU数量调整
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   
   # 训练参数
   MODEL_PATH="deepseek-ai/deepseek-llm-7b-chat"
   DATA_DIR="data/my_dataset/processed"
   OUTPUT_DIR="outputs/my_dataset_$(date +%Y%m%d_%H%M%S)"
   CONFIG_FILE="code/verl/trainer/config/my_dataset_config.yaml"
   
   # 创建输出目录
   mkdir -p ${OUTPUT_DIR}
   
   # 启动训练
   python -m verl.trainer.main_ppo \
       --config ${CONFIG_FILE} \
       --model.path ${MODEL_PATH} \
       --data.train_path ${DATA_DIR}/train.jsonl \
       --data.eval_path ${DATA_DIR}/dev.jsonl \
       --output_dir ${OUTPUT_DIR} \
       --logging.wandb_project "my-deepretrieval" \
       --logging.wandb_run_name "exp-$(date +%Y%m%d)" \
       --ppo.learning_rate 1e-6 \
       --ppo.batch_size 8 \
       --ppo.max_epochs 3 \
       --reward.type "my_custom" \
       --num_gpus 4 \
       --gradient_checkpointing true
   
   echo "训练完成，模型保存在: ${OUTPUT_DIR}"
   ```

2. 赋予执行权限（Linux/WSL）：
   ```bash
   chmod +x code/scripts/train/my_dataset.sh
   ```

#### 步骤 6：执行训练

**操作**：
1. 确保环境已安装所有依赖
2. 启动训练：
   ```powershell
   # Linux/WSL
   sh code/scripts/train/my_dataset.sh
   
   # 或直接使用 python（Windows PowerShell）
   cd code
   python -m verl.trainer.main_ppo --config verl/trainer/config/my_dataset_config.yaml
   ```

3. 监控训练进度：
   - 查看终端输出
   - 访问 WandB dashboard（如果配置了）
   - 检查 `outputs/` 目录下的日志文件

#### 步骤 7：创建评估脚本

**目标**：评估微调后模型的性能。

**操作**：
1. 在 `code/scripts/eval/` 创建 `my_dataset_eval.sh`：
   ```bash
   #!/bin/bash
   
   # 评估参数
   MODEL_CHECKPOINT="outputs/my_dataset_20250121/checkpoint-1000"
   DATA_DIR="data/my_dataset/processed"
   OUTPUT_FILE="eval_results/my_dataset_results.json"
   
   # 运行评估
   python code/verl/trainer/main_eval.py \
       --model_path ${MODEL_CHECKPOINT} \
       --test_data ${DATA_DIR}/test.jsonl \
       --corpus_path ${DATA_DIR}/corpus.jsonl \
       --output_file ${OUTPUT_FILE} \
       --metrics "recall@10,ndcg@10,mrr" \
       --batch_size 16
   
   echo "评估完成，结果保存在: ${OUTPUT_FILE}"
   ```

2. 运行评估：
   ```bash
   sh code/scripts/eval/my_dataset_eval.sh
   ```

#### 步骤 8：使用微调后的模型进行推理

**操作**：
1. 启动 vLLM 服务加载你的微调模型：
   ```bash
   vllm serve outputs/my_dataset_20250121/checkpoint-1000 \
       --host 0.0.0.0 \
       --port 8000 \
       --gpu-memory-utilization 0.9
   ```

2. 使用查询改写脚本：
   ```powershell
   python code/src/query_rewrite.py \
       --query "你的测试查询" \
       --api_url "http://localhost:8000/v1/chat/completions"
   ```

### 13.3 关键代码修改点总结

| 修改内容 | 文件位置 | 说明 |
|---------|---------|------|
| 数据预处理逻辑 | `code/data_preprocess/my_dataset.py` | 将原始数据转为训练格式 |
| 奖励函数 | `code/verl/utils/reward_score/my_reward.py` | 定义如何评估查询质量 |
| 训练配置 | `code/verl/trainer/config/my_dataset_config.yaml` | 超参数、路径配置 |
| 训练脚本 | `code/scripts/train/my_dataset.sh` | 封装训练命令 |
| 评估脚本 | `code/scripts/eval/my_dataset_eval.sh` | 封装评估命令 |
| Prompt模板（可选） | `code/src/query_rewrite.py` | 修改查询改写的提示词 |

### 13.4 进阶定制化选项

#### A. 修改 Prompt 模板
如果你想改变模型生成查询的方式，修改 `code/src/query_rewrite.py` 中的 prompt 构建逻辑：

```python
def build_custom_prompt(query: str, domain: str = "") -> str:
    """自定义提示词模板"""
    return f"""你是一个专业的{domain}领域查询优化助手。
    
请对以下查询进行改写，使其更适合检索系统：

原始查询: {query}

请按以下格式输出:
<think>分析查询意图和关键信息...</think>
<answer>{{"query": "改写后的查询"}}</answer>
"""
```

#### B. 集成自定义检索器
在 `code/verl/utils/reward_score/` 中集成你的检索系统：

```python
class MyRetrieverIntegration:
    def __init__(self, api_endpoint, api_key):
        self.endpoint = api_endpoint
        self.api_key = api_key
    
    def search(self, query, top_k=10):
        # 调用你的检索API
        response = requests.post(
            self.endpoint,
            json={"query": query, "top_k": top_k},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["results"]
```

#### C. 多任务联合训练
如果你有多个数据集，可以在配置中指定：

```yaml
data:
  datasets:
    - path: "data/dataset1/train.jsonl"
      weight: 0.5
    - path: "data/dataset2/train.jsonl"
      weight: 0.5
  sampling_strategy: "proportional"  # 或 "uniform"
```

### 13.5 常见问题与解决方案

**Q1: 训练过程中奖励一直不增长怎么办？**
- 检查奖励函数是否正确计算
- 降低学习率（例如从 1e-6 降到 5e-7）
- 确认数据质量和标注正确性
- 尝试先用 SFT（监督微调）预热

**Q2: 显存不足（OOM）如何处理？**
- 减小 `batch_size`
- 启用 `gradient_checkpointing`
- 使用更小的模型或量化版本
- 增加 GPU 数量并启用 FSDP

**Q3: 如何调试奖励函数？**
```python
# 在训练前单独测试奖励函数
from code.verl.utils.reward_score.my_reward import MyCustomReward

reward_fn = MyCustomReward(config)
original = "test query"
rewritten = "optimized test query"
score = reward_fn.compute_reward(original, rewritten)
print(f"Reward score: {score}")
```

**Q4: 想要使用自己的基座模型？**
- 修改配置文件中的 `model.path` 为你的模型路径
- 确保模型与 transformers/vLLM 兼容
- 根据模型调整 `max_length`、`vocab_size` 等参数

### 13.6 完整示例：从零开始的微调项目

假设你要为医疗领域构建查询改写系统：

```powershell
# 1. 准备环境
conda activate deepretrieval

# 2. 准备数据
mkdir -p data/medical_qa
# 将你的医疗问答数据放入 data/medical_qa/raw/

# 3. 创建预处理脚本并运行
python code/data_preprocess/medical_qa.py

# 4. 创建配置文件
# 编辑 code/verl/trainer/config/medical_qa_config.yaml

# 5. （可选）自定义奖励函数
# 创建 code/verl/utils/reward_score/medical_reward.py

# 6. 创建训练脚本
# 编辑 code/scripts/train/medical_qa.sh

# 7. 开始训练
sh code/scripts/train/medical_qa.sh

# 8. 评估模型
sh code/scripts/eval/medical_qa_eval.sh

# 9. 部署推理
vllm serve outputs/medical_qa_final/ --port 8000
python code/src/query_rewrite.py --query "糖尿病的早期症状"
```

### 13.7 最佳实践建议

1. **版本控制**：使用 git 管理你的修改，创建新分支进行实验
2. **小规模验证**：先用小数据集（100-1000条）验证流程，再扩展到全量数据
3. **日志记录**：充分利用 WandB 或 TensorBoard 记录实验
4. **超参数搜索**：系统地尝试不同学习率、batch size 组合
5. **定期保存检查点**：设置 `save_steps` 避免训练中断导致损失
6. **A/B 测试**：在真实场景中对比改写前后的检索效果

---

通过以上步骤，你应该能够完整地基于 DeepRetrieval 项目进行自定义微调。如有具体问题，可参考项目 issue 或相关论文。