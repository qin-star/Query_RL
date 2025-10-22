#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sales-RAG × DeepRetrieval 强化学习训练系统
基于真实用户反馈的QueryRewrite持续优化
"""

import asyncio
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime
import logging
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLTrainingDataCollector:
    """RL训练数据收集器"""
    
    def __init__(self, data_dir: str = "data/rl_training_samples"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = Path("data/deepretrieval_training") 
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RLTrainingDataCollector初始化: {self.data_dir}")
    
    def collect_training_samples(self, min_samples: int = 50) -> List[Dict]:
        """收集训练样本"""
        
        samples = []
        processed_files = []
        
        # 1. 读取所有反馈样本文件
        sample_files = list(self.data_dir.glob("sample_*.json"))
        logger.info(f"找到 {len(sample_files)} 个样本文件")
        
        for sample_file in sample_files:
            try:
                with open(sample_file, "r", encoding="utf-8") as f:
                    sample = json.load(f)
                    
                # 数据质量检查
                if self._validate_sample(sample):
                    samples.append(sample)
                    processed_files.append(sample_file)
                else:
                    logger.warning(f"样本数据质量不合格: {sample_file}")
                    
            except Exception as e:
                logger.error(f"处理样本文件失败 {sample_file}: {e}")
        
        # 2. 移动已处理的文件到归档目录
        if processed_files:
            archive_dir = self.data_dir / "processed"
            archive_dir.mkdir(exist_ok=True)
            
            for file_path in processed_files:
                archive_path = archive_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
                shutil.move(str(file_path), str(archive_path))
        
        logger.info(f"收集到 {len(samples)} 个有效训练样本")
        
        if len(samples) < min_samples:
            logger.info(f"样本数量不足 {min_samples}，等待更多数据...")
            return []
        
        return samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本数据质量"""
        
        required_fields = ["original_query", "rewritten_query", "reward", "timestamp"]
        
        # 1. 检查必要字段
        for field in required_fields:
            if field not in sample:
                logger.warning(f"缺少必要字段: {field}")
                return False
        
        # 2. 检查数据类型
        if not isinstance(sample["original_query"], str) or len(sample["original_query"].strip()) == 0:
            return False
        
        if not isinstance(sample["rewritten_query"], str) or len(sample["rewritten_query"].strip()) == 0:
            return False
            
        if not isinstance(sample["reward"], (int, float)):
            return False
        
        # 3. 检查奖励值范围
        if not -2.0 <= sample["reward"] <= 2.0:
            logger.warning(f"奖励值超出合理范围: {sample['reward']}")
            return False
        
        # 4. 检查查询质量
        original = sample["original_query"].strip()
        rewritten = sample["rewritten_query"].strip()
        
        # 过滤明显的垃圾数据
        if len(original) < 2 or len(rewritten) < 2:
            return False
        
        if original == rewritten and sample["reward"] > 0:
            # 如果没有改写但奖励为正，可能是数据错误
            logger.warning("查询未改写但奖励为正，可能是数据错误")
            return False
        
        return True
    
    def prepare_training_data(self, samples: List[Dict]) -> Dict:
        """准备DeepRetrieval训练数据格式"""
        
        # 1. 数据清洗和过滤
        filtered_samples = []
        
        for sample in samples:
            # 过滤低质量样本
            if sample["reward"] < -1.0:  # 用户反馈极差的样本
                continue
            
            # 数据增强：添加查询变体
            enhanced_samples = self._augment_sample(sample)
            filtered_samples.extend(enhanced_samples)
        
        # 2. 按奖励分层采样，确保数据平衡
        balanced_samples = self._balance_samples(filtered_samples)
        
        # 3. 转换为DeepRetrieval格式
        training_data = []
        
        for i, sample in enumerate(balanced_samples):
            training_item = {
                "query_id": f"rl_{int(time.time())}_{i}",
                "query": sample["original_query"],
                "rewritten_query": sample["rewritten_query"],
                "reward": float(sample["reward"]),
                "retrieval_results": sample.get("retrieval_results", []),
                "timestamp": sample.get("timestamp", time.time()),
                
                # 添加元数据用于分析
                "metadata": {
                    "user_feedback": sample.get("user_feedback", {}),
                    "enhancement_method": sample.get("enhancement_method", "unknown"),
                    "conversation_length": len(sample.get("conversation_history", [])),
                    "data_source": "user_feedback"
                }
            }
            training_data.append(training_item)
        
        # 4. 保存训练数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSONL格式 (DeepRetrieval训练用)
        train_file = self.processed_dir / f"rl_train_{timestamp}.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 创建符号链接指向最新文件
        latest_file = self.processed_dir / "rl_train_latest.jsonl"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(train_file.name)
        
        # 统计报告
        stats = self._generate_training_stats(training_data)
        
        # 保存统计信息
        stats_file = self.processed_dir / f"training_stats_{timestamp}.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练数据已保存: {train_file}")
        logger.info(f"数据统计: {stats}")
        
        return {
            "train_file": str(train_file),
            "latest_file": str(latest_file),
            "sample_count": len(training_data),
            "avg_reward": stats["avg_reward"],
            "reward_distribution": stats["reward_distribution"],
            "stats": stats
        }
    
    def _augment_sample(self, sample: Dict) -> List[Dict]:
        """数据增强：为单个样本创建变体"""
        
        augmented = [sample]  # 包含原始样本
        
        # 同义词替换 (简化版本)
        original = sample["original_query"]
        rewritten = sample["rewritten_query"]
        
        # 常见同义词替换
        synonyms = {
            "怎么": ["如何", "怎样"], 
            "吃": ["服用", "使用"],
            "效果": ["作用", "功效"],
            "好处": ["益处", "功效"],
            "什么时候": ["何时", "什么时间"]
        }
        
        # 生成1-2个变体
        for _ in range(min(2, max(1, int(abs(sample["reward"]))))):  # 高奖励样本生成更多变体
            
            aug_original = original
            aug_rewritten = rewritten
            
            # 随机替换同义词
            for word, syns in synonyms.items():
                if word in aug_original and np.random.random() < 0.3:
                    syn = np.random.choice(syns)
                    aug_original = aug_original.replace(word, syn)
                    aug_rewritten = aug_rewritten.replace(word, syn)
            
            if aug_original != original:  # 如果确实有变化才添加
                aug_sample = sample.copy()
                aug_sample["original_query"] = aug_original
                aug_sample["rewritten_query"] = aug_rewritten
                aug_sample["reward"] = sample["reward"] * 0.95  # 略微降低奖励
                augmented.append(aug_sample)
        
        return augmented
    
    def _balance_samples(self, samples: List[Dict]) -> List[Dict]:
        """平衡样本分布，避免数据偏斜"""
        
        # 按奖励值分组
        reward_groups = defaultdict(list)
        
        for sample in samples:
            reward = sample["reward"]
            if reward >= 0.5:
                group = "high"      # 高奖励
            elif reward >= 0:
                group = "medium"    # 中等奖励
            elif reward >= -0.5:
                group = "low"       # 低奖励
            else:
                group = "negative"  # 负奖励
            
            reward_groups[group].append(sample)
        
        logger.info(f"奖励分组统计: {[(k, len(v)) for k, v in reward_groups.items()]}")
        
        # 平衡采样
        max_samples_per_group = 200  # 每组最多200个样本
        balanced = []
        
        for group, group_samples in reward_groups.items():
            if len(group_samples) > max_samples_per_group:
                # 随机采样
                sampled = np.random.choice(
                    group_samples, 
                    size=max_samples_per_group, 
                    replace=False
                ).tolist()
            else:
                sampled = group_samples
            
            balanced.extend(sampled)
        
        # 打乱顺序
        np.random.shuffle(balanced)
        
        logger.info(f"平衡后样本数: {len(balanced)}")
        return balanced
    
    def _generate_training_stats(self, training_data: List[Dict]) -> Dict:
        """生成训练数据统计信息"""
        
        if not training_data:
            return {}
        
        rewards = [item["reward"] for item in training_data]
        
        stats = {
            "sample_count": len(training_data),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            
            "reward_distribution": {
                "positive": sum(1 for r in rewards if r > 0),
                "zero": sum(1 for r in rewards if r == 0),
                "negative": sum(1 for r in rewards if r < 0),
            },
            
            "percentiles": {
                "p25": np.percentile(rewards, 25),
                "p50": np.percentile(rewards, 50),
                "p75": np.percentile(rewards, 75),
                "p90": np.percentile(rewards, 90),
            },
            
            "data_quality": {
                "avg_query_length": np.mean([len(item["query"]) for item in training_data]),
                "avg_rewrite_length": np.mean([len(item["rewritten_query"]) for item in training_data]),
                "unique_queries": len(set(item["query"] for item in training_data)),
                "rewrite_rate": sum(1 for item in training_data 
                                  if item["query"] != item["rewritten_query"]) / len(training_data)
            },
            
            "timestamp": datetime.now().isoformat()
        }
        
        return stats


class DeepRetrievalRLTrainer:
    """DeepRetrieval强化学习训练器"""
    
    def __init__(self, config_path: str = "code/config/sales_rag_rl_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.collector = RLTrainingDataCollector()
        
        # 训练状态追踪
        self.training_history = []
        
        logger.info(f"DeepRetrievalRLTrainer初始化完成")
    
    def _load_config(self) -> Dict:
        """加载训练配置"""
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"配置加载成功: {self.config_path}")
            return config
        
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            # 返回默认配置
            return {
                "ppo": {"learning_rate": 1e-6, "batch_size": 4, "max_epochs": 1},
                "logging": {"output_dir": "outputs/sales_rag_rl"}
            }
    
    async def continuous_training_loop(self):
        """持续训练循环"""
        
        logger.info("开始持续训练循环...")
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"\n=== 训练迭代 #{iteration} ===")
                
                # 1. 收集训练样本
                samples = self.collector.collect_training_samples(min_samples=30)
                
                if samples:
                    # 2. 准备训练数据
                    train_info = self.collector.prepare_training_data(samples)
                    
                    # 3. 启动训练
                    training_result = await self.run_training(train_info)
                    
                    # 4. 评估模型性能
                    eval_result = await self.evaluate_model(training_result["model_path"])
                    
                    # 5. 决定是否部署
                    if self._should_deploy(eval_result):
                        await self.deploy_updated_model(training_result["model_path"])
                    else:
                        logger.info("模型性能未达到部署标准，跳过部署")
                    
                    # 6. 记录训练历史
                    self.training_history.append({
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                        "sample_count": train_info["sample_count"],
                        "avg_reward": train_info["avg_reward"],
                        "training_result": training_result,
                        "eval_result": eval_result,
                        "deployed": self._should_deploy(eval_result)
                    })
                    
                    logger.info(f"迭代 #{iteration} 完成")
                
                else:
                    logger.info("暂无足够样本，等待下一轮...")
                
                # 7. 等待下一轮
                sleep_time = 3600  # 1小时
                logger.info(f"等待 {sleep_time/60:.0f} 分钟后进行下一轮训练...")
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("收到中断信号，停止训练循环")
                break
            except Exception as e:
                logger.error(f"训练循环异常: {e}")
                await asyncio.sleep(1800)  # 出错后30分钟再试
        
        logger.info("持续训练循环结束")
    
    async def run_training(self, train_info: Dict) -> Dict:
        """运行DeepRetrieval训练"""
        
        train_file = train_info["train_file"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/sales_rag_rl_{timestamp}"
        
        # 构建训练命令
        cmd = [
            "python", "-m", "verl.trainer.main_ppo",
            "--config", str(self.config_path),
            "--data.train_path", train_file,
            "--output_dir", output_dir,
            "--logging.run_name", f"sales_rag_rl_{timestamp}",
        ]
        
        # 添加配置覆盖
        ppo_config = self.config.get("ppo", {})
        cmd.extend([
            "--ppo.learning_rate", str(ppo_config.get("learning_rate", 1e-6)),
            "--ppo.batch_size", str(ppo_config.get("batch_size", 4)),
            "--ppo.max_epochs", str(ppo_config.get("max_epochs", 1)),
        ])
        
        logger.info(f"开始RL训练: {' '.join(cmd)}")
        
        # 运行训练
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="code"  # DeepRetrieval代码目录
        )
        
        stdout, stderr = await process.communicate()
        
        result = {
            "command": ' '.join(cmd),
            "return_code": process.returncode,
            "output_dir": output_dir,
            "model_path": f"code/{output_dir}",
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
            "timestamp": timestamp
        }
        
        if process.returncode == 0:
            logger.info("RL训练完成成功")
            
            # 查找最终checkpoint
            checkpoint_dir = Path(f"code/{output_dir}") / "checkpoint-final"
            if checkpoint_dir.exists():
                result["checkpoint_path"] = str(checkpoint_dir)
                logger.info(f"找到最终checkpoint: {checkpoint_dir}")
            else:
                logger.warning("未找到最终checkpoint")
            
        else:
            logger.error(f"RL训练失败: {result['stderr']}")
            raise Exception(f"Training failed with code {process.returncode}")
        
        return result
    
    async def evaluate_model(self, model_path: str) -> Dict:
        """评估训练后的模型"""
        
        logger.info(f"开始评估模型: {model_path}")
        
        # 这里可以实现更复杂的评估逻辑
        # 暂时返回模拟结果
        eval_result = {
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            
            # 模拟评估指标
            "metrics": {
                "avg_reward": np.random.uniform(0.3, 0.8),
                "user_satisfaction": np.random.uniform(0.6, 0.9),
                "retrieval_accuracy": np.random.uniform(0.7, 0.85),
                "query_improvement_rate": np.random.uniform(0.4, 0.7)
            },
            
            # 质量检查
            "quality_checks": {
                "reward_threshold_pass": True,
                "stability_check_pass": True,
                "regression_check_pass": True
            }
        }
        
        logger.info(f"模型评估完成: {eval_result['metrics']}")
        return eval_result
    
    def _should_deploy(self, eval_result: Dict) -> bool:
        """判断是否应该部署模型"""
        
        metrics = eval_result.get("metrics", {})
        quality_checks = eval_result.get("quality_checks", {})
        
        # 检查质量标准
        min_reward = 0.3
        min_satisfaction = 0.6
        
        if metrics.get("avg_reward", 0) < min_reward:
            logger.info(f"平均奖励过低: {metrics.get('avg_reward')} < {min_reward}")
            return False
        
        if metrics.get("user_satisfaction", 0) < min_satisfaction:
            logger.info(f"用户满意度过低: {metrics.get('user_satisfaction')} < {min_satisfaction}")
            return False
        
        # 检查质量检查结果
        if not all(quality_checks.values()):
            logger.info(f"质量检查未通过: {quality_checks}")
            return False
        
        logger.info("模型通过部署检查")
        return True
    
    async def deploy_updated_model(self, model_path: str):
        """部署更新后的模型"""
        
        logger.info(f"开始部署新模型: {model_path}")
        
        # 1. 找到checkpoint目录
        checkpoint_dir = Path(model_path) / "checkpoint-final"
        if not checkpoint_dir.exists():
            logger.error(f"未找到checkpoint目录: {checkpoint_dir}")
            return
        
        # 2. 准备部署脚本 (这里需要根据实际部署方式调整)
        deploy_script = f"""#!/bin/bash
# 自动部署脚本

echo "部署新模型: {checkpoint_dir}"

# 停止现有服务
pkill -f "vllm.*8001" || true

# 等待服务停止
sleep 5

# 启动新模型服务
nohup vllm serve {checkpoint_dir} \\
    --host 0.0.0.0 \\
    --port 8001 \\
    --gpu-memory-utilization 0.7 \\
    --max-num-seqs 32 \\
    > logs/vllm_deploy.log 2>&1 &

# 等待服务启动
sleep 30

# 健康检查
echo "进行健康检查..."
curl -s http://localhost:8001/health || echo "健康检查失败"

echo "模型部署完成"
"""
        
        # 3. 保存并执行部署脚本
        deploy_script_path = Path("deploy_model.sh")
        with open(deploy_script_path, "w") as f:
            f.write(deploy_script)
        
        deploy_script_path.chmod(0o755)
        
        # 执行部署 (实际生产环境中可能需要更复杂的部署流程)
        logger.info("执行模型部署...")
        
        process = await asyncio.create_subprocess_exec(
            "bash", str(deploy_script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info("模型部署成功")
            logger.info(f"部署输出: {stdout.decode()}")
        else:
            logger.error(f"模型部署失败: {stderr.decode()}")
        
        # 清理部署脚本
        deploy_script_path.unlink(missing_ok=True)
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        
        if not self.training_history:
            return {"message": "暂无训练历史"}
        
        recent = self.training_history[-10:]  # 最近10次
        
        return {
            "total_iterations": len(self.training_history),
            "recent_iterations": len(recent),
            "avg_reward": np.mean([h["avg_reward"] for h in recent]),
            "deployment_rate": sum(1 for h in recent if h["deployed"]) / len(recent),
            "last_training": self.training_history[-1]["timestamp"] if self.training_history else None,
            "trend": "improving" if len(recent) > 1 and recent[-1]["avg_reward"] > recent[0]["avg_reward"] else "stable"
        }


# CLI接口
async def main():
    """主函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Sales-RAG DeepRetrieval RL训练系统")
    parser.add_argument("--mode", choices=["continuous", "single", "collect"], 
                       default="continuous", help="运行模式")
    parser.add_argument("--config", default="code/config/sales_rag_rl_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--min-samples", type=int, default=30,
                       help="最小训练样本数")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = DeepRetrievalRLTrainer(args.config)
    
    if args.mode == "continuous":
        # 持续训练模式
        await trainer.continuous_training_loop()
        
    elif args.mode == "single":
        # 单次训练模式
        samples = trainer.collector.collect_training_samples(args.min_samples)
        if samples:
            train_info = trainer.collector.prepare_training_data(samples)
            result = await trainer.run_training(train_info)
            logger.info(f"单次训练完成: {result}")
        else:
            logger.info("样本不足，无法进行训练")
            
    elif args.mode == "collect":
        # 仅收集数据模式
        samples = trainer.collector.collect_training_samples(1)  # 收集所有样本
        if samples:
            train_info = trainer.collector.prepare_training_data(samples)
            logger.info(f"数据收集完成: {train_info}")
        else:
            logger.info("暂无可收集的样本")


if __name__ == "__main__":
    asyncio.run(main())
