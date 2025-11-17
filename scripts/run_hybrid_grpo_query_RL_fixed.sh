#!/bin/bash
# 混合GRPO训练脚本 - GPU可见性修复版（基于Ray框架最佳实践）
# 参考技术博客：在Ray框架下正确设置GPU可见性
# 关键修正：在Ray初始化前设置环境变量，确保设备映射正确

set -x

echo "🚀 启动混合GRPO训练，目标GPU: 4,5,6,7"
echo "📋 训练配置："
echo "  - 模型: Qwen3-8B"
echo "  - 算法: 混合GRPO (GRPO权重: 0.7, 辅助权重: 0.3)"
echo "  - 组大小: 5"
echo "  - 动态权重: 启用"
echo "  - GPT-5辅助奖励: 组内中心化"

# === 方法一：在shell级别设置环境变量（基础方案） ===
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/home/jovyan2/query_rl:$PYTHONPATH

echo "✅ Shell级别 - CUDA_VISIBLE_DEVICES 设置为: $CUDA_VISIBLE_DEVICES"
echo "✅ Shell级别 - PYTHONPATH 设置为: $PYTHONPATH"

# === 方法二：Python运行时动态设置（推荐方案） ===
# 参考博客建议：在Ray初始化前强制设定环境变量
python3 -c "
import os
import sys
import subprocess

# 技术关键点：在Ray初始化前设置GPU可见性
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ['PYTHONPATH'] = '/home/jovyan2/query_rl:' + os.environ.get('PYTHONPATH', '')

# 确保Python能够找到verl模块
sys.path.insert(0, '/home/jovyan2/query_rl')

print(f'✅ Python运行时 - CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\")}')
print(f'✅ Python运行时 - PYTHONPATH: {os.environ.get(\"PYTHONPATH\")}')
print(f'✅ Python运行时 - sys.path: {sys.path[:2]}')

# 设备映射验证：检查逻辑GPU与物理GPU的对应关系
try:
    import torch
    print(f'✅ PyTorch可见GPU数量: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            print(f'  逻辑GPU {i} -> 物理GPU: {gpu_name}')
except Exception as e:
    print(f'⚠️  GPU检测失败: {e}')

print('🎯 设备映射验证完成，启动Ray训练...')
"

# === 构建训练命令 ===
# 注意：Ray会重新映射GPU序号，物理GPU4,5,6,7将变为逻辑GPU0,1,2,3
# 使用更简洁的方式传递参数，避免引号问题
train_params=(
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=true"
    "algorithm.use_kl_in_reward=false"
    "data.train_files=/home/jovyan2/query_rl/data/sales_rag/train.parquet"
    "data.val_files=/home/jovyan2/query_rl/data/sales_rag/val.parquet"
    "data.train_batch_size=16"
    "data.max_prompt_length=128"
    "data.max_response_length=256"
    "data.filter_overlong_prompts=true"
    "data.truncation=error"
    "+data.data_source=sales_rag_hybrid"
    "data.shuffle=true"
    "actor_rollout_ref.model.path=/home/jovyan2/query_rl/model/Qwen3-8B"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.ppo_mini_batch_size=8"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"
    "actor_rollout_ref.actor.use_kl_loss=true"
    "actor_rollout_ref.actor.kl_loss_coef=0.001"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.5"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.n=2"
    "actor_rollout_ref.rollout.temperature=0.7"
    "actor_rollout_ref.rollout.top_p=0.9"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8"
    "trainer.critic_warmup=0"
    "trainer.logger=[\"console\"]"
    "trainer.project_name=sales_rag_hybrid_grpo_fixed"
    "trainer.experiment_name=qwen3_8b_hybrid_grpo_query_rewrite_v3_1"
    "trainer.n_gpus_per_node=4"
    "trainer.nnodes=1"
    "trainer.save_freq=50"
    "trainer.test_freq=10"
    "trainer.total_epochs=20"
    "trainer.default_local_dir=checkpoints/SalesRAG_Hybrid_GRPO_Fixed/query_rewrite"
    "+algorithm.hybrid_grpo.enable=true"
    "+algorithm.hybrid_grpo.grpo_weight=0.7"
    "+algorithm.hybrid_grpo.auxiliary_weight=0.3"
    "+algorithm.hybrid_grpo.enable_dynamic_weight=true"
    "+algorithm.hybrid_grpo.weight_decay_rate=0.4"
    "+algorithm.hybrid_grpo.min_auxiliary_weight=0.1"
    "+algorithm.hybrid_grpo.auxiliary_centralization=true"
    "+algorithm.hybrid_grpo.auxiliary_normalization=std"
    "+algorithm.hybrid_grpo.scoring_model=GPT-5"
    "+algorithm.hybrid_grpo.group_size=5"
    "+seed=42"
)

echo "🚀 执行训练命令..."
echo "📊 注意：物理GPU 4,5,6,7 将映射为逻辑GPU 0,1,2,3"
echo "💡 内存管理：gpu_memory_utilization=0.5 可根据实际显存调整"

# === 执行训练 ===
# 使用更可靠的方式执行命令
python3 -m verl.trainer.main_ppo "${train_params[@]}"