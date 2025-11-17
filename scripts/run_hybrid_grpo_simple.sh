#!/bin/bash
# 混合GRPO训练脚本 - 简化修复版
# 修复要点：正确的PYTHONPATH和避免GPU检测崩溃

set -x

echo "🚀 启动混合GRPO训练，目标GPU: 4,5,6,7"

# === 环境设置 ===
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/home/jovyan2/query_rl/verl_code:$PYTHONPATH

echo "✅ CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "✅ PYTHONPATH: $PYTHONPATH"

# === 简单验证 ===
python3 -c "
import sys
print(f'✅ Python路径: {sys.path[0]}')
try:
    import verl.trainer.main_ppo
    print('✅ verl模块导入成功')
except Exception as e:
    print(f'❌ verl模块导入失败: {e}')
"

# === 执行训练 ===
echo "🎯 启动训练..."

cd /home/jovyan2/query_rl/verl_code

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    data.train_files=/home/jovyan2/query_rl/data/sales_rag/train.parquet \
    data.val_files=/home/jovyan2/query_rl/data/sales_rag/val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=128 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    +data.data_source=sales_rag_hybrid \
    data.shuffle=true \
    actor_rollout_ref.model.path=/home/jovyan2/query_rl/model/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=sales_rag_hybrid_grpo_fixed \
    trainer.experiment_name=qwen3_8b_hybrid_grpo_query_rewrite_v3_1 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    trainer.default_local_dir=checkpoints/SalesRAG_Hybrid_GRPO_Fixed/query_rewrite \
    +algorithm.hybrid_grpo.enable=true \
    +algorithm.hybrid_grpo.grpo_weight=0.7 \
    +algorithm.hybrid_grpo.auxiliary_weight=0.3 \
    +algorithm.hybrid_grpo.enable_dynamic_weight=true \
    +algorithm.hybrid_grpo.weight_decay_rate=0.4 \
    +algorithm.hybrid_grpo.min_auxiliary_weight=0.1 \
    +algorithm.hybrid_grpo.auxiliary_centralization=true \
    +algorithm.hybrid_grpo.auxiliary_normalization=std \
    +algorithm.hybrid_grpo.scoring_model=GPT-5 \
    +algorithm.hybrid_grpo.group_size=5 \
    +seed=42