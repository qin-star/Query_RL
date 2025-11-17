#!/bin/bash
# 混合GRPO训练脚本 - GPU可配置版（优化版）
# 基于verl官方GRPO配置格式，完全兼容
# 关键修正：GPT-5辅助奖励采用组内中心化，保持GRPO零均值特性
# GPU配置：通过Python级别设置确保Ray框架兼容性
set -x

export CUDA_VISIBLE_DEVICES=6,7

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    data.train_files=/home/jovyan2/query_rl/data/sales_rag/train.parquet \
    data.val_files=/home/jovyan2/query_rl/data/sales_rag/val.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=true \
    data.truncation="error" \
    +data.data_source=sales_rag_hybrid \
    data.shuffle=true \
    actor_rollout_ref.model.path=/home/jovyan2/query_rl/model/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='sales_rag_hybrid_grpo_fixed' \
    trainer.experiment_name='qwen3_8b_hybrid_grpo_query_rewrite_v3_1' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    trainer.default_local_dir='checkpoints/SalesRAG_Hybrid_GRPO_Fixed/query_rewrite' \
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
