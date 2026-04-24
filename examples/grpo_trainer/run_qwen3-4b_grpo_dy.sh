# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
# Modified version with rubric generation enabled
ray stop
set -x

rm -rf ~/.triton ~/.cache/torch/inductor ~/.cache/torch/extension_cache
export VLLM_USE_V1=1
export USER="tianxy-cs"
export WANDB_PROJECT="rubric"
export WANDB_ENTITY="maggie-ict"
exp_name='qwen3_4b_grpo_rub_v1_2'
export WANDB_RUN_ID="qwen3_4b_grpo_rub_v1_2"
export WANDB_RESUME="allow"

export WANDB_API_KEY="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"

# Auto login to wandb
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY not set, attempting to use cached credentials"
    wandb login --relogin 2>/dev/null || echo "Please set WANDB_API_KEY or run 'wandb login' manually"
fi

CKPTS_DIR="/primus_xpfs_workspace_T04/txy/models/${WANDB_PROJECT}/${exp_name}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/primus_xpfs_workspace_T04/txy/data/DAPO-Math-17k/train.parquet" \
    data.val_files="/primus_xpfs_workspace_T04/txy/data/AIME24/dev.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="/primus_xpfs_workspace_T04/txy/models/Qwen3-4B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_training_steps=3000 \
    trainer.total_epochs=60 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.enable_rubric_generation=true \
    trainer.rubric_generation.template_path="/primus_xpfs_workspace_T04/txy/projects/verl/rubric/init_rubric.txt" \
    trainer.rubric_generation.update_template_path="/primus_xpfs_workspace_T04/txy/projects/verl/rubric/update_rubric.txt" \
    trainer.rubric_generation.output_dir="/primus_xpfs_workspace_T04/txy/projects/verl/rubric" \
    trainer.rubric_generation.num_prompts=8 \
    trainer.rubric_generation.num_rollouts_per_prompt=2 \
    trainer.rubric_generation.num_low_var_groups=8 \
    trainer.rubric_generation.judge_max_workers=32 \
    trainer.rubric_generation.llm_env_path="/primus_xpfs_workspace_T04/txy/projects/verl/rubric/.env" $@
