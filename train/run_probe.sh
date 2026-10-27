#!/bin/bash
# 对照实验：只改 offload / micro_batch，其余与 train-001 完全一致。
# 目的：量化"配置低效"相对"模型大小"的贡献。
set -u
R=${GRPO_BASELINE:?set GRPO_BASELINE to the baseline experiment dir}
PY=$R/vendor/verl/.venv/bin/python
OUT=$(cd "$(dirname "$0")" && pwd)/probe-$1
mkdir -p "$OUT"

NGPU=${2:-8}
STEPS=${3:-4}

case $NGPU in
  8) DEV=0,1,2,3,4,5,6,7 ;;
  4) DEV=4,5,6,7 ;;
  *) DEV=0,1,2,3 ;;
esac

export CUDA_VISIBLE_DEVICES=$DEV
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

exec $PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator="grpo" \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  algorithm.filter_groups.enable=false \
  data.train_files="[\"$R/data/train.jsonl\"]" \
  data.val_files="[\"$R/data/dev.jsonl\"]" \
  data.train_batch_size=32 \
  data.val_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=false \
  data.truncation="error" \
  data.dataloader_num_workers=0 \
  data.seed=300 \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="$R/execution/20261017/model" \
  actor_rollout_ref.model.trust_remote_code=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.strategy="fsdp" \
  actor_rollout_ref.actor.optim.lr=1e-06 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI:-16} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO:-8} \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
  actor_rollout_ref.actor.policy_loss.loss_mode="vanilla" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.data_loader_seed=300 \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${MDTYPE:-bfloat16}" \
  actor_rollout_ref.actor.fsdp_config.dtype="bfloat16" \
  actor_rollout_ref.actor.fsdp_config.param_offload=${POFF:-false} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OOFF:-false} \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.seed=300 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${RMICRO:-8} \
  actor_rollout_ref.ref.fsdp_config.param_offload=${POFF:-false} \
  actor_rollout_ref.ref.fsdp_config.model_dtype="bfloat16" \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.rollout.name="vllm" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP:-2} \
  actor_rollout_ref.rollout.dtype="bfloat16" \
  actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_MEM:-0.25} \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${RMICRO:-8} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.seed=300 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  reward.custom_reward_function.path="$R/reward.py" \
  reward.custom_reward_function.name="compute_score" \
  reward.num_workers=2 \
  reward.reward_model.enable=false \
  ray_kwargs.ray_init.num_cpus=${NCPU:-40} \
  "+ray_kwargs.ray_init.num_gpus=$NGPU" \
  "+ray_kwargs.ray_init.address=\"local\"" \
  "+ray_kwargs.ray_init.include_dashboard=false" \
  actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false} \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.project_name="rl-lab-probe" \
  trainer.experiment_name="probe-$1" \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$STEPS \
  trainer.val_before_train=false \
  trainer.default_local_dir="$OUT" \
  2>&1
