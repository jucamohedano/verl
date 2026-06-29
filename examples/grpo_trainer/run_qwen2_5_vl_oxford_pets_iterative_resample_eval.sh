#!/usr/bin/env bash
# Eval-only iterative resampling for Qwen2.5-VL on Oxford Pets.
#
# This is a test-time baseline: no GRPO update is performed. For each sample,
# the custom AgentLoop draws N independent attempts; if all fail simple label
# matching, it feeds those failed attempts back into context and repeats for T
# rounds.

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Paths and method knobs - override with environment variables as needed.
# ---------------------------------------------------------------------------
# TRAIN_FILE="${TRAIN_FILE:-/leonardo_scratch/fast/EUHPC_D33_243/lmms-ocw/grpo_datasets/oxford_pets/train.parquet}"
TRAIN_FILE="${TRAIN_FILE:-/leonardo_scratch/fast/EUHPC_D33_243/lmms-ocw/grpo_datasets/oxford_pets_classify_with_think/oxford_pets/train.parquet}"
# VAL_FILE="${VAL_FILE:-/leonardo_scratch/fast/EUHPC_D33_243/lmms-ocw/grpo_datasets/oxford_pets/test.parquet}"
VAL_FILE="${VAL_FILE:-/leonardo_scratch/fast/EUHPC_D33_243/lmms-ocw/grpo_datasets/oxford_pets_classify_with_think/oxford_pets/test.parquet}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

ATTEMPTS_PER_ROUND="${ATTEMPTS_PER_ROUND:-64}"      # n
MAX_ROUNDS="${MAX_ROUNDS:-1}"                      # T
PER_ATTEMPT_MAX_TOKENS="${PER_ATTEMPT_MAX_TOKENS:-256}"
MAX_FEEDBACK_CHARS="${MAX_FEEDBACK_CHARS:-2000}"
ENABLE_FEEDBACK="${ENABLE_FEEDBACK:-false}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-iterative_resample_outputs/oxford_pets_baseline_qwen2_5_vl}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

unset ROCR_VISIBLE_DEVICES
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/$USER/ray}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VERL_LOG_LEVEL="${VERL_LOG_LEVEL:-INFO}"
if [[ "$TRAINER_LOGGER" == *wandb* ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
fi
mkdir -p "$RAY_TMPDIR" "$VALIDATION_DATA_DIR"

AGENT_LOOP_CONFIG="$RAY_TMPDIR/iterative_resample_agent_loop.yaml"
cat > "$AGENT_LOOP_CONFIG" <<EOF
- name: iterative_resample
  _target_: verl.experimental.agent_loop.iterative_resample_agent_loop.IterativeResampleAgentLoop
  attempts_per_round: ${ATTEMPTS_PER_ROUND}
  max_rounds: ${MAX_ROUNDS}
  per_attempt_max_tokens: ${PER_ATTEMPT_MAX_TOKENS}
  max_feedback_chars: ${MAX_FEEDBACK_CHARS}
  enable_feedback: ${ENABLE_FEEDBACK}
EOF

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=256 \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=iterative_resample \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name='ttw_iterative_resample' \
    trainer.experiment_name="qwen2.5_vl_7b_oxford_pets_iter_n${ATTEMPTS_PER_ROUND}_T${MAX_ROUNDS}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    "$@"

rm -rf "$RAY_TMPDIR"
