#!/usr/bin/env bash
# GRPO + LoRA for OVEN RSA-trace open-world image classification.
#
# Defaults are safe for a smoke run. Override TRAIN_FILE/VAL_FILE and
# TOTAL_TRAINING_STEPS for longer runs.

set -euo pipefail
set -x

VERL_ROOT=${VERL_ROOT:-/leonardo_scratch/fast/EUHPC_D33_243/verl}
OVEN_ROOT=${OVEN_ROOT:-/leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval}
FAST=${FAST:-/leonardo_scratch/fast/EUHPC_D33_243}
CONDA_ENV=${CONDA_ENV:-verl-vllm110}
CONDA_SH=${CONDA_SH:-}
USE_CONDA=${USE_CONDA:-1}
VERL_VENV=${VERL_VENV:-"${VERL_ROOT}/.venv"}
PYTHON_BIN=${PYTHON_BIN:-python}

UV_CACHE_DIR=${UV_CACHE_DIR:-"${FAST}/.cache/uv"}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-"${FAST}/.cache/pip"}
TMPDIR=${TMPDIR:-"${FAST}/tmp"}
HF_HOME=${HF_HOME:-"${FAST}/.cache/huggingface"}
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_DIR=${WANDB_DIR:-"${FAST}/wandb_runs"}
VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
VLLM_USE_V1=${VLLM_USE_V1:-1}
HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-}
PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
PYTORCH_NVML_BASED_CUDA_CHECK=${PYTORCH_NVML_BASED_CUDA_CHECK:-1}
CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
VERL_IMPORT_PROBE=${VERL_IMPORT_PROBE:-0}
KEEP_RAY_TMPDIR_ON_FAILURE=${KEEP_RAY_TMPDIR_ON_FAILURE:-1}
VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS=${VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS:-nccl,nixl,hccl,kimi,mooncake}

DATASET_DIR=${DATASET_DIR:-"${OVEN_ROOT}/data/processed/verl_oven_rsa_trace_smoke_512"}
TRAIN_FILE=${TRAIN_FILE:-"${DATASET_DIR}/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATASET_DIR}/val.parquet"}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
REWARD_FN_PATH=${REWARD_FN_PATH:-"${VERL_ROOT}/verl/utils/reward_score/oven_boxed.py"}
OVEN_TAXONOMY_INDEX=${OVEN_TAXONOMY_INDEX:-"${OVEN_ROOT}/data/processed/oven_taxonomy_index.json"}

PROJECT_NAME=${PROJECT_NAME:-oven_rsa_trace_grpo}
EXP_NAME=${EXP_NAME:-qwen3_vl_4b_oven_rsa_trace_smoke}
CKPTS_DIR=${CKPTS_DIR:-"${VERL_ROOT}/checkpoints/${PROJECT_NAME}/${EXP_NAME}"}

N_GPUS=${N_GPUS:-4}
N_NODES=${N_NODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_MIN_MODEL_LEN=${ROLLOUT_MIN_MODEL_LEN:-20000}
ROLLOUT_DEFAULT_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
if (( ROLLOUT_DEFAULT_MODEL_LEN < ROLLOUT_MIN_MODEL_LEN )); then
    ROLLOUT_DEFAULT_MODEL_LEN=$ROLLOUT_MIN_MODEL_LEN
fi
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$ROLLOUT_DEFAULT_MODEL_LEN}
if (( ROLLOUT_MAX_MODEL_LEN < ROLLOUT_MIN_MODEL_LEN )); then
    echo "[warn] ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MAX_MODEL_LEN is below ROLLOUT_MIN_MODEL_LEN=$ROLLOUT_MIN_MODEL_LEN; raising it for Qwen3-VL vLLM profiling" >&2
    ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MIN_MODEL_LEN
fi
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-$TRAIN_BATCH_SIZE}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_MODEL_LEN )); then
    echo "[warn] ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS is below ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MAX_MODEL_LEN; raising it" >&2
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_MODEL_LEN
fi
ROLLOUT_LIMIT_IMAGES=${ROLLOUT_LIMIT_IMAGES:-1}
if (( TRAIN_BATCH_SIZE % ROLLOUT_AGENT_NUM_WORKERS != 0 )); then
    echo "[error] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by ROLLOUT_AGENT_NUM_WORKERS=$ROLLOUT_AGENT_NUM_WORKERS" >&2
    exit 2
fi
MODEL_USE_REMOVE_PADDING=${MODEL_USE_REMOVE_PADDING:-True}
MODEL_USE_FUSED_KERNELS=${MODEL_USE_FUSED_KERNELS:-True}
MODEL_ENABLE_GRADIENT_CHECKPOINTING=${MODEL_ENABLE_GRADIENT_CHECKPOINTING:-False}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp2}
REF_STRATEGY=${REF_STRATEGY:-fsdp2}
ACTOR_FSDP_SIZE=${ACTOR_FSDP_SIZE:--1}
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-bf16}
REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bf16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOGGER=${LOGGER:-'["console"]'}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_MERGE=${LORA_MERGE:-True}
LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.001}

case "${VLLM_USE_V1,,}" in
    1|true) VLLM_USE_V1=1 ;;
    *)
        echo "[error] VLLM_USE_V1=$VLLM_USE_V1 is incompatible with this VERL async vLLM rollout path; use VLLM_USE_V1=1" >&2
        exit 2
        ;;
esac

if [[ "$USE_CONDA" == "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then
        source "$CONDA_SH"
    elif [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [[ -f "/leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[error] could not initialize conda; set CONDA_SH or USE_CONDA=0 VERL_VENV=/path/to/venv" >&2
        exit 1
    fi
    conda activate "$CONDA_ENV"
elif [[ -f "$VERL_VENV/bin/activate" ]]; then
    source "$VERL_VENV/bin/activate"
fi

which "$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
import tensordict
import torch
print("[info] python:", sys.executable)
print("[info] tensordict:", tensordict.__version__)
print("[info] torch:", torch.__version__)
try:
    import torchaudio
except Exception as exc:
    raise SystemExit(
        "[error] torchaudio is installed but incompatible with this Torch build. "
        "Reinstall torchaudio to match torch before launching VERL.\n"
        f"torch={torch.__version__}\n"
        f"{type(exc).__name__}: {exc}"
    ) from exc
print("[info] torchaudio:", torchaudio.__version__)
PY

for required in "$TRAIN_FILE" "$VAL_FILE" "$REWARD_FN_PATH" "$OVEN_TAXONOMY_INDEX"; do
    if [[ ! -f "$required" ]]; then
        echo "[error] required file missing: $required" >&2
        exit 1
    fi
done

export OVEN_TAXONOMY_INDEX
export UV_CACHE_DIR PIP_CACHE_DIR TMPDIR
export HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE
export HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export WANDB_MODE WANDB_DIR
export VLLM_ALLREDUCE_USE_SYMM_MEM VLLM_USE_V1
export HYDRA_FULL_ERROR RAY_DEDUP_LOGS
export PYTHONFAULTHANDLER TORCH_SHOW_CPP_STACKTRACES TORCH_DISABLE_ADDR2LINE PYTORCH_NVML_BASED_CUDA_CHECK CUDA_MODULE_LOADING
export VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID:-$$}}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
unset ROCR_VISIBLE_DEVICES
mkdir -p "$RAY_TMPDIR" "$CKPTS_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" \
    "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR"

SITE_CUSTOMIZE_DIR="$RAY_TMPDIR/sitecustomize"
mkdir -p "$SITE_CUSTOMIZE_DIR"
cat > "$SITE_CUSTOMIZE_DIR/sitecustomize.py" <<'PY'
import importlib.abc
import os
import sys

blocked = {
    f"verl.checkpoint_engine.{name.strip()}_checkpoint_engine"
    for name in os.environ.get("VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS", "").split(",")
    if name.strip()
}


class _BlockOptionalCheckpointBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise ImportError(f"{fullname} disabled by VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS")
        return None


if blocked:
    sys.meta_path.insert(0, _BlockOptionalCheckpointBackends())
PY
export PYTHONPATH="$SITE_CUSTOMIZE_DIR${PYTHONPATH:+:$PYTHONPATH}"

cleanup() {
    status=$?
    if [[ "$status" -eq 0 || "$KEEP_RAY_TMPDIR_ON_FAILURE" != "1" ]]; then
        rm -rf "$RAY_TMPDIR"
    else
        echo "[warn] preserving RAY_TMPDIR after failure: $RAY_TMPDIR" >&2
    fi
}
trap cleanup EXIT

cd "$VERL_ROOT"

"$PYTHON_BIN" - <<'PY'
import os
from transformers import AutoConfig

model_path = os.environ["MODEL_PATH"]
try:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
except Exception as exc:
    raise SystemExit(
        "[error] Transformers cannot load the model config. "
        "For Qwen3-VL, upgrade the verl conda env Transformers package or use a "
        "model supported by the current env.\n"
        f"model={model_path}\n"
        f"{type(exc).__name__}: {exc}"
    ) from exc
print("[info] model config:", type(cfg).__name__, getattr(cfg, "model_type", None))
PY

"$PYTHON_BIN" - <<'PY'
from verl.utils.reward_score.oven_boxed import compute_score

assert compute_score("oven", r"\boxed{Air gun}", "Air gun") == 1.0
assert compute_score("oven", r"\boxed{bolt-action rifle}", "Air gun") == 0.05
assert compute_score("oven", "Air gun", "Air gun") == 0.0
print("[info] oven_boxed reward smoke OK")
PY

if [[ "$VERL_IMPORT_PROBE" == "1" ]]; then
    for module in \
        verl.utils.device \
        verl.utils.model \
        verl.experimental.reward_loop \
        verl.trainer.ppo.ray_trainer \
        verl.trainer.main_ppo \
        verl.third_party.vllm \
        verl.workers.rollout.vllm_rollout.vllm_rollout
    do
        "$PYTHON_BIN" -X faulthandler - <<PY
import faulthandler
import importlib

faulthandler.enable()
module = "$module"
print(f"[probe] importing {module}", flush=True)
importlib.import_module(module)
print(f"[probe] ok {module}", flush=True)
PY
    done
fi

RAY_INIT_ARGS=()
if [[ -n "$RAY_OBJECT_STORE_MEMORY" ]]; then
    RAY_INIT_ARGS+=(+ray_kwargs.ray_init.object_store_memory="$RAY_OBJECT_STORE_MEMORY")
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    +actor_rollout_ref.model.lora.merge="$LORA_MERGE" \
    actor_rollout_ref.model.use_remove_padding="$MODEL_USE_REMOVE_PADDING" \
    actor_rollout_ref.model.use_fused_kernels="$MODEL_USE_FUSED_KERNELS" \
    actor_rollout_ref.model.enable_gradient_checkpointing="$MODEL_ENABLE_GRADIENT_CHECKPOINTING" \
    \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy="$ACTOR_STRATEGY" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$ACTOR_FSDP_SIZE" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$ACTOR_MODEL_DTYPE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_UTIL" \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image="$ROLLOUT_LIMIT_IMAGES" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.strategy="$REF_STRATEGY" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$REF_MODEL_DTYPE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    reward.custom_reward_function.path="$REWARD_FN_PATH" \
    reward.custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes="$N_NODES" \
    trainer.default_local_dir="$CKPTS_DIR" \
    trainer.resume_mode=auto \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    "${RAY_INIT_ARGS[@]}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OVEN_TAXONOMY_INDEX="$OVEN_TAXONOMY_INDEX" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="$HF_HOME" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="'$HF_HUB_OFFLINE'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_OFFLINE="'$HF_DATASETS_OFFLINE'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="'$TRANSFORMERS_OFFLINE'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE="$WANDB_MODE" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR="$WANDB_DIR" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALLREDUCE_USE_SYMM_MEM="'$VLLM_ALLREDUCE_USE_SYMM_MEM'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'$VLLM_USE_V1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONFAULTHANDLER="'$PYTHONFAULTHANDLER'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_SHOW_CPP_STACKTRACES="'$TORCH_SHOW_CPP_STACKTRACES'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_DISABLE_ADDR2LINE="'$TORCH_DISABLE_ADDR2LINE'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NVML_BASED_CUDA_CHECK="'$PYTORCH_NVML_BASED_CUDA_CHECK'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_MODULE_LOADING="$CUDA_MODULE_LOADING" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS="'$VERL_DISABLE_OPTIONAL_CHECKPOINT_BACKENDS'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
    "$@"
