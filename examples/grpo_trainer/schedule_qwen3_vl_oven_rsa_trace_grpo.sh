#!/usr/bin/env bash
set -euo pipefail

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    cat <<'EOF'
usage: schedule_qwen3_vl_oven_rsa_trace_grpo.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

Schedule single-node GRPO + LoRA training for the OVEN RSA-trace dataset.

Slurm options:
  -A, --account <ACCOUNT>       Slurm account (required unless env SLURM_ACCOUNT is set)
  -p, --partition <PARTITION>   Partition (default: boost_usr_prod)
  -g, --gpus <N>                GPUs per node (default: 4)
  -c, --cpus <N>                CPUs per task (default: 32)
  -m, --mem <MEM>               Memory (default: 256G)
  -t, --time <TIME>             Time limit (default: 24:00:00)
  -n, --name <NAME>             Job name (default: oven-grpo)

Run options:
  --mode <smoke|full>           Dataset preset (default: smoke)
  --dataset-dir <PATH>          Override dataset directory
  --train-file <PATH>           Override train parquet
  --val-file <PATH>             Override val parquet
  --model <MODEL>               Model path/id (default: Qwen/Qwen3-VL-4B-Instruct)
  --steps <N>                   Total training steps
  --save-freq <N>               Checkpoint frequency
  --test-freq <N>               Validation frequency
  --val-before-train <BOOL>     Run validation before training
  --exp-name <NAME>             VERL experiment name
  --project-name <NAME>         VERL project name (default: oven_rsa_trace_grpo)
  --taxonomy-index <PATH>       OVEN taxonomy index
  --reward-fn <PATH>            Reward function path
  --ckpts-dir <PATH>            Checkpoint directory
  --verl-root <PATH>            VERL repo root
  --oven-root <PATH>            oven-mllm-eval repo root
  --fast-root <PATH>            Fast scratch root for caches (default: $FAST or /leonardo_scratch/fast/EUHPC_D33_243)
  --conda-env <NAME>            Conda env to activate (default: verl-v080)
  --conda-sh <PATH>             Path to conda.sh if conda is not on PATH
  --no-conda                    Do not activate conda; use --venv/current python
  --venv <PATH>                 Python venv to activate when --no-conda is set

W&B/logging:
  --wandb                       Use ["console","wandb"] logger
  --logger <JSON>               VERL logger value (default: ["console"])
  --wandb-project <NAME>        W&B project
  --wandb-name <NAME>           W&B run name
  --wandb-group <NAME>          W&B group
  --wandb-tags <TAGS>           W&B comma-separated tags

Other:
  --dry-run                     Write and print the sbatch script but do not submit

Resource env overrides:
  TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE, PPO_MAX_TOKEN_LEN_PER_GPU,
  ROLLOUT_N, ROLLOUT_AGENT_NUM_WORKERS, ROLLOUT_GPU_UTIL, ROLLOUT_GPU_MEMORY_UTILIZATION, MAX_RESPONSE_LENGTH, ROLLOUT_MAX_MODEL_LEN,
  ROLLOUT_MAX_NUM_SEQS, ROLLOUT_MAX_NUM_BATCHED_TOKENS, ROLLOUT_LIMIT_IMAGES,
  ROLLOUT_ENFORCE_EAGER, ROLLOUT_ENABLE_CHUNKED_PREFILL, ROLLOUT_FREE_CACHE_ENGINE,
  RAY_OBJECT_STORE_MEMORY, MODEL_USE_REMOVE_PADDING, MODEL_USE_FUSED_KERNELS,
  MODEL_ENABLE_GRADIENT_CHECKPOINTING, MODEL_ATTN_IMPLEMENTATION, ACTOR_STRATEGY, REF_STRATEGY, ACTOR_FSDP_SIZE, ACTOR_MODEL_DTYPE,
  REF_MODEL_DTYPE, LORA_RANK, LORA_ALPHA
  LORA_MERGE=True uses merged LoRA weight sync instead of vLLM dynamic LoRA.
  VLLM_USE_V1 must stay 1 for this VERL async vLLM rollout path.
  VERL_IMPORT_PROBE=1 enables subprocess import probes before training.

Presets:
  smoke:  data/processed/verl_oven_rsa_trace_smoke_512, steps=2, no checkpoints,
          no validation, conservative batch/rollout settings
  full:   data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42,
          steps=100, save/test every 25 steps, no pre-train validation
EOF
    exit 0
fi

shell_quote() {
    printf '%q' "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_OVEN_ROOT="$(cd "$DEFAULT_VERL_ROOT/../oven-mllm-eval" 2>/dev/null && pwd || true)"

VERL_ROOT="${VERL_ROOT:-$DEFAULT_VERL_ROOT}"
OVEN_ROOT="${OVEN_ROOT:-${DEFAULT_OVEN_ROOT:-/leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval}}"
FAST="${FAST:-/leonardo_scratch/fast/EUHPC_D33_243}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"
SLURM_GPUS="${SLURM_GPUS:-4}"
SLURM_CPUS="${SLURM_CPUS:-32}"
SLURM_MEM="${SLURM_MEM:-256G}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
SLURM_NAME="${SLURM_NAME:-oven-grpo}"

# Modules for the vllm 0.19+cu126 wheel (torch 2.10 / CUDA 12.8 libs).
# `module purge` first to drop the login-node default stack (nvhpc etc.).
MODULE_GCC="${MODULE_GCC:-gcc/12.2.0}"
MODULE_CUDA="${MODULE_CUDA:-cuda/12.6}"

MODE="smoke"
DATASET_DIR=""
TRAIN_FILE=""
VAL_FILE=""
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-oven_rsa_trace_grpo}"
EXP_NAME=""
TOTAL_TRAINING_STEPS=""
SAVE_FREQ=""
TEST_FREQ=""
VAL_BEFORE_TRAIN=""
REWARD_FN_PATH="${REWARD_FN_PATH:-$VERL_ROOT/verl/utils/reward_score/oven_boxed.py}"
OVEN_TAXONOMY_INDEX="${OVEN_TAXONOMY_INDEX:-$OVEN_ROOT/data/processed/oven_taxonomy_index.json}"
CKPTS_DIR=""
CONDA_ENV="${CONDA_ENV:-verl-v080}"
CONDA_SH="${CONDA_SH:-}"
USE_CONDA="${USE_CONDA:-1}"
VERL_VENV="${VERL_VENV:-$VERL_ROOT/.venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$FAST/.cache/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$FAST/.cache/pip}"
TMPDIR="${TMPDIR:-$FAST/tmp}"
HF_HOME="${HF_HOME:-$FAST/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-$FAST/wandb_runs}"
VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-}"
PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
VERL_IMPORT_PROBE="${VERL_IMPORT_PROBE:-0}"
KEEP_RAY_TMPDIR_ON_FAILURE="${KEEP_RAY_TMPDIR_ON_FAILURE:-1}"
MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-}"
MODEL_USE_FUSED_KERNELS="${MODEL_USE_FUSED_KERNELS:-}"
MODEL_ENABLE_GRADIENT_CHECKPOINTING="${MODEL_ENABLE_GRADIENT_CHECKPOINTING:-}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-}"
ACTOR_STRATEGY="${ACTOR_STRATEGY:-}"
REF_STRATEGY="${REF_STRATEGY:-}"
ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:-}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-}"
REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"
LORA_MERGE="${LORA_MERGE:-}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-}"
ROLLOUT_N="${ROLLOUT_N:-}"
ROLLOUT_TP="${ROLLOUT_TP:-}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-${ROLLOUT_GPU_MEMORY_UTILIZATION:-}}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}"
ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-}"

LOGGER='["console"]'
WANDB_PROJECT=""
WANDB_NAME=""
WANDB_GROUP="oven_rsa_trace"
WANDB_TAGS="oven,rsa_trace,grpo,qwen3_vl"
DRY_RUN="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        EXTRA_ARGS=("$@")
        break
    fi
    case "$1" in
        -A|--account)            SLURM_ACCOUNT="$2"; shift 2 ;;
        -p|--partition)          SLURM_PARTITION="$2"; shift 2 ;;
        -g|--gpus)               SLURM_GPUS="$2"; shift 2 ;;
        -c|--cpus)               SLURM_CPUS="$2"; shift 2 ;;
        -m|--mem)                SLURM_MEM="$2"; shift 2 ;;
        -t|--time)               SLURM_TIME="$2"; shift 2 ;;
        -n|--name)               SLURM_NAME="$2"; shift 2 ;;
        --mode)                  MODE="$2"; shift 2 ;;
        --dataset-dir)           DATASET_DIR="$2"; shift 2 ;;
        --train-file)            TRAIN_FILE="$2"; shift 2 ;;
        --val-file)              VAL_FILE="$2"; shift 2 ;;
        --model)                 MODEL_PATH="$2"; shift 2 ;;
        --steps)                 TOTAL_TRAINING_STEPS="$2"; shift 2 ;;
        --save-freq)             SAVE_FREQ="$2"; shift 2 ;;
        --test-freq)             TEST_FREQ="$2"; shift 2 ;;
        --val-batch-size)         VAL_BATCH_SIZE="$2"; shift 2 ;;
        --val-before-train)      VAL_BEFORE_TRAIN="$2"; shift 2 ;;
        --exp-name)              EXP_NAME="$2"; shift 2 ;;
        --project-name)          PROJECT_NAME="$2"; shift 2 ;;
        --train-batch-size)      TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --ppo-mini-batch-size)   PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
        --rollout-n)             ROLLOUT_N="$2"; shift 2 ;;
        --rollout-tp)            ROLLOUT_TP="$2"; shift 2 ;;
        --rollout-agents)        ROLLOUT_AGENT_NUM_WORKERS="$2"; shift 2 ;;
        --rollout-min-model-len) ROLLOUT_MIN_MODEL_LEN="$2"; shift 2 ;;
        --max-prompt-length)     MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --max-response-length)   MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --total-epochs)          TOTAL_EPOCHS="$2"; shift 2 ;;
        --gpu-util)              ROLLOUT_GPU_UTIL="$2"; shift 2 ;;
        --max-token-len-per-gpu) PPO_MAX_TOKEN_LEN_PER_GPU="$2"; shift 2 ;;
        --taxonomy-index)        OVEN_TAXONOMY_INDEX="$2"; shift 2 ;;
        --reward-fn)             REWARD_FN_PATH="$2"; shift 2 ;;
        --ckpts-dir)             CKPTS_DIR="$2"; shift 2 ;;
        --verl-root)             VERL_ROOT="$2"; shift 2 ;;
        --oven-root)             OVEN_ROOT="$2"; shift 2 ;;
        --fast-root)
            FAST="$2"
            UV_CACHE_DIR="$FAST/.cache/uv"
            PIP_CACHE_DIR="$FAST/.cache/pip"
            TMPDIR="$FAST/tmp"
            HF_HOME="$FAST/.cache/huggingface"
            HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
            HF_DATASETS_CACHE="$HF_HOME/datasets"
            WANDB_DIR="$FAST/wandb_runs"
            shift 2
            ;;
        --conda-env)             CONDA_ENV="$2"; USE_CONDA="1"; shift 2 ;;
        --conda-sh)              CONDA_SH="$2"; shift 2 ;;
        --no-conda)              USE_CONDA="0"; shift ;;
        --venv)                  VERL_VENV="$2"; USE_CONDA="0"; shift 2 ;;
        --wandb)                 LOGGER='["console","wandb"]'; shift ;;
        --logger)                LOGGER="$2"; shift 2 ;;
        --wandb-project)         WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-name)            WANDB_NAME="$2"; shift 2 ;;
        --wandb-group)           WANDB_GROUP="$2"; shift 2 ;;
        --wandb-tags)            WANDB_TAGS="$2"; shift 2 ;;
        --dry-run)               DRY_RUN="1"; shift ;;
        *) echo "[error] unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$SLURM_ACCOUNT" ]]; then
    echo "[error] pass -A/--account or set SLURM_ACCOUNT" >&2
    exit 2
fi

case "$MODE" in
    smoke)
        DATASET_DIR="${DATASET_DIR:-$OVEN_ROOT/data/processed/verl_oven_rsa_trace_smoke_512}"
        TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
        SAVE_FREQ="${SAVE_FREQ:--1}"
        TEST_FREQ="${TEST_FREQ:--1}"
        VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
        EXP_NAME="${EXP_NAME:-qwen3_vl_4b_oven_rsa_trace_smoke}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
        PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
        PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
        ROLLOUT_N="${ROLLOUT_N:-1}"
        ROLLOUT_TP="${ROLLOUT_TP:-4}"
        ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-1}"
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.50}"
        ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
        ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
        ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
        ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
        ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-20000}"
        ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-1}"
        MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-False}"
        MODEL_USE_FUSED_KERNELS="${MODEL_USE_FUSED_KERNELS:-False}"
        MODEL_ENABLE_GRADIENT_CHECKPOINTING="${MODEL_ENABLE_GRADIENT_CHECKPOINTING:-False}"
        # No-flash smoke: use_remove_padding/use_fused_kernels=True hard-import flash_attn, which has no
        # prebuilt wheel for torch 2.10. So the smoke runs sdpa with both off to validate the stack.
        MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
        ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
        REF_STRATEGY="${REF_STRATEGY:-fsdp2}"
        ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:-1}"
        ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bf16}"
        REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-bf16}"
        LORA_RANK="${LORA_RANK:-0}"
        LORA_ALPHA="${LORA_ALPHA:-32}"
        LORA_MERGE="${LORA_MERGE:-False}"
        MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
        MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
        RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8589934592}"
        ;;
    full)
        DATASET_DIR="${DATASET_DIR:-$OVEN_ROOT/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42}"
        TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
        SAVE_FREQ="${SAVE_FREQ:-25}"
        TEST_FREQ="${TEST_FREQ:-25}"
        VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
        EXP_NAME="${EXP_NAME:-qwen3_vl_4b_oven_rsa_trace_250k_lora}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
        PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
        PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
        ROLLOUT_N="${ROLLOUT_N:-4}"
        ROLLOUT_TP="${ROLLOUT_TP:-4}"
        ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.50}"
        ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
        ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}"
        ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
        ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
        ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-20000}"
        ROLLOUT_LIMIT_IMAGES="${ROLLOUT_LIMIT_IMAGES:-1}"
        MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-True}"
        MODEL_USE_FUSED_KERNELS="${MODEL_USE_FUSED_KERNELS:-True}"
        MODEL_ENABLE_GRADIENT_CHECKPOINTING="${MODEL_ENABLE_GRADIENT_CHECKPOINTING:-True}"
        MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"
        ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
        REF_STRATEGY="${REF_STRATEGY:-fsdp2}"
        ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:--1}"
        ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bf16}"
        REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-bf16}"
        LORA_RANK="${LORA_RANK:-64}"
        LORA_ALPHA="${LORA_ALPHA:-32}"
        LORA_MERGE="${LORA_MERGE:-True}"
        MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
        MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
        MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4096}"
        VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$MAX_VAL_SAMPLES}"
        RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-17179869184}"
        ;;
    *)
        echo "[error] --mode must be smoke or full: $MODE" >&2
        exit 2
        ;;
esac

TRAIN_FILE="${TRAIN_FILE:-$DATASET_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATASET_DIR/val.parquet}"
WANDB_PROJECT="${WANDB_PROJECT:-$PROJECT_NAME}"
WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
CKPTS_DIR="${CKPTS_DIR:-$VERL_ROOT/checkpoints/$PROJECT_NAME/$EXP_NAME}"
ROLLOUT_MIN_MODEL_LEN="${ROLLOUT_MIN_MODEL_LEN:-20000}"
ROLLOUT_DEFAULT_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
if (( ROLLOUT_DEFAULT_MODEL_LEN < ROLLOUT_MIN_MODEL_LEN )); then
    ROLLOUT_DEFAULT_MODEL_LEN=$ROLLOUT_MIN_MODEL_LEN
fi
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$ROLLOUT_DEFAULT_MODEL_LEN}"
if (( ROLLOUT_MAX_MODEL_LEN < ROLLOUT_MIN_MODEL_LEN )); then
    echo "[warn] ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MAX_MODEL_LEN is below ROLLOUT_MIN_MODEL_LEN=$ROLLOUT_MIN_MODEL_LEN; raising it for Qwen3-VL vLLM profiling" >&2
    ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MIN_MODEL_LEN
fi
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}"
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_MODEL_LEN )); then
    echo "[warn] ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS is below ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MAX_MODEL_LEN; raising it" >&2
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_MODEL_LEN
fi
if (( TRAIN_BATCH_SIZE % ROLLOUT_AGENT_NUM_WORKERS != 0 )); then
    echo "[error] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by ROLLOUT_AGENT_NUM_WORKERS=$ROLLOUT_AGENT_NUM_WORKERS" >&2
    exit 2
fi

case "${VLLM_USE_V1,,}" in
    1|true) VLLM_USE_V1=1 ;;
    *)
        echo "[error] VLLM_USE_V1=$VLLM_USE_V1 is incompatible with this VERL async vLLM rollout path; unset it or set VLLM_USE_V1=1" >&2
        exit 2
        ;;
esac

LOG_DIR="$VERL_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
SBATCH_FILE="$LOG_DIR/${SLURM_NAME}_$(date +%Y%m%d_%H%M%S).sbatch"

{
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=$SLURM_NAME
#SBATCH --output=$LOG_DIR/%j.out
#SBATCH --error=$LOG_DIR/%j.err
#SBATCH --partition=$SLURM_PARTITION
#SBATCH --account=$SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --gres=gpu:$SLURM_GPUS
#SBATCH --mem=$SLURM_MEM
#SBATCH --time=$SLURM_TIME

set -euo pipefail
set -x

cd $(shell_quote "$VERL_ROOT")

module purge
module load $(shell_quote "$MODULE_GCC")
module load $(shell_quote "$MODULE_CUDA")
export CC=gcc CXX=g++
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

CONDA_SH_PATH=$(shell_quote "$CONDA_SH")
if [[ $(shell_quote "$USE_CONDA") == "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "\$(conda shell.bash hook)"
    elif [[ -n "\$CONDA_SH_PATH" && -f "\$CONDA_SH_PATH" ]]; then
        source "\$CONDA_SH_PATH"
    elif [[ -n "\${CONDA_EXE:-}" ]]; then
        CONDA_BASE="\$(dirname "\$(dirname "\$CONDA_EXE")")"
        source "\$CONDA_BASE/etc/profile.d/conda.sh"
    elif [[ -f "/leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "\$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[error] could not initialize conda; pass --conda-sh or --no-conda --venv" >&2
        exit 1
    fi
    conda activate $(shell_quote "$CONDA_ENV")
elif [[ -f $(shell_quote "$VERL_VENV/bin/activate") ]]; then
    source $(shell_quote "$VERL_VENV/bin/activate")
else
    echo "[warn] venv not found: $(shell_quote "$VERL_VENV") — using current python" >&2
fi

which python3
python3 - <<'PY'
import sys
print("[info] python:", sys.executable)
PY

export VERL_ROOT=$(shell_quote "$VERL_ROOT")
export CONDA_ENV=$(shell_quote "$CONDA_ENV")
export OVEN_ROOT=$(shell_quote "$OVEN_ROOT")
export FAST=$(shell_quote "$FAST")
export DATASET_DIR=$(shell_quote "$DATASET_DIR")
export TRAIN_FILE=$(shell_quote "$TRAIN_FILE")
export VAL_FILE=$(shell_quote "$VAL_FILE")
export MODEL_PATH=$(shell_quote "$MODEL_PATH")
export REWARD_FN_PATH=$(shell_quote "$REWARD_FN_PATH")
export OVEN_TAXONOMY_INDEX=$(shell_quote "$OVEN_TAXONOMY_INDEX")
export PROJECT_NAME=$(shell_quote "$PROJECT_NAME")
export EXP_NAME=$(shell_quote "$EXP_NAME")
export CKPTS_DIR=$(shell_quote "$CKPTS_DIR")

export N_GPUS=$(shell_quote "$SLURM_GPUS")
export N_NODES=1
export TOTAL_TRAINING_STEPS=$(shell_quote "$TOTAL_TRAINING_STEPS")
export SAVE_FREQ=$(shell_quote "$SAVE_FREQ")
export TEST_FREQ=$(shell_quote "$TEST_FREQ")
export VAL_BEFORE_TRAIN=$(shell_quote "$VAL_BEFORE_TRAIN")
export LOGGER=$(shell_quote "$LOGGER")
export TRAIN_BATCH_SIZE=$(shell_quote "$TRAIN_BATCH_SIZE")
export PPO_MINI_BATCH_SIZE=$(shell_quote "$PPO_MINI_BATCH_SIZE")
export PPO_MAX_TOKEN_LEN_PER_GPU=$(shell_quote "$PPO_MAX_TOKEN_LEN_PER_GPU")
export ROLLOUT_N=$(shell_quote "$ROLLOUT_N")
export ROLLOUT_TP=$(shell_quote "$ROLLOUT_TP")
export ROLLOUT_AGENT_NUM_WORKERS=$(shell_quote "$ROLLOUT_AGENT_NUM_WORKERS")
export ROLLOUT_GPU_UTIL=$(shell_quote "$ROLLOUT_GPU_UTIL")
export ROLLOUT_ENFORCE_EAGER=$(shell_quote "$ROLLOUT_ENFORCE_EAGER")
export ROLLOUT_ENABLE_CHUNKED_PREFILL=$(shell_quote "$ROLLOUT_ENABLE_CHUNKED_PREFILL")
export ROLLOUT_FREE_CACHE_ENGINE=$(shell_quote "$ROLLOUT_FREE_CACHE_ENGINE")
export MAX_PROMPT_LENGTH=$(shell_quote "$MAX_PROMPT_LENGTH")
export MAX_RESPONSE_LENGTH=$(shell_quote "$MAX_RESPONSE_LENGTH")
export MAX_VAL_SAMPLES=$(shell_quote "$MAX_VAL_SAMPLES")
export VAL_BATCH_SIZE=$(shell_quote "$VAL_BATCH_SIZE")
export ROLLOUT_MAX_MODEL_LEN=$(shell_quote "$ROLLOUT_MAX_MODEL_LEN")
export ROLLOUT_MAX_NUM_SEQS=$(shell_quote "$ROLLOUT_MAX_NUM_SEQS")
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=$(shell_quote "$ROLLOUT_MAX_NUM_BATCHED_TOKENS")
export ROLLOUT_LIMIT_IMAGES=$(shell_quote "$ROLLOUT_LIMIT_IMAGES")
export MODEL_USE_REMOVE_PADDING=$(shell_quote "$MODEL_USE_REMOVE_PADDING")
export MODEL_USE_FUSED_KERNELS=$(shell_quote "$MODEL_USE_FUSED_KERNELS")
export MODEL_ENABLE_GRADIENT_CHECKPOINTING=$(shell_quote "$MODEL_ENABLE_GRADIENT_CHECKPOINTING")
export MODEL_ATTN_IMPLEMENTATION=$(shell_quote "$MODEL_ATTN_IMPLEMENTATION")
export ACTOR_STRATEGY=$(shell_quote "$ACTOR_STRATEGY")
export REF_STRATEGY=$(shell_quote "$REF_STRATEGY")
export ACTOR_FSDP_SIZE=$(shell_quote "$ACTOR_FSDP_SIZE")
export ACTOR_MODEL_DTYPE=$(shell_quote "$ACTOR_MODEL_DTYPE")
export REF_MODEL_DTYPE=$(shell_quote "$REF_MODEL_DTYPE")
export RAY_OBJECT_STORE_MEMORY=$(shell_quote "$RAY_OBJECT_STORE_MEMORY")
export LORA_RANK=$(shell_quote "$LORA_RANK")
export LORA_ALPHA=$(shell_quote "$LORA_ALPHA")
export LORA_MERGE=$(shell_quote "$LORA_MERGE")

export WANDB_PROJECT=$(shell_quote "$WANDB_PROJECT")
export WANDB_NAME=$(shell_quote "$WANDB_NAME")
export WANDB_GROUP=$(shell_quote "$WANDB_GROUP")
export WANDB_TAGS=$(shell_quote "$WANDB_TAGS")
export WANDB_MODE=$(shell_quote "$WANDB_MODE")
export WANDB_DIR=$(shell_quote "$WANDB_DIR")

export UV_CACHE_DIR=$(shell_quote "$UV_CACHE_DIR")
export PIP_CACHE_DIR=$(shell_quote "$PIP_CACHE_DIR")
export TMPDIR=$(shell_quote "$TMPDIR")
export HF_HOME=$(shell_quote "$HF_HOME")
export HUGGINGFACE_HUB_CACHE=$(shell_quote "$HUGGINGFACE_HUB_CACHE")
export HF_DATASETS_CACHE=$(shell_quote "$HF_DATASETS_CACHE")
export HF_HUB_OFFLINE=$(shell_quote "$HF_HUB_OFFLINE")
export HF_DATASETS_OFFLINE=$(shell_quote "$HF_DATASETS_OFFLINE")
export TRANSFORMERS_OFFLINE=$(shell_quote "$TRANSFORMERS_OFFLINE")
export VLLM_ALLREDUCE_USE_SYMM_MEM=$(shell_quote "$VLLM_ALLREDUCE_USE_SYMM_MEM")
export VLLM_USE_V1=$(shell_quote "$VLLM_USE_V1")
export HYDRA_FULL_ERROR=$(shell_quote "$HYDRA_FULL_ERROR")
export RAY_DEDUP_LOGS=$(shell_quote "$RAY_DEDUP_LOGS")
export PYTHONFAULTHANDLER=$(shell_quote "$PYTHONFAULTHANDLER")
export TORCH_SHOW_CPP_STACKTRACES=$(shell_quote "$TORCH_SHOW_CPP_STACKTRACES")
export TORCH_DISABLE_ADDR2LINE=$(shell_quote "$TORCH_DISABLE_ADDR2LINE")
export PYTORCH_NVML_BASED_CUDA_CHECK=$(shell_quote "$PYTORCH_NVML_BASED_CUDA_CHECK")
export CUDA_MODULE_LOADING=$(shell_quote "$CUDA_MODULE_LOADING")
export VERL_IMPORT_PROBE=$(shell_quote "$VERL_IMPORT_PROBE")
export KEEP_RAY_TMPDIR_ON_FAILURE=$(shell_quote "$KEEP_RAY_TMPDIR_ON_FAILURE")

mkdir -p "\$UV_CACHE_DIR" "\$PIP_CACHE_DIR" "\$TMPDIR" "\$HF_HOME" \\
    "\$HUGGINGFACE_HUB_CACHE" "\$HF_DATASETS_CACHE" "\$WANDB_DIR"

export RAY_TMPDIR=\${RAY_TMPDIR:-/tmp/r\${SLURM_JOB_ID}}
export VLLM_LOGGING_LEVEL=\${VLLM_LOGGING_LEVEL:-WARN}

echo "[info] GRPO job \$SLURM_JOB_ID on \$(hostname)"
echo "  mode:       $(shell_quote "$MODE")"
echo "  train:      \$TRAIN_FILE"
echo "  val:        \$VAL_FILE"
echo "  model:      \$MODEL_PATH"
echo "  reward:     \$REWARD_FN_PATH"
echo "  taxonomy:   \$OVEN_TAXONOMY_INDEX"
echo "  logger:     \$LOGGER"
echo "  ckpts:      \$CKPTS_DIR"
echo "  steps:      \$TOTAL_TRAINING_STEPS"
echo "  batch:      train=\$TRAIN_BATCH_SIZE ppo=\$PPO_MINI_BATCH_SIZE max_token_per_gpu=\$PPO_MAX_TOKEN_LEN_PER_GPU"
echo "  rollout:    n=\$ROLLOUT_N tp=\$ROLLOUT_TP agent_workers=\$ROLLOUT_AGENT_NUM_WORKERS gpu_util=\$ROLLOUT_GPU_UTIL seqs=\$ROLLOUT_MAX_NUM_SEQS batched_tokens=\$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "  actor/ref:  actor=\$ACTOR_STRATEGY ref=\$REF_STRATEGY fsdp_size=\$ACTOR_FSDP_SIZE dtype=\$ACTOR_MODEL_DTYPE/\$REF_MODEL_DTYPE remove_padding=\$MODEL_USE_REMOVE_PADDING fused=\$MODEL_USE_FUSED_KERNELS grad_ckpt=\$MODEL_ENABLE_GRADIENT_CHECKPOINTING attn=\$MODEL_ATTN_IMPLEMENTATION"
echo "  lora:       rank=\$LORA_RANK alpha=\$LORA_ALPHA merge=\$LORA_MERGE"
echo "  lengths:    prompt=\$MAX_PROMPT_LENGTH response=\$MAX_RESPONSE_LENGTH model=\$ROLLOUT_MAX_MODEL_LEN images=\$ROLLOUT_LIMIT_IMAGES"
echo "  ray store:  \$RAY_OBJECT_STORE_MEMORY"
echo "  vllm v1:    \$VLLM_USE_V1"
echo "  probe:      \$VERL_IMPORT_PROBE"

EOF
    printf 'EXTRA_ARGS=(\n'
    for arg in "${EXTRA_ARGS[@]}"; do
        printf '    %q\n' "$arg"
    done
    printf ')\n\n'
    cat <<'EOF'
bash examples/grpo_trainer/run_qwen3_vl_oven_rsa_trace_grpo.sh "${EXTRA_ARGS[@]}"
EOF
} > "$SBATCH_FILE"

echo "[info] wrote $SBATCH_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
    cat "$SBATCH_FILE"
    exit 0
fi

sbatch "$SBATCH_FILE"
