#!/bin/bash
set -e

# TTT-Discover: Trimul (GPU Kernel Optimization)
# Usage:
#   bash scripts/single_host/trimul.sh --eval-env local   [extra hydra overrides...]
#   bash scripts/single_host/trimul.sh --eval-env modal   [extra hydra overrides...]
#
# Notes:
# - eval_env=local uses local GPUs (sets GPU_MODE_LOCAL=1)
# - eval_env=modal uses Modal (ensures GPU_MODE_LOCAL is unset)
# - Any remaining args are forwarded to `tinker_cookbook.recipes.ttt.train` as Hydra overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

set -a
. .env
set +a

export PYTHONPATH="${PWD}:${PWD}/tasks:${PWD}/gpu_mode:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false  # Silence fork warning from HuggingFace tokenizers

# ============================================================================
# Argument parsing
# ============================================================================
EVAL_ENV="local"
MODEL="gpt-oss-120b"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-env)
            EVAL_ENV="${2:-}"
            shift 2
            ;;
        --eval-env=*)
            EVAL_ENV="${1#*=}"
            shift 1
            ;;
        eval_env=*)
            EVAL_ENV="${1#*=}"
            shift 1
            ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            shift 1
            ;;
        model=*)
            MODEL="${1#*=}"
            shift 1
            ;;
        -h|--help)
            echo "Usage: bash scripts/single_host/trimul.sh --eval-env {local|modal} [hydra overrides...]"
            echo "Examples:"
            echo "  bash scripts/single_host/trimul.sh --eval-env local seed=0"
            echo "  bash scripts/single_host/trimul.sh --eval-env modal seed=0"
            echo ""
            echo "Model selection:"
            echo "  --model {gpt-oss-120b|qwen3-8b}   (default: gpt-oss-120b)"
            exit 0
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift 1
            ;;
    esac
done

if [[ "${EVAL_ENV}" != "local" && "${EVAL_ENV}" != "modal" ]]; then
    echo "Error: --eval-env must be one of {local|modal}. Got: '${EVAL_ENV}'" >&2
    exit 1
fi

case "${MODEL}" in
    gpt-oss-120b|openai/gpt-oss-120b)
        MODEL_NAME="openai/gpt-oss-120b"
        RENDERER_NAME="gpt_oss_high_reasoning"
        ;;
    qwen3-8b|Qwen/Qwen3-8B)
        MODEL_NAME="Qwen/Qwen3-8B"
        RENDERER_NAME="qwen3"
        ;;
    *)
        echo "Error: --model must be one of {gpt-oss-120b|qwen3-8b}. Got: '${MODEL}'" >&2
        exit 1
        ;;
esac

# ============================================================================
# GPU Configuration
# ============================================================================
if [[ "${EVAL_ENV}" == "local" ]]; then
    export GPU_MODE_LOCAL=1                # Use local GPUs instead of Modal
    export GPU_MODE_NUM_GPUS="${GPU_MODE_NUM_GPUS:-8}"  # Default to 8 GPUs if not set
    # export GPU_MODE_GPU_ID=0             # Uncomment to pin a single GPU instead of round-robin
else
    # Ensure we do not accidentally run local GPU eval when we meant Modal.
    unset GPU_MODE_LOCAL
    unset GPU_MODE_NUM_GPUS
    unset GPU_MODE_GPU_ID
fi

# ============================================================================
# API Keys - Set these before running!
# ============================================================================
# export TINKER_API_KEY="your-tinker-api-key"
# export WANDB_API_KEY="your-wandb-key"
# export WANDB_ENTITY="your-wandb-entity"

# ============================================================================
# Training Configuration
# ============================================================================
ENV="trimul"
PROBLEM_IDX="v0"

# Batch settings (adjust based on your setup)
GROUPS_PER_BATCH=8
GROUP_SIZE=64
NUM_EPOCHS=50

# Other hyperparameters
LEARNING_RATE=4e-5
MAX_TOKENS=26000
# Two-phase sampling is intended for gpt-oss (teacher-forcing to final when "thinking" exhausts).
# Default it ON for gpt-oss, OFF otherwise. Can be overridden via extra Hydra args.
if [[ "${MODEL_NAME}" == "openai/gpt-oss-120b" ]]; then
    TWO_PHASE_SAMPLING=true
else
    TWO_PHASE_SAMPLING=false
fi
PHASE1_MAX_TOKENS=26000
LORA_RANK=32
ADV_ESTIMATOR="entropic_adaptive_beta"
KL_PENALTY_COEF=0.1
EVAL_TIMEOUT=300
GPU_MODE_SCORE_SCALE=3000.0

# Logging
WANDB_PROJECT="ttt-discover-trimul"
WANDB_NAME="trimul-${EVAL_ENV}-h100-$(date +%Y%m%d-%H%M%S)"
MODEL_NAME_SAFE="${MODEL_NAME//\//-}"
LOG_PATH="./logs/${ENV}-${PROBLEM_IDX}-${MODEL_NAME_SAFE}-${EVAL_ENV}-$(date +%Y%m%d-%H%M%S)"

# ============================================================================
# Launch Training
# ============================================================================
echo "Starting Trimul training (eval_env=${EVAL_ENV})..."
if [[ "${EVAL_ENV}" == "local" ]]; then
    echo "  GPU_MODE_LOCAL=$GPU_MODE_LOCAL"
    echo "  GPU_MODE_NUM_GPUS=$GPU_MODE_NUM_GPUS"
else
    echo "  GPU_MODE_LOCAL: not set (using Modal)"
fi
echo "  Model: $MODEL_NAME"
echo "  Renderer: $RENDERER_NAME"
echo "  Batch: ${GROUPS_PER_BATCH}x${GROUP_SIZE}"
echo ""

python main_tinker_submitit.py --local \
    env=${ENV} \
    problem_idx=${PROBLEM_IDX} \
    model_name="${MODEL_NAME}" \
    renderer_name="${RENDERER_NAME}" \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=${NUM_EPOCHS} \
    groups_per_batch=${GROUPS_PER_BATCH} \
    group_size=${GROUP_SIZE} \
    learning_rate=${LEARNING_RATE} \
    max_tokens=${MAX_TOKENS} \
    dynamic_max_tokens=true \
    two_phase_sampling=${TWO_PHASE_SAMPLING} \
    phase1_max_tokens=${PHASE1_MAX_TOKENS} \
    lora_rank=${LORA_RANK} \
    adv_estimator=${ADV_ESTIMATOR} \
    kl_penalty_coef=${KL_PENALTY_COEF} \
    eval_timeout=${EVAL_TIMEOUT} \
    gpu_mode_score_scale=${GPU_MODE_SCORE_SCALE} \
    log_path="${LOG_PATH}" \
    wandb_project="${WANDB_PROJECT}" \
    wandb_name="${WANDB_NAME}" \
    "${FORWARD_ARGS[@]}"
