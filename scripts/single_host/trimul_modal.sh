#!/bin/bash
set -e

# TTT-Discover: Trimul (GPU Kernel Optimization) - Modal Cloud Execution
# Usage: bash scripts/single_host/trimul_modal.sh
#
# Prerequisites:
#   1. Authenticate with Modal: `modal token new`
#   2. Deploy the Modal app:
#      cd gpu_mode/runners
#      modal deploy modal_runner_archs.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

set -a
. .env
set +a

export PYTHONPATH="${PWD}:${PWD}/tasks:${PWD}/gpu_mode:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false  # Silence fork warning from HuggingFace tokenizers

# ============================================================================
# GPU Configuration - Modal Cloud (H100)
# ============================================================================
# Do NOT set GPU_MODE_LOCAL - this enables Modal execution
# The gpu_type (H100) is configured in env_gpu_mode.py for trimul

# ============================================================================
# API Keys - Set these in .env or uncomment below
# ============================================================================
# export TINKER_API_KEY="your-tinker-api-key"
# export WANDB_API_KEY="your-wandb-key"
# export WANDB_ENTITY="your-wandb-entity"

# ============================================================================
# Training Configuration
# ============================================================================
MODEL_NAME="openai/gpt-oss-120b"
ENV="trimul"
PROBLEM_IDX="v0"

# Batch settings (adjust based on your setup)
GROUPS_PER_BATCH=8
GROUP_SIZE=64
NUM_EPOCHS=50

# Other hyperparameters
LEARNING_RATE=4e-5
MAX_TOKENS=26000
LORA_RANK=32
ADV_ESTIMATOR="entropic_adaptive_beta"
KL_PENALTY_COEF=0.1
EVAL_TIMEOUT=300
GPU_MODE_SCORE_SCALE=3000.0

# Logging
WANDB_PROJECT="ttt-discover"
WANDB_NAME="trimul-modal-h100-$(date +%Y%m%d-%H%M%S)"

# ============================================================================
# Launch Training
# ============================================================================
echo "Starting Trimul training on Modal (H100)..."
echo "  GPU_MODE_LOCAL: not set (using Modal)"
echo "  Model: $MODEL_NAME"
echo "  Batch: ${GROUPS_PER_BATCH}x${GROUP_SIZE}"
echo ""

python main_tinker_submitit.py --local \
    env=${ENV} \
    problem_idx=${PROBLEM_IDX} \
    model_name="${MODEL_NAME}" \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=${NUM_EPOCHS} \
    groups_per_batch=${GROUPS_PER_BATCH} \
    group_size=${GROUP_SIZE} \
    learning_rate=${LEARNING_RATE} \
    max_tokens=${MAX_TOKENS} \
    lora_rank=${LORA_RANK} \
    adv_estimator=${ADV_ESTIMATOR} \
    kl_penalty_coef=${KL_PENALTY_COEF} \
    eval_timeout=${EVAL_TIMEOUT} \
    gpu_mode_score_scale=${GPU_MODE_SCORE_SCALE} \
    wandb_project="${WANDB_PROJECT}" \
    wandb_name="${WANDB_NAME}"
