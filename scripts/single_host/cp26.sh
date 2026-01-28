
#!/bin/bash
set -e

# Parse model argument (default: qwen3-4b)
MODEL_ARG="${1:-qwen3-4b}"

# Parse rollouts argument (default: 512)
ROLLOUTS_ARG="${2:-512}"

# Parse epochs argument (default: 50)
NUM_EPOCHS="${3:-50}"

# Parse log directory for resuming (optional)
# Provide a log directory path to resume training in-place
# Example: ./logs/cp-26-meta-llama-Llama-3.1-8B-...-2026-01-28-06-17
RESUME_LOG_DIR="${4:-}"

case "${ROLLOUTS_ARG}" in
    512)
        groups_per_batch=8
        ;;
    1024)
        groups_per_batch=16
        ;;
    2048)
        groups_per_batch=32
        ;;
    *)
        echo "Invalid rollouts: ${ROLLOUTS_ARG}"
        echo "Valid options: 512, 1024, 2048"
        exit 1
        ;;
esac

case "${MODEL_ARG}" in
    gpt-oss-120b)
        model_name="openai/gpt-oss-120b"
        renderer_name="gpt-oss-120b"
        ;;
    qwen3-8b)
        model_name="Qwen/Qwen3-8B"
        renderer_name="qwen3"
        ;;
    qwen3-4b)
        model_name="Qwen/Qwen3-4B-Instruct-2507"
        renderer_name="qwen3"
        ;;
    llama3.2-1b)
        model_name="meta-llama/Llama-3.2-1B"
        renderer_name="llama3"
        ;;
    llama3.2-3b)
        model_name="meta-llama/Llama-3.2-3B"
        renderer_name="llama3"
        ;;
    llama3.1-8b)
        model_name="meta-llama/Llama-3.1-8B"
        renderer_name="llama3"
        ;;
    *)
        echo "Usage: $0 [gpt-oss-120b|qwen3-8b|qwen3-4b|llama3.2-1b|llama3.2-3b|llama3.1-8b] [512|1024|2048] [epochs] [resume_log_dir]"
        echo "Default: qwen3-4b 512 50"
        echo ""
        echo "To resume training, provide the log directory as the 4th argument."
        echo "Example: $0 llama3.1-8b 512 50 './logs/cp-26-meta-llama-Llama-3.1-8B-...-2026-01-28-06-17'"
        echo ""
        echo "This will continue training in the same directory, loading weights AND sampler state."
        exit 1
        ;;
esac

echo "Using model: ${model_name}, rollouts: ${ROLLOUTS_ARG} (groups_per_batch=${groups_per_batch} x group_size=64), epochs: ${NUM_EPOCHS}"

# Build log_path argument if resuming
resume_args=""
if [ -n "${RESUME_LOG_DIR}" ]; then
    if [ -d "${RESUME_LOG_DIR}" ]; then
        # Verify checkpoints exist
        CHECKPOINTS_FILE="${RESUME_LOG_DIR}/checkpoints.jsonl"
        if [ ! -f "${CHECKPOINTS_FILE}" ]; then
            echo "ERROR: checkpoints.jsonl not found in ${RESUME_LOG_DIR}"
            exit 1
        fi
        
        # Get the latest checkpoint step for display
        LATEST_CHECKPOINT=$(tail -1 "${CHECKPOINTS_FILE}")
        CHECKPOINT_STEP=$(echo "${LATEST_CHECKPOINT}" | python3 -c "import json,sys; print(json.load(sys.stdin)['batch'])")
        
        echo "Resuming in: ${RESUME_LOG_DIR}"
        echo "  Will continue from step: ${CHECKPOINT_STEP}"
        echo "  Results will be appended to existing metrics.jsonl"
        
        # Use log_path to trigger built-in auto-resume
        resume_args="log_path=${RESUME_LOG_DIR} behavior_if_log_dir_exists=resume"
    else
        echo "ERROR: ${RESUME_LOG_DIR} is not a valid directory"
        exit 1
    fi
fi

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export PYTHONPATH="${PWD}:${PWD}/tasks:${PYTHONPATH:-}"
unset RAY_ADDRESS

# Export all variables from .env (set -a makes sourced vars exported)
set -a
. .env
set +a

# Training params
common="learning_rate=4e-5 \
        adv_estimator=entropic_adaptive_beta \
        max_tokens=20000 \
        lora_rank=32 \
        num_cpus_per_task=2"

# # Launch
# python main_tinker_submitit.py --nodes "${nnodes}" \
#     --partition ${partition} \
#     --cpus-per-task ${cpus_per_node} \
#     --timeout_min 2880 \
#     env=denoising \
#     model_name="${model_name}" \
#     sampler_type=puct_backprop \
#     initial_exp_type="random"


python main_tinker_submitit.py --local \
    ${common} \
    env=cp \
    problem_idx=26 \
    model_name="${model_name}" \
    renderer_name="${renderer_name}" \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=${NUM_EPOCHS} \
    groups_per_batch=${groups_per_batch} \
    group_size=64 \
    wandb_project="ttt-discover-cp" \
    wandb_name="cp26-local-test" \
    ${resume_args} \
    2>&1 | grep -v "socket.send() raised exception"