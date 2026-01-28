
#!/bin/bash
set -e

# Parse model argument (default: qwen3-4b)
MODEL_ARG="${1:-qwen3-4b}"

# Parse rollouts argument (default: 512)
ROLLOUTS_ARG="${2:-512}"

# Parse epochs argument (default: 50)
NUM_EPOCHS="${3:-50}"

# Parse checkpoint path for resuming (optional)
CHECKPOINT_PATH="${4:-}"

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
        echo "Usage: $0 [gpt-oss-120b|qwen3-8b|qwen3-4b|llama3.2-1b|llama3.2-3b|llama3.1-8b] [512|1024|2048] [epochs] [checkpoint_path]"
        echo "Default: qwen3-4b 512 50"
        echo ""
        echo "To resume from a checkpoint, provide the checkpoint path as the 4th argument."
        echo "Example: $0 llama3.1-8b 512 50 'tinker://xxx/weights/000025'"
        exit 1
        ;;
esac

# Build checkpoint argument if provided
if [ -n "${CHECKPOINT_PATH}" ]; then
    checkpoint_arg="load_checkpoint_path=${CHECKPOINT_PATH}"
    echo "Using model: ${model_name}, rollouts: ${ROLLOUTS_ARG} (groups_per_batch=${groups_per_batch} x group_size=64), epochs: ${NUM_EPOCHS}"
    echo "Resuming from checkpoint: ${CHECKPOINT_PATH}"
else
    checkpoint_arg=""
    echo "Using model: ${model_name}, rollouts: ${ROLLOUTS_ARG} (groups_per_batch=${groups_per_batch} x group_size=64), epochs: ${NUM_EPOCHS}"
fi

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
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
    wandb_project="ttt-discover" \
    wandb_name="cp26-local-test" \
    ${checkpoint_arg} \
    2>&1 | grep -v "socket.send() raised exception"