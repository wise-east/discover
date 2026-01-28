
#!/bin/bash
set -e

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="${PWD}:${PWD}/tasks:${PYTHONPATH:-}"
unset RAY_ADDRESS

# Export all variables from .env (set -a makes sourced vars exported)
set -a
. .env
set +a

# Cluster config
model_name="Qwen/Qwen3-8B"

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
    renderer_name=qwen3 \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=50 \
    groups_per_batch=8 \
    group_size=64 \
    wandb_project="ttt-discover" \
    wandb_name="cp26-local-test" \
    2>&1 | grep -v "socket.send() raised exception"