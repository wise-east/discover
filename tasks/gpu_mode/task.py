import os
import threading
from gpu_mode.libkernelbot.submission import compute_score

# Use local GPU execution if GPU_MODE_LOCAL=1, otherwise use Modal
USE_LOCAL_GPU = os.environ.get("GPU_MODE_LOCAL", "0") == "1"

if USE_LOCAL_GPU:
    from gpu_mode.run_local import run_local
else:
    from gpu_mode.run_modal import run_on_modal


# Thread-safe counter for round-robin GPU assignment
_gpu_counter_lock = threading.Lock()
_gpu_counter = 0


def _get_next_gpu_id() -> int:
    """Get the next GPU ID in round-robin fashion across all available GPUs."""
    global _gpu_counter
    num_gpus = int(os.environ.get("GPU_MODE_NUM_GPUS", "1"))
    with _gpu_counter_lock:
        gpu_id = _gpu_counter % num_gpus
        _gpu_counter += 1
    return gpu_id


def get_gpu_mode_error(msg):
    return {
        "score": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "performance": -1_000_000,
    }


async def run_gpu_mode_task(generation: str, gpu_type: str, task_name: str, score_scale: float, app_name: str):
    """
    Run GPU kernel evaluation.
    
    Set GPU_MODE_LOCAL=1 to use local GPUs instead of Modal.
    Set GPU_MODE_NUM_GPUS to specify how many GPUs to use for parallel evaluation (default: 1).
    Set GPU_MODE_GPU_ID to use a single specific GPU instead of round-robin (overrides NUM_GPUS).
    """
    
    if USE_LOCAL_GPU:
        # Use specific GPU if set, otherwise round-robin across available GPUs
        if "GPU_MODE_GPU_ID" in os.environ:
            gpu_id = int(os.environ.get("GPU_MODE_GPU_ID", "0"))
        else:
            gpu_id = _get_next_gpu_id()
        
        result, task = await run_local(
            submission_code=generation,
            mode="leaderboard",
            task_name=task_name,
            gpu_id=gpu_id,
        )
    else:
        result, task = await run_on_modal(
            submission_code=generation,
            gpu_type=gpu_type,
            mode="leaderboard",
            task_name=task_name,
            app_name=app_name,
        )

    if not result.success:
        return get_gpu_mode_error(f"Error: Failed to run test: {result.error}.")
    
    # Unexpected
    if "test" not in result.runs:
        return get_gpu_mode_error(f"Unexpected result: Failed to find test results.")

    test_results = result.runs["test"]

    # Probably compile error
    if not test_results.run.success:
        return get_gpu_mode_error(f"Failed to run tests: {test_results.run.stderr}")

    # Failed test cases
    if not test_results.run.passed:
        return get_gpu_mode_error(f"Failed to pass test cases.")

    if task is not None and "leaderboard" in result.runs:
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000
            msg = f"\nOverall leaderboard score (microseconds, {task.ranking_by.value}): {score_us} us"
        except Exception as e:
            return get_gpu_mode_error(f"Could not compute leaderboard score: {e}")

    score = score_scale / score_us

    return {
        "score": score,
        "msg": msg,
        "correctness": 1.0,
        "performance": -score_us,
    }
