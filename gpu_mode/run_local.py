#!/usr/bin/env python3
"""
Local GPU runner for kernel evaluation - runs directly on local GPUs.

Supported tasks:
  - trimul: BioML TriMul task
  - mla_decode_nvidia: MLA Decode task

Usage:
    python gpu_mode/run_local.py --submission path/to/submission.py --task trimul
    python gpu_mode/run_local.py --submission path/to/submission.py --task trimul --mode test
"""

import argparse
import asyncio
import os
import tempfile
import threading
from pathlib import Path

from libkernelbot.consts import SubmissionMode
from libkernelbot.run_eval import FullResult, run_config, make_system_info
from libkernelbot.submission import compute_score
from libkernelbot.task import LeaderboardTask, build_task_config, make_task_definition


PROJECT_ROOT = Path(__file__).resolve().parent
TRIMUL_TASK_YAML = PROJECT_ROOT / "bioml" / "trimul" / "task.yml"
MLA_DECODE_NVIDIA_TASK_YAML = PROJECT_ROOT / "mla-decode" / "task.yml"


def load_task(task_name: str = "trimul") -> LeaderboardTask:
    """Load a LeaderboardTask from its YAML definition."""
    task_map = {
        "trimul": TRIMUL_TASK_YAML,
        "mla_decode_nvidia": MLA_DECODE_NVIDIA_TASK_YAML,
    }
    
    if task_name not in task_map:
        valid = ", ".join(task_map.keys())
        raise ValueError(f"Invalid task name '{task_name}'. Valid tasks: {valid}")
    
    task_yaml = task_map[task_name]
    if not task_yaml.exists():
        raise FileNotFoundError(f"Could not find task definition at {task_yaml}")
    
    definition = make_task_definition(task_yaml)
    return definition.task


async def run_local(
    submission_code: str,
    mode: str = "leaderboard",
    task_name: str = "trimul",
    gpu_id: int = 0,
) -> tuple[FullResult, LeaderboardTask]:
    """
    Run a submission locally on the specified GPU.

    Args:
        submission_code: Contents of the user's `submission.py`
        mode: One of: test, benchmark, leaderboard
        task_name: One of "trimul" or "mla_decode_nvidia"
        gpu_id: Which GPU to use (CUDA_VISIBLE_DEVICES)
    """
    # Load task from YAML
    task = load_task(task_name)

    # Map mode to enum
    try:
        mode_enum = SubmissionMode(mode)
    except ValueError as e:
        valid = ", ".join(m.value for m in SubmissionMode)
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid}") from e

    # Build config
    config = build_task_config(
        task=task,
        submission_content=submission_code,
        arch=None,
        mode=mode_enum,
    )
    
    # Set GPU for this evaluation via subprocess environment (avoids race conditions)
    config["extra_env"] = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    print(f"Running {task_name} locally on GPU {gpu_id} with mode='{mode}'...")

    # Run in a process pool to allow true parallelism across GPUs
    # Each process has its own working directory, avoiding os.chdir() conflicts
    # run_config is synchronous and would otherwise block the event loop
    result = await asyncio.get_event_loop().run_in_executor(
        _get_process_pool(),
        _run_config_in_tmpdir,
        config
    )

    return result, task


# Module-level process pool for parallel GPU evaluations
_process_pool = None
_pool_lock = threading.Lock()


def _get_process_pool():
    """Get or create the shared process pool."""
    global _process_pool
    if _process_pool is None:
        with _pool_lock:
            if _process_pool is None:
                import concurrent.futures
                import multiprocessing
                # Use "spawn" to avoid CUDA/NVML issues with forked processes
                ctx = multiprocessing.get_context("spawn")
                # Use number of GPUs as max workers (each worker can use a different GPU)
                num_gpus = int(os.environ.get("GPU_MODE_NUM_GPUS", "8"))
                _process_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_gpus,
                    mp_context=ctx
                )
    return _process_pool


def _run_config_in_tmpdir(config: dict) -> FullResult:
    """Run config in an isolated temp directory (called in subprocess)."""
    import tempfile
    import os
    from libkernelbot.run_eval import run_config
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        return run_config(config)


def print_benchmark_details(result: FullResult):
    """Print per-benchmark statistics."""
    if "leaderboard" not in result.runs:
        return

    run_res = result.runs["leaderboard"].run
    if not run_res or not run_res.result:
        return

    data = run_res.result
    if "benchmark-count" not in data:
        return

    num_benchmarks = int(data["benchmark-count"])
    print(f"\nLeaderboard benchmarks: {num_benchmarks}")
    for i in range(num_benchmarks):
        prefix = f"benchmark.{i}."
        mean_ns = float(data.get(prefix + "mean", 0.0))
        std_ns = float(data.get(prefix + "std", 0.0))
        best_ns = float(data.get(prefix + "best", 0.0))

        mean_us = mean_ns / 1e3
        std_us = std_ns / 1e3
        best_us = best_ns / 1e3

        spec = data.get(prefix + "spec", "")
        print(f"  Benchmark {i}: {spec}")
        print(f"    mean: {mean_us:.2f} μs, std: {std_us:.2f} μs, best: {best_us:.2f} μs")


def print_result(result: FullResult, task: LeaderboardTask | None = None):
    """Pretty print results."""
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")

    if not result.success:
        print(f"Error: {result.error}")
        return

    print(f"\nSystem: {result.system.gpu} ({result.system.runtime})")

    for run_name, run_result in result.runs.items():
        print(f"\n[{run_name}]")
        
        if run_result.run:
            run = run_result.run
            status = "PASS" if run.passed else "FAIL"
            print(f"  Status: {status}")
            print(f"  Duration: {run.duration:.2f}s")
            
            if run.stderr and not run.passed:
                # Truncate long errors
                stderr = run.stderr[:1000] + "..." if len(run.stderr) > 1000 else run.stderr
                print(f"  Stderr: {stderr}")

    # Print benchmark details and score
    if task is not None and "leaderboard" in result.runs:
        print_benchmark_details(result)
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            score_us = score_seconds * 1_000_000
            print(f"\n>>> Leaderboard Score: {score_us:.3f} μs ({task.ranking_by.value}) <<<")
        except Exception as e:
            print(f"\nCould not compute score: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kernel submission locally on GPU")
    parser.add_argument("--submission", "-s", required=True, help="Path to submission.py")
    parser.add_argument("--task", "-t", default="trimul", choices=["trimul", "mla_decode_nvidia"])
    parser.add_argument("--mode", "-m", default="leaderboard", choices=["test", "benchmark", "leaderboard"])
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID to use (default: 0)")
    return parser.parse_args()


async def main():
    args = parse_args()

    submission_path = Path(args.submission)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_path}")

    submission_code = submission_path.read_text()

    result, task = await run_local(
        submission_code=submission_code,
        mode=args.mode,
        task_name=args.task,
        gpu_id=args.gpu,
    )

    print_result(result, task)
    
    # Return score for programmatic use
    if task and "leaderboard" in result.runs:
        try:
            score_seconds = compute_score(result, task, submission_id=-1)
            return score_seconds * 1_000_000
        except:
            pass
    return None


if __name__ == "__main__":
    asyncio.run(main())
