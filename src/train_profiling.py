import time
import psutil
import logging
def profile_stage(stage_name, func, *args, **kwargs):
    """
    Profiles the execution of a pipeline stage by measuring:
    - Elapsed time
    - RAM used (in MB)
    - CPU usage (percentage)
    Args:
        stage_name (str): Name of the pipeline stage.
        func (callable): The function to execute.
        *args, **kwargs: Arguments to pass to the function.
    Returns:
        The result of func(*args, **kwargs).
    """
    logging.info(f"[PROFILE] Starting stage: {stage_name}")

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6  # MB

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time

    mem_after = process.memory_info().rss / 1e6
    cpu_after = psutil.cpu_percent(interval=None)

    logging.info(f"[PROFILE] ⤷ {stage_name} completed in {elapsed_time:.2f}s")
    logging.info(f"[PROFILE] ⤷ RAM used: {mem_after - mem_before:.2f} MB")
    logging.info(f"[PROFILE] ⤷ CPU usage during stage: {cpu_after:.2f}%")

    return result
