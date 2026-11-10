"""One-thread torch setup and a single steady-state timing.

Imported after the process has pinned its BLAS threads. Each benchmark
script sets those environment variables before importing numpy or torch.
"""

import gc
import time


def pin_torch_threads():
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A previous import in this process already initialized the pool.
        pass
    return torch


def steady_call_ms(fn, warmup=1):
    """Milliseconds for one call of ``fn`` after ``warmup`` discarded calls."""
    for _ in range(warmup):
        fn()
    gc.collect()
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0
