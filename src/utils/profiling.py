import time, os, psutil, torch
from typing import Callable, Tuple, Any
from contextlib import contextmanager

def time_fn(fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - start

@contextmanager
def memory_monitor(device: str):
    """Context manager capturing host RSS delta and peak device memory (CUDA only)."""
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    yield
    rss_after = process.memory_info().rss
    peak_cuda = None
    if device == 'cuda' and torch.cuda.is_available():
        peak_cuda = torch.cuda.max_memory_allocated()
    memory_monitor.result = {  # type: ignore
        'host_mem_delta': rss_after - rss_before,
        'device_peak_mem': peak_cuda
    }

__all__ = ['time_fn','memory_monitor']
