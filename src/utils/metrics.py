import torch, time, statistics
from typing import List, Dict

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, preds = outputs.max(1)
    return (preds == targets).float().mean().item()

class TimingStats:
    """Aggregate timing samples and compute summary statistics (ms)."""
    def __init__(self, samples: List[float]):
        if not samples:
            raise ValueError("samples list is empty")
        self.samples = samples
        self.count = len(samples)
        self.mean = statistics.fmean(samples)
        self.median = statistics.median(samples)
        # Use p95 if enough samples else max
        self.p95 = statistics.quantiles(samples, n=100)[94] if self.count >= 20 else max(samples)
        self.min = min(samples)
        self.max = max(samples)

    def to_dict(self) -> Dict[str, float]:
        return {
            'count': self.count,
            'mean_ms': self.mean * 1000,
            'median_ms': self.median * 1000,
            'p95_ms': self.p95 * 1000,
            'min_ms': self.min * 1000,
            'max_ms': self.max * 1000,
        }

class Timer:
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self._t0

__all__ = ['accuracy','TimingStats','Timer']
