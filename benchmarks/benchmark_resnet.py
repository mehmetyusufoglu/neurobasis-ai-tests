"""ResNet (default: ResNet18) CIFAR-10 inference micro-benchmark.

Features:
  * Auto / selectable device (cuda|xpu|mps|cpu)
  * Warmup + timed iteration loops with CUDA sync for accurate latency
  * Mixed precision (fp16/bf16) via autocast where supported
  * Reports mean / median / p95 latency per batch and throughput (samples/s)

Example:
  python benchmarks/benchmark_resnet.py --device cuda --batch-size 128 \
      --warmup 10 --iters 50 --precision fp16

 python benchmarks/benchmark_resnet.py --device cuda --batch-size 128 --warmup 5 --iters 20 --precision fp16 --model resnet18
"""

import torch, time, sys, argparse, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from models import get_resnet  # type: ignore
from data import get_cifar10  # type: ignore
from utils.device import auto_device, autocast_context  # type: ignore


def benchmark_resnet_inference(device='auto', batch_size=128, warmup=10, iters=50, precision='fp32', limit_batches=None, model_name='resnet18'):
    if device == 'auto':
        device = auto_device()
    train_loader, test_loader = get_cifar10('./data', batch_size=batch_size)
    model = get_resnet(model_name, pretrained=False, num_classes=10).to(device).eval()

    def batch_iter(loader):
        for bi, (images, _) in enumerate(loader):
            if limit_batches is not None and bi >= limit_batches:
                break
            yield images

    timings = []

    # Warmup (use train set for diversity)
    with torch.no_grad():
        for i, images in enumerate(batch_iter(train_loader)):
            if i >= warmup:
                break
            images = images.to(device, non_blocking=True)
            with autocast_context(device, precision):
                _ = model(images)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Timed (use test set for stability)
    seen = 0
    with torch.no_grad():
        for i, images in enumerate(batch_iter(test_loader)):
            if i >= iters:
                break
            images = images.to(device, non_blocking=True)
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with autocast_context(device, precision):
                _ = model(images)
            if device == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - start
            timings.append(dt)
            seen += images.size(0)

    mean = statistics.fmean(timings)
    median = statistics.median(timings)
    p95 = statistics.quantiles(timings, n=100)[94] if len(timings) >= 20 else max(timings)
    throughput = (seen / len(timings)) / mean
    print(f"Model={model_name} Device={device} Precision={precision} Batch={batch_size} Iter={len(timings)}")
    print(f"Mean {mean*1000:.3f} ms  Median {median*1000:.3f} ms  P95 {p95*1000:.3f} ms  Throughput {throughput:.1f} samples/s")


def parse_args():
    ap = argparse.ArgumentParser(description='ResNet CIFAR10 inference benchmark')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--precision', default='fp32', help='fp32|fp16|bf16')
    ap.add_argument('--limit-batches', type=int, default=None)
    ap.add_argument('--model', default='resnet18', help='resnet18|resnet34|resnet50 etc.')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    benchmark_resnet_inference(device=args.device,
                               batch_size=args.batch_size,
                               warmup=args.warmup,
                               iters=args.iters,
                               precision=args.precision,
                               limit_batches=args.limit_batches,
                               model_name=args.model)
