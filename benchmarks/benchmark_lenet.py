"""LeNet inference micro-benchmark.

Measures per-batch latency and throughput after a warmup phase using the
training split (same complexity as test) and optional accuracy on a subset of
the test set. Supports precision selection and loading a trained checkpoint.

Key flags:
    --weights <path>       Load saved state_dict/ checkpoint before timing
    --eval-accuracy        Report test accuracy over first N batches (--acc-batches)
    --precision fp32|fp16|bf16  Mixed precision via autocast when supported

Example:
    python benchmarks/benchmark_lenet.py --device cuda --batch-size 256 \
            --warmup 5 --iters 50 --weights checkpoints/lenet_mnist.pt --eval-accuracy
"""

import torch, time, sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from models import LeNet  # type: ignore
from data import get_mnist
from utils.device import autocast_context, auto_device  # type: ignore

def benchmark_lenet_inference(device='auto', batch_size=128, warmup=20, iters=100, precision='fp32', limit_batches=None, eval_accuracy=False, acc_batches=50, weights: str|None=None):
    if device == 'auto':
        device = auto_device()
    model = LeNet().to(device)
    # Optional load pretrained weights (expects checkpoint with 'state_dict' or raw state_dict)
    if weights:
        ckpt = torch.load(weights, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights} (missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()
    train_loader, test_loader = get_mnist('./data', batch_size=batch_size)
    loader = train_loader  # timing on train set (similar complexity)
    timings = []
    seen_samples = 0

    def batch_iter():
        for bi, (images, _) in enumerate(loader):
            if limit_batches is not None and bi >= limit_batches:
                break
            yield images

    # Warmup
    with torch.no_grad():
        for i, images in enumerate(batch_iter()):
            if i >= warmup:
                break
            images = images.to(device, non_blocking=True)
            with autocast_context(device, precision):
                _ = model(images)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Timed iterations
    with torch.no_grad():
        for i, images in enumerate(batch_iter()):
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
            seen_samples += images.size(0)

    import statistics
    mean = statistics.fmean(timings)
    p50 = statistics.median(timings)
    p95 = statistics.quantiles(timings, n=100)[94] if len(timings) >= 20 else max(timings)
    throughput = (seen_samples / len(timings)) / mean  # samples/sec
    print(f"Device={device} Precision={precision} Batch={batch_size} Iter={len(timings)}")
    print(f"Mean: {mean*1000:.3f} ms  Median: {p50*1000:.3f} ms  P95: {p95*1000:.3f} ms  Throughput: {throughput:.1f} samples/s")

    if eval_accuracy:
        # Evaluate top-1 accuracy on test set (limited batches for speed)
        correct = total = 0
        with torch.no_grad():
            for bi, (images, labels) in enumerate(test_loader):
                if acc_batches is not None and bi >= acc_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)
                with autocast_context(device, precision):
                    logits = model(images)
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        if total:
            acc = 100.0 * correct / total
            print(f"Test Accuracy (first {min(total, acc_batches*batch_size if acc_batches else total)} samples): {acc:.2f}%")

def parse_args():
    ap = argparse.ArgumentParser(description='LeNet inference micro-benchmark')
    ap.add_argument('--device', default='auto', help='cuda|cpu|mps|xpu|auto')
    ap.add_argument('--batch-size', '-b', type=int, default=128)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--precision', default='fp32', help='fp32|fp16|bf16')
    ap.add_argument('--limit-batches', type=int, default=None, help='Cap dataset batches for timing (debug)')
    ap.add_argument('--eval-accuracy', action='store_true', help='Also compute test accuracy (subset)')
    ap.add_argument('--acc-batches', type=int, default=50, help='Number of test batches for accuracy (set -1 for all)')
    ap.add_argument('--weights', type=str, default=None, help='Path to model checkpoint (optional)')
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    acc_batches = None if args.acc_batches == -1 else args.acc_batches
    benchmark_lenet_inference(device=args.device,
                              batch_size=args.batch_size,
                              warmup=args.warmup,
                              iters=args.iters,
                              precision=args.precision,
                              limit_batches=args.limit_batches,
                              eval_accuracy=args.eval_accuracy,
                              acc_batches=acc_batches,
                              weights=args.weights)
