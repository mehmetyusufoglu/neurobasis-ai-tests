import torch, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from models import LeNet  # type: ignore
from data import get_mnist

def benchmark_lenet_inference(device='cpu', batch_size=32):
    model = LeNet().to(device).eval()
    loader, _ = get_mnist('./data', batch_size=batch_size)
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
    for images, labels in loader:
            images = images.to(device)
            start = time.time()
            outputs = model(images)
            end = time.time()
            total_time += (end - start)
            total_samples += images.size(0)
    print(f"LeNet Inference: {total_samples} samples, Total time: {total_time:.4f}s, Avg/sample: {total_time/total_samples:.6f}s")

if __name__ == "__main__":
    benchmark_lenet_inference(device='cpu', batch_size=32)
