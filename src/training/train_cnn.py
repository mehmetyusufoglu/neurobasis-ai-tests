"""General CNN training utility for small vision benchmarks (LeNet / ResNet).

Purpose
-------
Minimal, reproducible reference training loop to produce checkpoints for
inference benchmarking and cross-language (Python vs Alpaka3 C++) parity tests.

Pipeline Steps
--------------
1. Select device (CUDA -> XPU -> MPS -> CPU).
2. Load dataset (MNIST padded to 32x32 for LeNetClassic; or CIFAR10) via
     helper loaders in `data`.
3. Instantiate model (`lenet` or `resnet18` / other resnet*).
4. Create optimizer (Adam) + loss (CrossEntropyLoss).
5. Train epoch: forward -> loss -> backward -> optimizer.step().
6. Eval epoch: forward-only with metrics.
7. Append per-epoch JSON record to results file (`--out`).
8. Save best checkpoint (`--save-path`) keyed on evaluation accuracy.

Checkpoint Format
-----------------
torch.save({'model': name, 'dataset': dataset, 'epoch': N, 'state_dict': model.state_dict(), 'metrics': last_record})
The benchmark script can load either this dict or a raw state_dict.

Determinism
-----------
`--seed` sets Python + Torch CPU/GPU RNG and enforces deterministic CuDNN.
This may reduce ultimate peak performance slightly but ensures repeatability.

Typical LeNet (MNIST) Run
-------------------------
python src/training/train_cnn.py --model lenet --dataset mnist --epochs 5 \
    --batch-size 128 --save-path checkpoints/lenet_mnist.pt --out results_lenet.jsonl

Expected LeNet test accuracy: ~96-97% after 1 epoch, ~98-99% after 5 epochs.

Benchmark Usage
---------------
After training: pass `--weights checkpoints/lenet_mnist.pt` to
`benchmarks/benchmark_lenet.py` to measure optimized inference latency.

Lightweight by design: No LR scheduling, mixed precision, or augmentation —
add as needed for extended experiments.
"""
import time, json, os, random, sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict

# Allow running as a standalone script without installing the package
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from models import LeNet, get_resnet  # type: ignore
from data import get_mnist, get_cifar10  # type: ignore


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == 'lenet':
        return LeNet(num_classes)
    elif name.startswith('resnet'):
        return get_resnet(name, pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model {name}')


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()
    for batch in loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        pred = outputs.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return {
        'loss': total_loss / total,
        'acc': correct / total,
        'time': time.time() - start
    }


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return {
        'loss': total_loss / total,
        'acc': correct / total,
        'time': time.time() - start
    }


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cnn(model_name: str='lenet',
            dataset: str='mnist',
            epochs: int=1,
            batch_size: int=64,
            data_dir: str='./data',
            out: str='results_cnn.jsonl',
            save_path: str|None=None,
            lr: float=1e-3,
            seed: int=42):
    set_seed(seed)
    device = select_device()
    num_classes = 10
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist(data_dir, batch_size)
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10(data_dir, batch_size)
    else:
        raise ValueError('dataset must be mnist or cifar10')

    model = build_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics = []
    best_acc = -1.0
    for epoch in range(1, epochs+1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        ev = eval_epoch(model, test_loader, criterion, device)
        record = {'epoch': epoch, 'train': tr, 'eval': ev, 'model': model_name, 'dataset': dataset, 'device': str(device)}
        metrics.append(record)
        with open(out, 'a') as f:
            f.write(json.dumps(record) + '\n')
        # Checkpoint best
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if ev['acc'] > best_acc:
                best_acc = ev['acc']
                torch.save({'model': model_name,
                            'dataset': dataset,
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'metrics': record}, save_path)
                print(f"Saved new best checkpoint (acc={best_acc*100:.2f}%) -> {save_path}")
    return metrics

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='lenet')
    p.add_argument('--dataset', default='mnist')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--out', default='results_cnn.jsonl')
    p.add_argument('--save-path', default='checkpoints/lenet_mnist.pt')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run_cnn(args.model, args.dataset, args.epochs, args.batch_size, args.data_dir, args.out, args.save_path, args.lr, args.seed)
