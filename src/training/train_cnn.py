import time, json, os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict
from ..models import LeNet, get_resnet
from ..data import get_mnist, get_cifar10


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


def run_cnn(model_name: str='lenet', dataset: str='mnist', epochs: int=1, batch_size: int=64, data_dir: str='./data', out: str='results_cnn.jsonl'):
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    metrics = []
    for epoch in range(1, epochs+1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        ev = eval_epoch(model, test_loader, criterion, device)
        record = {'epoch': epoch, 'train': tr, 'eval': ev, 'model': model_name, 'dataset': dataset, 'device': str(device)}
        metrics.append(record)
        with open(out, 'a') as f:
            f.write(json.dumps(record) + '\n')
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
    args = p.parse_args()
    run_cnn(args.model, args.dataset, args.epochs, args.batch_size, args.data_dir, args.out)
