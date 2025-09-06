"""LeNet model variants used for benchmarking and cross-language parity tests.

We expose two canonical variants to mirror different common implementations and
to align with the Alpaka3 C++ reference you provided.

Variant A (LeNetClassic)  -- Matches your Alpaka description
Input: 1x32x32 (MNIST padded by 2 pixels on each side)
  Conv1: 1 -> 6, kernel 5, stride 1, pad 0  => 28x28x6
  MaxPool 2x2 stride 2                       => 14x14x6
  Conv2: 6 -> 16, kernel 5, stride 1, pad 0 => 10x10x16
  MaxPool 2x2 stride 2                       => 5x5x16
  Flatten (400) -> FC1 120 -> FC2 84 -> FC3 10
Activation: ReLU (modern replacement for original tanh); pooling: MaxPool.

Variant B (LeNetAvg)     -- Previous Python version (28x28 input)
Input: 1x28x28
  Conv1: padding=2 keeps spatial 28x28
  AvgPool -> 14x14
  Conv2 (no pad) -> 10x10
  AvgPool -> 5x5 (same flattened size 400)
  FC stack identical.

Benchmarks default to LeNetClassic via the alias `LeNet` to ensure parity with
the Alpaka3 implementation (32x32 pathway, MaxPool).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetClassic(nn.Module):
    """Alpaka-aligned LeNet (expects 32x32 input, uses MaxPool)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)   # 32 -> 28
        self.pool = nn.MaxPool2d(2, 2)               # 28 -> 14, 10 -> 5
        self.conv2 = nn.Conv2d(6, 16, 5)             # 14 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNetAvg(nn.Module):
    """Avg-pooling variant (accepts 28x28 MNIST directly)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)   # 28 -> 28
        self.conv2 = nn.Conv2d(6, 16, 5)             # 14 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)   # 10 -> 5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Public alias used elsewhere (benchmarks pick the classic / Alpaka-aligned one)
LeNet = LeNetClassic

__all__ = [
    'LeNetClassic', 'LeNetAvg', 'LeNet'
]
