## Repository Layout (Inference Benchmark Suite)

```
ai_model_tests/
  benchmarks/         # Executable scripts for model inference timing
  src/
    models/           # Model constructors / lightweight wrappers (LeNet, ResNet, BERT, GPT2)
    data/             # Dataset loaders (MNIST, CIFAR10, text dataset helpers)
    utils/            # Device selection, precision/AMP helpers, metrics, profiling
    training/         # Optional training script (not required for inference benchmarks)
```

No deprecated modules: all code lives under `src/`.
AI Model Inference Benchmark Suite
==================================

Purpose
-------
Lightweight, reproducible PyTorch-based inference benchmarks for classic CNN (LeNet, ResNet50) and LLM (BERT, GPT-2) models across heterogeneous HPC GPU/CPU platforms (NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS, CPU). Results can be compared with Alpaka3 C++ AI kernel implementations.

Key Features
------------
- Synthetic or real sample data (MNIST for LeNet, random tensors otherwise) to isolate kernel performance.
- Unified CLI for latency & throughput with warmup, precision selection, batch sweep.
- Device auto-detection (cuda / rocm, xpu, mps, cpu).
- Optional JSON + TSV result export for aggregation.
- Memory + parameter counts + simple FLOPs (where easily computable / estimated).

Quick Start
-----------
**Setup (run these commands from the `ai_model_tests/` directory):**
```bash
# 1. Navigate to the project directory first
cd ai_model_tests/

# 2. Create and activate virtual environment
# Check if environment exists: ls -la .venv/
# Check if already active: echo $VIRTUAL_ENV
python -m venv .venv && source .venv/bin/activate

# If .venv already exists, just activate:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Individual Model Benchmarks:**
```bash
# LeNet on MNIST (with trained checkpoint)
python benchmarks/benchmark_lenet.py --device cuda --eval-accuracy --weights checkpoints/lenet_mnist.pt

# ResNet18 on CIFAR-10 
python benchmarks/benchmark_resnet.py --device cuda --batch-size 128 --iters 50

# Train a model first (optional - checkpoints provided)
python src/training/train_cnn.py --model lenet --dataset mnist --epochs 1 --save-path checkpoints/lenet_mnist.pt
```

**Unified Benchmark Script:**
```bash
# Single run (ResNet50, batch 8, 50 timed iterations)
python scripts/benchmark_inference.py --model resnet50 --batch-size 8 --iters 50

# BERT base, sequence length 128, batch 4
python scripts/benchmark_inference.py --model bert-base-uncased --sequence-length 128 --batch-size 4

# GPT2 small, fp16 autocast if supported, output JSON
python scripts/benchmark_inference.py --model gpt2 --precision fp16 --sequence-length 128 --batch-size 2 --json results/gpt2_fp16.json

# Sweep batch sizes for ResNet50
python scripts/benchmark_inference.py --model resnet50 --batch-size 1 2 4 8 16 --iters 100 --export-tsv results/resnet_sweep.tsv
```

## Data Management

**Dataset Download & Storage:**
All datasets are automatically downloaded on first use and stored in the `data/` directory.

**Available Data Loaders (`src/data/`):**
- `mnist.py`: MNIST handwritten digits (28x28 → padded to 32x32)
  - Auto-downloads from `http://yann.lecun.com/exdb/mnist/`
  - Used by: LeNet training/benchmarking
  - Storage: `data/MNIST/raw/` (~50MB)
  
- `cifar10.py`: CIFAR-10 natural images (32x32, 10 classes)
  - Auto-downloads from `https://www.cs.toronto.edu/~kriz/cifar.html`
  - Used by: ResNet training/benchmarking
  - Storage: `data/cifar-10-batches-py/` (~170MB)
  - Includes data augmentation (RandomCrop, RandomHorizontalFlip)

- `text.py`: Text dataset utilities for BERT/GPT-2
  - Placeholder for tokenized text datasets
  - Most benchmarks use synthetic/random tokens

**How Data Loaders Are Called:**
```python
# In training scripts (src/training/train_cnn.py):
from data import get_mnist, get_cifar10

# MNIST with 32x32 padding for LeNet
train_loader, test_loader = get_mnist(data_dir="./data", batch_size=128)

# CIFAR-10 with augmentation for ResNet
train_loader, test_loader = get_cifar10(data_dir="./data", batch_size=128)
```

**Manual Data Directory Setup (optional):**
```bash
# Pre-create data directory
mkdir -p data/

# First run will auto-download:
python src/training/train_cnn.py --model lenet --dataset mnist --epochs 1
```

Directory Layout
----------------
```
ai_model_tests/
  src/
    data/          # Dataset loaders (mnist.py, cifar10.py, text.py)
    models/        # Model definitions / wrappers  
    utils/         # Benchmark + device helpers
    training/      # Training loops
  benchmarks/      # Individual model benchmark scripts
  scripts/         # Unified benchmark frontend
  data/            # Downloaded datasets (auto-created, gitignored)
    MNIST/         # MNIST dataset files
    cifar-10-batches-py/  # CIFAR-10 dataset files
  checkpoints/     # Trained model weights
  results/         # Benchmark output files (gitignored)
```

Extending
---------
Add a new model:
1. Create `models/your_model.py` exposing `build_model(name: str, **kwargs)` or a class.
2. Register inside `models/__init__.py` in `MODEL_BUILDERS`.
3. Provide default input shape logic in `infer_default_inputs` within `scripts/benchmark_inference.py` if special.

Notes on Precision
------------------
`--precision` controls an autocast context (fp32, fp16, bf16). If unsupported on the chosen device it falls back to fp32 with a warning.

Reproducibility
---------------
Set seeds via `--seed` (affects synthetic input generation only; does not change pretrained weights).

Planned (Next Steps)
--------------------
- Add FLOP counting (torch.profiler, fvcore optional).
- Add torch.compile() toggle for PyTorch 2.x.
- Add ONNX export & runtime comparison.
- Integrate energy measurement hooks (e.g., NVML / ROCm SMI) if available.
- Add multi-stream and multi-process scaling scripts.

License
-------
MIT (add if appropriate).
# AI Model Tests

Benchmark common CNN and LLM reference implementations (PyTorch) across heterogeneous HPC GPU/CPU systems and compare with Alpaka3 AI kernels.

## Goals
- Provide minimal, clean reference training & inference loops for: LeNet (MNIST), ResNet (CIFAR10 / ImageNet subset), BERT (masked LM), GPT-2 (causal LM).
- Measure: throughput (samples/sec, tokens/sec), latency, memory footprint, FLOP-estimates, energy hooks (placeholder), reproducibility metadata.
- Support multiple accelerators (NVIDIA / AMD / Intel GPUs, CPU fallback) via PyTorch device abstraction.
- Export structured JSON/CSV logs for later comparison with Alpaka3 C++ kernels.

## Layout
```
src/
  data/            Dataset + datamodule utilities
  models/          Model definitions & wrappers
  training/        Generic train/eval loops
  benchmarks/      Benchmark runners producing metrics
config/            YAML config files (hyperparams, model variants)
scripts/           Helper shell/python scripts
tests/             Basic sanity tests
```

## Quick Start
```bash
# 1. Clone or navigate to the project directory
cd ai_model_tests/

# 2. Setup Python environment
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 3. Run a quick benchmark test
python benchmarks/benchmark_lenet.py --device cuda --eval-accuracy --weights checkpoints/lenet_mnist.pt

# 4. Or train a model first, then benchmark
python src/training/train_cnn.py --model lenet --dataset mnist --epochs 1
```

## Example Commands
```bash
# Example: Train LeNet on MNIST
python src/training/train_cnn.py --model lenet --dataset mnist --epochs 1

# Example: Benchmark GPT-2 inference  
python scripts/benchmark_inference.py --model gpt2 --max-length 64 --batch-size 2
```

## HPC Notes
- Uses `torch.cuda.is_available()`, `torch.backends.mps` (future), and `torch.xpu` (Intel Extension if installed) detection.
- Set `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK` externally as needed.
- For deterministic CNN runs (baseline): set `export CUBLAS_WORKSPACE_CONFIG=:16:8` on NVIDIA.

## Benchmark Output
JSON lines (*.jsonl) with: system hw summary (collected), model name, config hash, epoch stats, per-iteration timing (p50/p95), throughput, tokens/s, memory (max allocated), commit hash (if git repo).

## Next Steps
- Add energy measurement integration (e.g., NVML / ROCm SMI / RAPL) wrappers.
- Add Alpaka3 interop doc pointer.

## License
MIT (placeholder)
