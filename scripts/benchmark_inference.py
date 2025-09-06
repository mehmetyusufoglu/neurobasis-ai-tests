#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable

import torch

# Ensure src/ is on path (single canonical code tree)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import LeNet, get_resnet, load_bert, load_gpt2  # type: ignore  # noqa: E402
from utils.device import auto_device, device_properties, autocast_context  # noqa: E402
from utils.metrics import TimingStats, Timer  # noqa: E402
from utils.profiling import memory_monitor  # noqa: E402


# Factory mapping for supported models
def _model_factories() -> Dict[str, Callable[[], torch.nn.Module]]:
    return {
        'lenet': lambda: LeNet(),
        'resnet50': lambda: get_resnet('resnet50', pretrained=False, num_classes=1000),
        'bert-base-uncased': lambda: load_bert()[0],  # returns (model, tokenizer)
        'gpt2': lambda: load_gpt2()[0],
    }


def get_model(name: str) -> torch.nn.Module:
    factories = _model_factories()
    if name not in factories:
        raise ValueError(f"Unknown model '{name}'. Available: {list(factories)}")
    return factories[name]()


def model_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args():
    p = argparse.ArgumentParser(description="Unified inference benchmark")
    p.add_argument("--model", required=True, help="Model name: lenet|resnet50|bert-base-uncased|gpt2")
    p.add_argument("--batch-size", "-b", nargs="+", type=int, default=[1], help="One or more batch sizes")
    p.add_argument("--device", default=None, help="Override device (cuda|xpu|mps|cpu)")
    p.add_argument("--precision", default="fp32", help="fp32|fp16|bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50, help="Timed iterations")
    p.add_argument("--sequence-length", type=int, default=128, help="Sequence length for NLP models")
    p.add_argument("--export-tsv", default=None, help="Append/Write TSV results file")
    p.add_argument("--json", default=None, help="Write JSON result (single batch size only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-data", action="store_true", help="Use real dataset if available (MNIST for LeNet)")
    p.add_argument("--disable-grad", action="store_true", help="Force torch.no_grad (default yes)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if available (PyTorch 2.x)")
    return p.parse_args()


def infer_default_inputs(model_name: str, batch: int, seq_len: int, device: str):
    if model_name == "lenet":
        return torch.randn(batch, 1, 28, 28, device=device)
    if model_name == "resnet50":
        return torch.randn(batch, 3, 224, 224, device=device)
    if model_name in ("bert-base-uncased", "gpt2"):
        vocab_size = 30522 if "bert" in model_name else 50257
        return torch.randint(0, vocab_size, (batch, seq_len), device=device)
    raise ValueError(f"No input rule for {model_name}")


def run_inference(model, model_name: str, input_tensor: torch.Tensor):
    if model_name in ("bert-base-uncased", "gpt2"):
        if model_name.startswith("bert"):
            outputs = model(input_tensor, attention_mask=torch.ones_like(input_tensor))
        else:
            outputs = model(input_tensor)
        return outputs
    return model(input_tensor)


def benchmark(args) -> List[Dict[str, Any]]:
    torch.manual_seed(args.seed)
    device = args.device or auto_device()
    results = []
    base_model = get_model(args.model)
    base_model.to(device)
    if args.compile and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(base_model)  # type: ignore
        except Exception as e:  # pragma: no cover
            print(f"[warn] torch.compile failed: {e}")

    param_count = model_param_count(base_model)
    dev_props = device_properties(device)

    for bsz in args.batch_size:
        inp = infer_default_inputs(args.model, bsz, args.sequence_length, device)

        # Warmup
        with torch.no_grad():
            for _ in range(args.warmup):
                with autocast_context(device, args.precision):
                    _ = run_inference(base_model, args.model, inp)
        if device == "cuda":
            torch.cuda.synchronize()

        timings = []
        with memory_monitor(device):
            with torch.no_grad():
                for _ in range(args.iters):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    with Timer() as t:
                        with autocast_context(device, args.precision):
                            _ = run_inference(base_model, args.model, inp)
                        if device == "cuda":
                            torch.cuda.synchronize()
                    timings.append(t.dt)
        mem_stats = getattr(memory_monitor, 'result', {})  # type: ignore
        stats = TimingStats(timings)
        throughput = bsz / (stats.mean)  # samples per second

        record = {
            "model": args.model,
            "batch_size": bsz,
            "precision": args.precision,
            "device": device,
            **stats.to_dict(),
            "throughput_sps_mean": throughput,
            "params": param_count,
            **{f"dev_{k}": v for k, v in dev_props.items() if k != 'device'},
            **mem_stats,
        }
        results.append(record)
        print_summary(record)
    return results


def print_summary(rec: Dict[str, Any]):
    from tabulate import tabulate
    display = {k: v for k, v in rec.items() if k not in {"host_mem_delta", "device_peak_mem"}}
    rows = [(k, f"{v}") for k, v in display.items()]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))
    if rec.get("device_peak_mem") is not None:
        print(f"Device peak memory bytes: {rec['device_peak_mem']}")
    print(f"Host RSS delta bytes: {rec.get('host_mem_delta')}")
    print("-" * 60)


def append_tsv(path: str, records: List[Dict[str, Any]]):
    import csv
    fieldnames = sorted({k for r in records for k in r.keys()})
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        if not exists:
            w.writeheader()
        for r in records:
            w.writerow(r)


def write_json(path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def main():
    args = parse_args()
    recs = benchmark(args)
    if args.export_tsv:
        append_tsv(args.export_tsv, recs)
    if args.json:
        if len(recs) != 1:
            print("[warn] JSON export with multiple batch sizes will use the first record")
        write_json(args.json, recs[0])


if __name__ == "__main__":
    main()
