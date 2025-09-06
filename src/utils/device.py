import torch
from contextlib import contextmanager

def auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def device_properties(device: str) -> dict:
    props = {'device': device}
    if device == 'cuda':
        idx = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        props.update({
            'name': p.name,
            'total_memory': p.total_memory,
            'multi_processor_count': p.multi_processor_count,
            'capability': f'{p.major}.{p.minor}'
        })
    return props

def normalize_precision(prec: str) -> str:
    prec = prec.lower()
    if prec not in {'fp32','fp16','bf16'}:
        raise ValueError('precision must be one of fp32|fp16|bf16')
    return prec

@contextmanager
def autocast_context(device: str, precision: str):
    precision = normalize_precision(precision)
    if precision == 'fp32':
        yield
        return
    use_bf16 = precision == 'bf16'
    if device in ('cuda','xpu','mps'):
        with torch.amp.autocast(device_type='cuda' if device=='cuda' else device, dtype=torch.bfloat16 if use_bf16 else torch.float16):
            yield
    else:
        yield

__all__ = ['auto_device','device_properties','autocast_context','normalize_precision']
