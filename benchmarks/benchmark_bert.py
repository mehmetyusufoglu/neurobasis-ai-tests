import torch, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from models import load_bert  # type: ignore

def benchmark_bert_inference(device='cpu', batch_size=8):
    model, tokenizer = load_bert()
    model = model.to(device).eval()
    sentences = ["This is a test sentence."] * batch_size
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()
    print(f"BERT Inference: {batch_size} samples, Total time: {end-start:.4f}s, Avg/sample: {(end-start)/batch_size:.6f}s")

if __name__ == "__main__":
    benchmark_bert_inference(device='cpu', batch_size=8)
