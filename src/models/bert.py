from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Tuple

_DEF_MODEL = 'bert-base-uncased'

def load_bert(model_name: str=_DEF_MODEL):
    """Load BERT masked language model + tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tok
