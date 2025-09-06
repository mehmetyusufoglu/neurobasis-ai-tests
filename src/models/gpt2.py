from transformers import AutoModelForCausalLM, AutoTokenizer

_DEF_MODEL = 'gpt2'

def load_gpt2(model_name: str=_DEF_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tok))
    return model, tok
