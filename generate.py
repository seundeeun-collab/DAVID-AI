"""Sample code from a locally trained model."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from model import DecoderOnlyTransformer, ModelConfig

ROOT=Path(__file__).resolve().parent
@torch.no_grad()
def generate(prompt="def add(a, b):", max_new_tokens=200, temperature=.7, top_p=.9, top_k=50, checkpoint_path=None, config_path=None):
    tok=Tokenizer.from_file(str(ROOT/"files"/"tokenizer.json")); cfgpath=Path(config_path) if config_path else ROOT/"model_config.json"
    cfg=ModelConfig(**json.loads(cfgpath.read_text())) if cfgpath.exists() else ModelConfig(vocab_size=tok.get_vocab_size())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=DecoderOnlyTransformer(cfg).to(device)
    weights=Path(checkpoint_path) if checkpoint_path else ROOT/"model.safetensors"
    if not weights.exists(): raise FileNotFoundError("No model.safetensors yet. Train with python train.py first.")
    model.load_state_dict(load_file(str(weights),device=str(device))); model.eval(); ids=torch.tensor([tok.encode(prompt).ids],device=device)
    for _ in range(max_new_tokens):
        logits,_=model(ids[:,-cfg.max_seq_len:]); scores=logits[:,-1,:]/max(temperature,1e-5)
        if top_k: cutoff=torch.topk(scores,min(top_k,scores.size(-1))).values[:,-1,None]; scores=scores.masked_fill(scores<cutoff,-float("inf"))
        probs=torch.softmax(scores,dim=-1); sorted_p,idx=torch.sort(probs,descending=True); remove=sorted_p.cumsum(-1)-sorted_p>top_p; sorted_p[remove]=0; probs=torch.zeros_like(probs).scatter(1,idx,sorted_p); probs=probs/probs.sum(-1,keepdim=True)
        nxt=torch.multinomial(probs,1); ids=torch.cat((ids,nxt),dim=1)
    return tok.decode(ids[0].tolist(),skip_special_tokens=True)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("prompt",nargs="?",default="def add(a, b):"); p.add_argument("--tokens",type=int,default=200); a=p.parse_args(); code=generate(a.prompt,a.tokens); print(code); (ROOT/"output.py").write_text(code,encoding="utf-8")
