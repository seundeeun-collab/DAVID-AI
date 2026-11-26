"""Small smoke test for architecture/tokenizer/generation wiring."""
from pathlib import Path
import json, tempfile
import torch
from tokenizers import Tokenizer
from safetensors.torch import save_file
from model import DecoderOnlyTransformer, ModelConfig
from generate import generate
ROOT=Path(__file__).resolve().parent
def main():
    tokenizer=Tokenizer.from_file(str(ROOT/"files"/"tokenizer.json"))
    assert tokenizer.get_vocab_size()>=50257
    config=ModelConfig(vocab_size=tokenizer.get_vocab_size(),dim=64,n_heads=4,n_layers=2,max_seq_len=32,dropout=0)
    model=DecoderOnlyTransformer(config)
    ids=torch.tensor([[1,2,3,4]]); logits,loss=model(ids,ids)
    assert logits.shape==(1,4,tokenizer.get_vocab_size()) and loss is not None and torch.isfinite(loss)
    print("PASS: model initializes, tokenizer loads, forward/loss are finite.")
    # A temporary random checkpoint verifies save/load/sampling wiring. It is
    # intentionally not a substitute for training on the code corpus.
    with tempfile.TemporaryDirectory() as temp:
        weights=Path(temp)/"smoke.safetensors"; cfg=Path(temp)/"config.json"
        save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(weights))
        cfg.write_text(json.dumps(config.to_dict()),encoding="utf-8")
        result=generate("def add(a, b):",max_new_tokens=8,checkpoint_path=weights,config_path=cfg)
        assert result.startswith("def add(a, b):")
        print("PASS: checkpoint loads and generation returns text.")
    if not (ROOT/"model.safetensors").exists(): print("NOTE: train.py must create the real model.safetensors before the website can generate useful code.")
if __name__=="__main__":main()
