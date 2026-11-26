"""Small smoke test for architecture/tokenizer/generation wiring."""
from pathlib import Path
import json, tempfile
import torch
from tokenizers import Tokenizer
from safetensors.torch import save_file
from model import DecoderOnlyTransformer, ModelConfig
from generate import generate
from chat import chat
from chat_format import encode_conversation, build_prompt, eot_id
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
        reply=chat([{"role":"user","content":"Hi!"}],max_new_tokens=8,checkpoint_path=weights,config_path=cfg)
        assert isinstance(reply,str) and "<|user|>" not in reply
        print("PASS: chat mode builds a prompt and returns a reply.")
    # Chat training data: only the assistant's reply (and its end token) is trained on.
    msgs=[{"role":"system","content":"Be kind."},{"role":"user","content":"What is a noun?"},{"role":"assistant","content":"A naming word."}]
    ids,mask=encode_conversation(tokenizer,msgs)
    trained=tokenizer.decode([t for t,m in zip(ids,mask) if m])
    assert len(ids)==len(mask) and trained.strip()=="A naming word." and ids[mask.index(1)+sum(mask)-1]==eot_id(tokenizer)
    prompt=tokenizer.decode(build_prompt(tokenizer,msgs[1:],64))
    assert prompt.endswith("<|assistant|>\n") and "What is a noun?" in prompt
    print("PASS: chat format trains only on assistant replies and prompts end ready for a reply.")
    # Masked targets (-100) are ignored; a window with none left gives loss 0, not NaN.
    x=torch.tensor([[1,2,3,4]]); y=torch.full((1,4),-100); _,l=model(x,y)
    assert l.item()==0.0
    y[0,2]=5; _,l=model(x,y); assert torch.isfinite(l) and l.item()>0
    print("PASS: masked loss ignores user/system tokens.")
    if not (ROOT/"model.safetensors").exists(): print("NOTE: train.py must create the real model.safetensors before the website can generate useful code.")
if __name__=="__main__":main()
