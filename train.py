"""Train the decoder on locally stored Python files (no hosted inference)."""
import argparse, json, math, random, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from tokenizers import Tokenizer
from model import DecoderOnlyTransformer, ModelConfig

ROOT = Path(__file__).resolve().parent

class CodeDataset(Dataset):
    def __init__(self, files, tokenizer, context):
        ids = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip(): ids.extend(tokenizer.encode(text).ids + [tokenizer.token_to_id("<|endoftext|>") or 50256])
            except (OSError, ValueError): pass
        self.tokens = torch.tensor(ids, dtype=torch.long); self.context = context
    def __len__(self): return max(0, (len(self.tokens) - 1) // self.context)
    def __getitem__(self, i):
        start = i * self.context
        return self.tokens[start:start+self.context+1]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--context",type=int,default=512); p.add_argument("--accumulation",type=int,default=4); p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--max-steps",type=int,default=0); p.add_argument("--dim",type=int,default=512); p.add_argument("--layers",type=int,default=6); p.add_argument("--file-limit",type=int,default=0,help="Optional small subset for a quick training smoke test."); a=p.parse_args()
    data=ROOT/"files"/"data"; tokpath=ROOT/"files"/"tokenizer.json"
    files=sorted(data.glob("*.py"));
    if a.file_limit: files=files[:a.file_limit]
    if not tokpath.exists(): raise FileNotFoundError("Missing files/tokenizer.json; run python download_assets.py")
    if len(files)<2: raise RuntimeError("Need at least two .py files under files/data; run python download_data.py --count 10000")
    tokenizer=Tokenizer.from_file(str(tokpath)); random.Random(42).shuffle(files); split=max(1,int(.9*len(files)))
    train_ds=CodeDataset(files[:split],tokenizer,a.context); val_ds=CodeDataset(files[split:],tokenizer,a.context)
    if not len(train_ds): raise RuntimeError("Training files did not produce enough tokens for one context window")
    train_dl=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,drop_last=False); val_dl=DataLoader(val_ds,batch_size=a.batch_size)
    config=ModelConfig(vocab_size=tokenizer.get_vocab_size(),dim=a.dim,n_layers=a.layers,max_seq_len=max(1024,a.context))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=DecoderOnlyTransformer(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    total=max(1,a.max_steps or math.ceil(len(train_dl)/a.accumulation)*a.epochs); warmup=max(1,int(total*.05))
    def lr_at(step): return min((step+1)/warmup, .5*(1+math.cos(math.pi*min(step,total)/total)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_at)
    amp=device.type=="cuda"; scaler=torch.amp.GradScaler("cuda",enabled=amp); log=ROOT/"training.jsonl"; step=0; best=float("inf"); stale=0
    # Clone tied embedding/head tensors because safetensors does not serialize aliases.
    def save(path): save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},str(path))
    with log.open("a",encoding="utf-8") as lf:
        for epoch in range(a.epochs):
            model.train(); optimizer.zero_grad(set_to_none=True); running=0.0
            for bi,batch in enumerate(train_dl):
                batch=batch.to(device); x,y=batch[:,:-1],batch[:,1:]
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): _,loss=model(x,y)
                scaler.scale(loss/a.accumulation).backward(); running+=loss.item()
                if (bi+1)%a.accumulation==0 or bi+1==len(train_dl):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step+=1
                    if step%10==0: print(f"epoch={epoch+1} step={step} loss={running/10:.4f} lr={scheduler.get_last_lr()[0]:.2e}",flush=True); lf.write(json.dumps({"epoch":epoch+1,"step":step,"loss":running/10,"lr":scheduler.get_last_lr()[0]})+"\n"); lf.flush(); running=0
                    if step%500==0: save(ROOT/f"checkpoint-{step}.safetensors")
                    if a.max_steps and step>=a.max_steps: break
            model.eval(); losses=[]
            with torch.no_grad():
                for j,batch in enumerate(val_dl):
                    if j>=100: break
                    batch=batch.to(device); x=batch[:,:-1]; y=batch[:,1:]; _,vl=model(x,y)
                    if vl is not None: losses.append(vl.item())
            val=sum(losses)/len(losses) if losses else float("nan"); ppl=math.exp(min(val,20)) if math.isfinite(val) else float("nan")
            print(f"evaluation epoch={epoch+1} loss={val:.4f} perplexity={ppl:.2f}",flush=True); lf.write(json.dumps({"epoch":epoch+1,"validation_loss":val,"perplexity":ppl})+"\n"); lf.flush()
            if val<best: best=val; stale=0; save(ROOT/"model.safetensors")
            else: stale+=1
            if stale>=2: print("Early stopping: validation loss did not improve for two epochs."); break
            if a.max_steps and step>=a.max_steps: break
    (ROOT/"model_config.json").write_text(json.dumps(config.to_dict(),indent=2),encoding="utf-8")
    print(f"Saved model.safetensors on {device}. Training log: {log}")
if __name__=="__main__": main()
