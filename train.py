"""Train the decoder on locally stored Python files (no hosted inference)."""
import argparse, hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from tokenizers import Tokenizer
from model import DecoderOnlyTransformer, ModelConfig

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "files" / "tokens"


def build_token_cache(files, tokenizer, out_path):
    """Tokenize one file at a time and append uint16 ids to a .bin file on disk.

    Only one file's tokens are in RAM at once, so this scales from 10,000 to
    1,000,000 files on a small machine. GPT-2 ids (< 65,536) fit in uint16.
    """
    eot = tokenizer.token_to_id("<|endoftext|>")
    eot = 50256 if eot is None else eot
    total, tmp = 0, out_path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        for i, path in enumerate(files, 1):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if text.strip():
                ids = tokenizer.encode(text).ids
                ids.append(eot)
                np.asarray(ids, dtype=np.uint16).tofile(f)
                total += len(ids)
            if i % 1000 == 0:
                print(f"tokenized {i}/{len(files)} files ({total:,} tokens)", flush=True)
    tmp.replace(out_path)
    return total


def cached_tokens(name, files, tokenizer, rebuild):
    """Reuse files/tokens/<name>.bin when the same file list built it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    bin_path, meta_path = CACHE / f"{name}.bin", CACHE / f"{name}.json"
    digest = hashlib.sha256("\n".join(p.name for p in files).encode()).hexdigest()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if rebuild or not bin_path.exists() or meta.get("files") != digest:
        print(f"Building {name} token cache from {len(files)} files...", flush=True)
        count = build_token_cache(files, tokenizer, bin_path)
        meta_path.write_text(json.dumps({"files": digest, "tokens": count}))
    return bin_path


class TokenDataset(Dataset):
    """Context windows read straight from the memory-mapped token file."""
    def __init__(self, bin_path, context):
        size = bin_path.stat().st_size
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r") if size else np.zeros(0, dtype=np.uint16)
        self.context = context
    def __len__(self): return max(0, (len(self.tokens) - 1) // self.context)
    def __getitem__(self, i):
        start = i * self.context
        return torch.from_numpy(self.tokens[start:start + self.context + 1].astype(np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context", type=int, default=512); p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--dim", type=int, default=512); p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8); p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--file-limit", type=int, default=0, help="Optional small subset for a quick training smoke test.")
    p.add_argument("--rebuild-cache", action="store_true", help="Re-tokenize files even if a matching cache exists.")
    a = p.parse_args()

    data = ROOT / "files" / "data"; tokpath = ROOT / "files" / "tokenizer.json"
    files = sorted(data.glob("*.py"))
    if a.file_limit: files = files[:a.file_limit]
    if not tokpath.exists(): raise FileNotFoundError("Missing files/tokenizer.json; run python download_assets.py")
    if len(files) < 2: raise RuntimeError("Need at least two .py files under files/data; run python download_data.py --count 10000")
    tokenizer = Tokenizer.from_file(str(tokpath))
    if tokenizer.get_vocab_size() > 65535: raise RuntimeError("Token cache uses uint16; this tokenizer is too large for it.")
    random.Random(42).shuffle(files); split = max(1, int(.9 * len(files)))
    train_ds = TokenDataset(cached_tokens("train", files[:split], tokenizer, a.rebuild_cache), a.context)
    val_ds = TokenDataset(cached_tokens("val", files[split:], tokenizer, a.rebuild_cache), a.context)
    if not len(train_ds): raise RuntimeError("Training files did not produce enough tokens for one context window")
    if not len(val_ds): print("WARNING: validation set is smaller than one context window; validation will be skipped.")
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size)

    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), dim=a.dim, n_heads=a.heads, n_layers=a.layers, max_seq_len=max(1024, a.context))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}. "
          f"Train windows: {len(train_ds):,}, validation windows: {len(val_ds):,}", flush=True)
    # Save the config up front so generate.py can load any checkpoint written below.
    (ROOT / "model_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    n_batches = len(train_dl)
    total = max(1, a.max_steps or math.ceil(n_batches / a.accumulation) * a.epochs); warmup = max(1, int(total * .05))
    def lr_at(step): return min((step + 1) / warmup, .5 * (1 + math.cos(math.pi * min(step, total) / total)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log = ROOT / "training.jsonl"; step = 0; best = float("inf"); stale = 0; saved = False
    # Clone tied embedding/head tensors because safetensors does not serialize aliases.
    def save(path): save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(path))

    with log.open("a", encoding="utf-8") as lf:
        for epoch in range(a.epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            running_loss, running_n = 0.0, 0
            for bi, batch in enumerate(train_dl):
                batch = batch.to(device); x, y = batch[:, :-1], batch[:, 1:]
                # The last group of an epoch may hold fewer than `accumulation` batches; scale by its real size.
                group_start = (bi // a.accumulation) * a.accumulation
                group_size = min(a.accumulation, n_batches - group_start)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp): _, loss = model(x, y)
                scaler.scale(loss / group_size).backward()
                running_loss += loss.item(); running_n += 1
                if bi + 1 == group_start + group_size:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step += 1
                    if step % a.log_every == 0:
                        # Average over every micro-batch seen since the last log line (fixes over-reported loss).
                        avg = running_loss / running_n
                        print(f"epoch={epoch+1} step={step}/{total} loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                        lf.write(json.dumps({"epoch": epoch + 1, "step": step, "loss": avg, "lr": scheduler.get_last_lr()[0]}) + "\n"); lf.flush()
                        running_loss, running_n = 0.0, 0
                    if step % 500 == 0: save(ROOT / f"checkpoint-{step}.safetensors")
                    if a.max_steps and step >= a.max_steps: break

            model.eval(); losses = []
            with torch.no_grad():
                for j, batch in enumerate(val_dl):
                    if j >= 100: break
                    batch = batch.to(device); x = batch[:, :-1]; y = batch[:, 1:]; _, vl = model(x, y)
                    losses.append(vl.item())
            val = sum(losses) / len(losses) if losses else float("nan")
            ppl = math.exp(min(val, 20)) if math.isfinite(val) else float("nan")
            print(f"evaluation epoch={epoch+1} loss={val:.4f} perplexity={ppl:.2f}", flush=True)
            lf.write(json.dumps({"epoch": epoch + 1, "validation_loss": val, "perplexity": ppl}) + "\n"); lf.flush()
            if not math.isfinite(val):
                save(ROOT / "model.safetensors"); saved = True  # No validation data: keep the latest weights.
            elif val < best:
                best = val; stale = 0; save(ROOT / "model.safetensors"); saved = True
            else:
                stale += 1
            if stale >= 2: print("Early stopping: validation loss did not improve for two epochs."); break
            if a.max_steps and step >= a.max_steps: break
    if not saved: save(ROOT / "model.safetensors")
    print(f"Saved model.safetensors on {device}. Training log: {log}")


if __name__ == "__main__": main()
