"""Train DAVID-AI in stages on local token caches (no hosted inference).

    python train.py                      # Stage 0: Python code (files/data/*.py), as before
    python train.py --stage english      # Stage 1: TinyStories English (+ some Python)
    python train.py --stage chat         # Stage 2: smol-smoltalk conversations

Stages 1 and 2 continue from model.safetensors when it exists, so each stage builds on
the last. Run python prepare_data.py first to download and tokenize their data.
"""
import argparse, hashlib, json, math, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from model import DecoderOnlyTransformer, ModelConfig

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "files" / "tokens"
# Per-stage defaults; any flag given on the command line wins.
STAGE_DEFAULTS = {
    "code":    {"lr": 5e-4, "epochs": 3, "context": 512, "batch_size": 16},
    "english": {"lr": 5e-4, "epochs": 1, "context": 512, "batch_size": 16},
    # Chat uses the full 1,024-token window so most conversations fit whole.
    "chat":    {"lr": 1e-4, "epochs": 2, "context": 1024, "batch_size": 8},
}


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
    """Context windows read straight from a memory-mapped token file.

    Returns (inputs, targets). With a .mask file, targets the mask marks 0 (the
    user's and system's lines in chat data) become -100 and are not trained on.
    """
    def __init__(self, bin_path, context, mask_path=None):
        size = bin_path.stat().st_size
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r") if size else np.zeros(0, dtype=np.uint16)
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r") if mask_path and size else None
        if self.mask is not None and len(self.mask) != len(self.tokens):
            raise RuntimeError(f"{mask_path.name} does not match {bin_path.name}; re-run prepare_data.py --rebuild")
        self.context = context
    def __len__(self): return max(0, (len(self.tokens) - 1) // self.context)
    def __getitem__(self, i):
        s, e = i * self.context, i * self.context + self.context + 1
        chunk = torch.from_numpy(self.tokens[s:e].astype(np.int64))
        x, y = chunk[:-1], chunk[1:].clone()
        if self.mask is not None:
            y[torch.from_numpy(self.mask[s + 1:e].astype(np.bool_)).logical_not()] = -100
        return x, y


def mix(primary, extra, share, seed=0):
    """Add `extra` windows so they make up about `share` of the result (keeps old skills)."""
    if not extra or share <= 0 or not len(extra): return primary
    n = min(len(extra), int(len(primary) * share / (1 - share)))
    idx = random.Random(seed).sample(range(len(extra)), n)
    print(f"Mixing in {n:,} extra windows ({share:.0%}) to keep earlier skills.", flush=True)
    return ConcatDataset([primary, Subset(extra, idx)])


def need(path, hint):
    if not path.exists(): raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}. {hint}")
    return path


def load_datasets(a, tokenizer):
    """Build (train, validation) datasets for the chosen stage."""
    c = a.context
    if a.stage == "code":
        files = sorted((ROOT / "files" / "data").glob("*.py"))
        if a.file_limit: files = files[:a.file_limit]
        if len(files) < 2: raise RuntimeError("Need at least two .py files under files/data; run python download_data.py --count 10000")
        random.Random(42).shuffle(files); split = max(1, int(.9 * len(files)))
        return (TokenDataset(cached_tokens("train", files[:split], tokenizer, a.rebuild_cache), c),
                TokenDataset(cached_tokens("val", files[split:], tokenizer, a.rebuild_cache), c))
    code = TokenDataset(CACHE / "train.bin", c) if (CACHE / "train.bin").exists() else None
    if a.stage == "english":
        hint = "Run python prepare_data.py --english 600000"
        train = TokenDataset(need(CACHE / "english_train.bin", hint), c)
        val = TokenDataset(need(CACHE / "english_val.bin", hint), c)
        if a.mix_code and not code: print("NOTE: no Python cache (files/tokens/train.bin), so no code is mixed in. Run prepare_data.py --code to add it.")
        return mix(train, code, a.mix_code), val
    hint = "Run python prepare_data.py --chat 200000"
    train = TokenDataset(need(CACHE / "chat_train.bin", hint), c, need(CACHE / "chat_train.mask", hint))
    val = TokenDataset(need(CACHE / "chat_val.bin", hint), c, need(CACHE / "chat_val.mask", hint))
    english = TokenDataset(CACHE / "english_train.bin", c) if (CACHE / "english_train.bin").exists() else None
    train = mix(train, english, a.mix_english, seed=1)
    return mix(train, code, a.mix_code, seed=2), val


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGE_DEFAULTS, default="code", help="What to learn: code, english or chat.")
    p.add_argument("--epochs", type=int); p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int); p.add_argument("--context", type=int)
    p.add_argument("--accumulation", type=int, default=4); p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=0, help="Stop cleanly and save after this many hours (Kaggle sessions end at 12).")
    p.add_argument("--dim", type=int, default=512); p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8); p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1000, help="Validate (and keep the best weights) every N steps; 0 = once per epoch.")
    p.add_argument("--save-every", type=int, default=500, help="Overwrite checkpoint-<stage>-latest.safetensors every N steps.")
    p.add_argument("--resume", type=str, default="", help="Weights to continue from. Default for english/chat: model.safetensors if it exists.")
    p.add_argument("--from-scratch", action="store_true", help="Start from random weights even if model.safetensors exists.")
    p.add_argument("--mix-code", type=float, default=0.1, help="english/chat: share of Python windows mixed in (0 to turn off).")
    p.add_argument("--mix-english", type=float, default=0.1, help="chat: share of TinyStories windows mixed in.")
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (keep 0 on Windows).")
    p.add_argument("--file-limit", type=int, default=0, help="code: optional small subset for a quick smoke test.")
    p.add_argument("--rebuild-cache", action="store_true", help="code: re-tokenize files even if a matching cache exists.")
    a = p.parse_args()
    for k, v in STAGE_DEFAULTS[a.stage].items():
        if getattr(a, k) is None: setattr(a, k, v)

    tokpath = ROOT / "files" / "tokenizer.json"
    if not tokpath.exists(): raise FileNotFoundError("Missing files/tokenizer.json; run python download_assets.py")
    tokenizer = Tokenizer.from_file(str(tokpath))
    if tokenizer.get_vocab_size() > 65535: raise RuntimeError("Token cache uses uint16; this tokenizer is too large for it.")

    # Decide where the weights come from before building data, so the context fits the model.
    resume = Path(a.resume) if a.resume else (ROOT / "model.safetensors" if a.stage != "code" else None)
    if a.from_scratch or (resume and not resume.exists() and not a.resume): resume = None
    if resume and not resume.exists(): raise FileNotFoundError(f"--resume file not found: {resume}")
    cfg_path = ROOT / "model_config.json"
    if resume:
        if not cfg_path.exists(): raise FileNotFoundError("Found weights to resume but no model_config.json next to them.")
        config = ModelConfig(**json.loads(cfg_path.read_text()))
        if a.context > config.max_seq_len: raise ValueError(f"--context {a.context} is longer than the model's max_seq_len {config.max_seq_len}")
        print(f"Continuing from {resume.name} (dim={config.dim}, layers={config.n_layers}); --dim/--layers/--heads are ignored.", flush=True)
    else:
        config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), dim=a.dim, n_heads=a.heads, n_layers=a.layers, max_seq_len=max(1024, a.context))
        print("Starting from random weights.", flush=True)

    train_ds, val_ds = load_datasets(a, tokenizer)
    if not len(train_ds): raise RuntimeError("Training data did not produce enough tokens for one context window")
    if not len(val_ds): print("WARNING: validation set is smaller than one context window; validation will be skipped.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, num_workers=a.workers, pin_memory=pin)

    model = DecoderOnlyTransformer(config).to(device)
    if resume: model.load_state_dict(load_file(str(resume), device=str(device)))
    print(f"Stage {a.stage}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}. "
          f"Train windows: {len(train_ds):,}, validation windows: {len(val_ds):,}, lr={a.lr:g}, epochs={a.epochs}", flush=True)
    # Save the config up front so generate.py can load any checkpoint written below.
    cfg_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    n_batches = len(train_dl)
    total = max(1, a.max_steps or math.ceil(n_batches / a.accumulation) * a.epochs); warmup = max(1, int(total * .05))
    def lr_at(step): return min((step + 1) / warmup, .5 * (1 + math.cos(math.pi * min(step, total) / total)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log = ROOT / "training.jsonl"; step = 0; best = float("inf"); stale = 0; saved = False
    started = time.time(); stop = False
    # Clone tied embedding/head tensors because safetensors does not serialize aliases.
    def save(path): save_file({k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}, str(path))

    def evaluate():
        model.eval(); losses = []
        with torch.no_grad():
            for j, (x, y) in enumerate(val_dl):
                if j >= 100: break
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    _, vl = model(x.to(device), y.to(device))
                losses.append(vl.item())
        model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    def keep_if_best(val, lf, epoch):
        nonlocal best, saved
        ppl = math.exp(min(val, 20)) if math.isfinite(val) else float("nan")
        print(f"evaluation stage={a.stage} epoch={epoch} step={step} loss={val:.4f} perplexity={ppl:.2f}", flush=True)
        lf.write(json.dumps({"stage": a.stage, "epoch": epoch, "step": step, "validation_loss": val, "perplexity": ppl}) + "\n"); lf.flush()
        if not math.isfinite(val) or val < best:
            if math.isfinite(val): best = val
            save(ROOT / "model.safetensors"); saved = True  # No validation data: keep the latest weights.
            return True
        return False

    with log.open("a", encoding="utf-8") as lf:
        for epoch in range(1, a.epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            running_loss, running_n, seen_tokens, t0 = 0.0, 0, 0, time.time()
            for bi, (x, y) in enumerate(train_dl):
                x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
                # The last group of an epoch may hold fewer than `accumulation` batches; scale by its real size.
                group_start = (bi // a.accumulation) * a.accumulation
                group_size = min(a.accumulation, n_batches - group_start)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp): _, loss = model(x, y)
                scaler.scale(loss / group_size).backward()
                running_loss += loss.item(); running_n += 1; seen_tokens += x.numel()
                if bi + 1 == group_start + group_size:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step += 1
                    if step % a.log_every == 0:
                        # Average over every micro-batch seen since the last log line.
                        avg, tps = running_loss / running_n, seen_tokens / max(time.time() - t0, 1e-6)
                        print(f"stage={a.stage} epoch={epoch} step={step}/{total} loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} tokens/s={tps:,.0f}", flush=True)
                        lf.write(json.dumps({"stage": a.stage, "epoch": epoch, "step": step, "loss": avg, "lr": scheduler.get_last_lr()[0], "tokens_per_sec": tps}) + "\n"); lf.flush()
                        running_loss, running_n, seen_tokens, t0 = 0.0, 0, 0, time.time()
                    if a.save_every and step % a.save_every == 0: save(ROOT / f"checkpoint-{a.stage}-latest.safetensors")
                    if a.eval_every and step % a.eval_every == 0 and len(val_ds): keep_if_best(evaluate(), lf, epoch)
                    if a.max_steps and step >= a.max_steps: stop = True
                    if a.max_hours and time.time() - started > a.max_hours * 3600:
                        print(f"Reached --max-hours {a.max_hours}; stopping cleanly.", flush=True); stop = True
                    if stop: break

            if keep_if_best(evaluate(), lf, epoch): stale = 0
            else: stale += 1
            if stale >= 2: print("Early stopping: validation loss did not improve for two epochs."); break
            if stop: break
    if not saved: save(ROOT / "model.safetensors")
    shutil.copyfile(ROOT / "model.safetensors", ROOT / f"model-{a.stage}.safetensors")  # A copy per stage, to compare or roll back.
    print(f"Saved model.safetensors (best validation loss {best:.4f}) and model-{a.stage}.safetensors. Training log: {log}")


if __name__ == "__main__": main()
