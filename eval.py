"""Test a checkpoint the same way every time, so you can compare training runs honestly.

    python eval.py                                  # tests model.safetensors
    python eval.py --checkpoint model-english.safetensors

It measures validation loss on every token cache it finds (lower is better) and asks a
fixed set of English, chat and coding questions. Results go to eval_results.json.
"""
import argparse, json, math, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from chat import chat
from generate import _load, generate
from train import CACHE, TokenDataset

ROOT = Path(__file__).resolve().parent
STORY_STARTS = ["Once upon a time, there was a little girl named", "Tom was sad because his dog"]
CHAT_QUESTIONS = ["Hi! Who are you?", "What is a noun? Give one example.",
                  "Can you rewrite this sentence correctly: he go to school yesterday.",
                  "Give me three tips to learn English faster.", "What is Python used for?"]
CODE_PROMPTS = ["# Write a function that adds two numbers\n", "def is_even(n):"]
CACHES = [("english", "english_val", False), ("chat", "chat_val", True), ("code", "val", False)]


@torch.no_grad()
def val_loss(model, device, name, masked, context, batches=50):
    bin_path = CACHE / f"{name}.bin"
    if not bin_path.exists(): return None
    ds = TokenDataset(bin_path, context, CACHE / f"{name}.mask" if masked else None)
    if not len(ds): return None
    losses = []
    for i, (x, y) in enumerate(DataLoader(ds, batch_size=8)):
        if i >= batches: break
        losses.append(model(x.to(device), y.to(device))[1].item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(ROOT / "model.safetensors")); p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--context", type=int, default=512); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ckpt = Path(a.checkpoint); torch.manual_seed(a.seed)
    tok, cfg, device, model = _load(ckpt)
    results = {"checkpoint": ckpt.name, "time": time.strftime("%Y-%m-%d %H:%M"), "validation": {}, "answers": []}

    print(f"== Validation loss for {ckpt.name} (lower is better)")
    for label, name, masked in CACHES:
        loss = val_loss(model, device, name, masked, min(a.context, cfg.max_seq_len))
        if loss is None: continue
        results["validation"][label] = {"loss": loss, "perplexity": math.exp(min(loss, 20))}
        print(f"  {label:8s} loss={loss:.4f} perplexity={math.exp(min(loss, 20)):.1f}")

    def ask(kind, prompt, answer):
        results["answers"].append({"kind": kind, "prompt": prompt, "answer": answer})
        print(f"\n[{kind}] {prompt.strip()}\n{answer.strip()}")
    print("\n== Fixed questions (same every run)")
    for s in STORY_STARTS: ask("english", s, generate(s, a.tokens, checkpoint_path=ckpt))
    for q in CHAT_QUESTIONS: ask("chat", q, chat([{"role": "user", "content": q}], a.tokens, checkpoint_path=ckpt))
    for c in CODE_PROMPTS: ask("code", c, generate(c, a.tokens, checkpoint_path=ckpt))
    out = ROOT / "eval_results.json"
    history = json.loads(out.read_text()) if out.exists() else []
    history.append(results); out.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nSaved to {out.name} (every run is kept, newest last).")


if __name__ == "__main__": main()
