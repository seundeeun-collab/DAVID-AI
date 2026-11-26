"""Download and tokenize the English (Stage 1) and chat (Stage 2) training data.

    python prepare_data.py --english 600000      # TinyStories, simple fluent English
    python prepare_data.py --chat 200000         # smol-smoltalk conversations
    python prepare_data.py --code                # token cache from files/data/*.py

Everything is streamed, tokenized one item at a time and appended to uint16 files in
files/tokens/, so RAM stays small. Chat data also gets a uint8 .mask file marking the
assistant tokens that are trained on. Re-running with the same settings reuses the cache.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from chat_format import encode_conversation, eot_id

ROOT = Path(__file__).resolve().parent
TOKENS = ROOT / "files" / "tokens"
TINYSTORIES = "hf://datasets/roneneldan/TinyStories/data/{split}-*.parquet"      # CDLA-Sharing-1.0
SMOLTALK = "hf://datasets/HuggingFaceTB/smol-smoltalk/data/{split}-*.parquet"     # Apache-2.0


class TokenWriter:
    """Append token ids (and optionally a train mask) to files/tokens/<name>.bin/.mask."""
    def __init__(self, name, masked=False):
        TOKENS.mkdir(parents=True, exist_ok=True)
        self.bin, self.mask = TOKENS / f"{name}.bin", (TOKENS / f"{name}.mask" if masked else None)
        self.meta = TOKENS / f"{name}.json"
        self.f = self.bin.with_suffix(".bin.tmp").open("wb")
        self.m = self.mask.with_suffix(".mask.tmp").open("wb") if masked else None
        self.tokens = self.items = 0
    def write(self, ids, mask=None):
        np.asarray(ids, dtype=np.uint16).tofile(self.f)
        if self.m is not None: np.asarray(mask, dtype=np.uint8).tofile(self.m)
        self.tokens += len(ids); self.items += 1
    def close(self, settings):
        self.f.close(); self.bin.with_suffix(".bin.tmp").replace(self.bin)
        if self.m is not None: self.m.close(); self.mask.with_suffix(".mask.tmp").replace(self.mask)
        self.meta.write_text(json.dumps({"settings": settings, "tokens": self.tokens, "items": self.items}))
        print(f"  wrote {self.bin.name}: {self.items:,} items, {self.tokens:,} tokens", flush=True)


def up_to_date(name, settings, masked=False):
    meta = TOKENS / f"{name}.json"
    if not (meta.exists() and (TOKENS / f"{name}.bin").exists()): return False
    if masked and not (TOKENS / f"{name}.mask").exists(): return False
    return json.loads(meta.read_text()).get("settings") == settings


def stream(pattern, split):
    from datasets import load_dataset  # Imported here so --code works without `datasets`.
    return load_dataset("parquet", data_files={"train": pattern.format(split=split)}, split="train", streaming=True)


def prepare_english(tok, count, rebuild):
    eot = eot_id(tok)
    for name, split, n in (("english_train", "train", count), ("english_val", "validation", max(500, min(5000, count // 100)))):
        settings = {"source": "TinyStories", "split": split, "count": n}
        if not rebuild and up_to_date(name, settings): print(f"  {name} is up to date"); continue
        print(f"Tokenizing {n:,} TinyStories ({split})...", flush=True)
        w = TokenWriter(name)
        for row in stream(TINYSTORIES, split):
            text = (row.get("text") or "").replace("<|endoftext|>", "").strip()
            if len(text) < 20: continue
            w.write(tok.encode(text).ids + [eot])
            if w.items % 20000 == 0: print(f"  {w.items:,}/{n:,} stories ({w.tokens:,} tokens)", flush=True)
            if w.items >= n: break
        w.close(settings)


def prepare_chat(tok, count, max_tokens, rebuild):
    eot = eot_id(tok)
    for name, split, n in (("chat_train", "train", count), ("chat_val", "test", max(300, min(3000, count // 50)))):
        settings = {"source": "smol-smoltalk", "split": split, "count": n, "max_tokens": max_tokens}
        if not rebuild and up_to_date(name, settings, masked=True): print(f"  {name} is up to date"); continue
        print(f"Tokenizing {n:,} smol-smoltalk conversations ({split})...", flush=True)
        w = TokenWriter(name, masked=True); skipped = 0
        for row in stream(SMOLTALK, split):
            messages = row.get("messages") or []
            if not any(m.get("role") == "assistant" for m in messages): continue
            ids, mask = encode_conversation(tok, messages, eot)
            # Skip conversations longer than the model's window so every one is seen whole.
            if len(ids) > max_tokens: skipped += 1; continue
            w.write(ids, mask)
            if w.items % 10000 == 0: print(f"  {w.items:,}/{n:,} conversations ({w.tokens:,} tokens)", flush=True)
            if w.items >= n: break
        if skipped: print(f"  skipped {skipped:,} conversations longer than {max_tokens} tokens")
        w.close(settings)


def prepare_code(tok, rebuild):
    from train import cached_tokens  # Same cache train.py --stage code uses.
    files = sorted((ROOT / "files" / "data").glob("*.py"))
    if len(files) < 2: raise RuntimeError("No Python files in files/data; run python download_data.py --count 10000")
    random.Random(42).shuffle(files); split = max(1, int(.9 * len(files)))
    cached_tokens("train", files[:split], tok, rebuild); cached_tokens("val", files[split:], tok, rebuild)
    print("  Python code cache is ready (files/tokens/train.bin, val.bin)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--english", type=int, default=0, help="Number of TinyStories to tokenize (0 = skip).")
    p.add_argument("--chat", type=int, default=0, help="Number of smol-smoltalk conversations (0 = skip).")
    p.add_argument("--chat-max-tokens", type=int, default=1024, help="Skip conversations longer than this (train.py --stage chat uses a 1,024 window).")
    p.add_argument("--code", action="store_true", help="Build the Python token cache from files/data.")
    p.add_argument("--rebuild", action="store_true", help="Re-download and re-tokenize even if a matching cache exists.")
    a = p.parse_args()
    if not (a.english or a.chat or a.code): p.error("choose at least one of --english N, --chat N, --code")
    tok = Tokenizer.from_file(str(ROOT / "files" / "tokenizer.json"))
    if a.code: prepare_code(tok, a.rebuild)
    if a.english: prepare_english(tok, a.english, a.rebuild)
    if a.chat: prepare_chat(tok, a.chat, a.chat_max_tokens, a.rebuild)
    print("Done. Next: python train.py --stage english   (then --stage chat)")


if __name__ == "__main__": main()
