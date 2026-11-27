"""Build nanoGPT training data from real open-source code in 11 languages.

Streams codeparrot/github-code-clean (never downloads the whole dataset), keeps
files for the languages below, tags each file with its language, and writes
train.bin / val.bin (GPT-2 tokens) + meta.json for nanoGPT's train.py.

Usage:  python data/code/prepare.py --per_lang 20000 --max_minutes 60
"""
import argparse, json, os, time
import numpy as np
import tiktoken
from datasets import load_dataset

# dataset name -> tag used in prompts (keep in sync with the website)
LANGS = {"Python": "python", "JavaScript": "javascript", "TypeScript": "typescript",
         "HTML": "html", "CSS": "css", "Java": "java", "C++": "cpp", "GO": "go",
         "Rust": "rust", "PHP": "php", "SQL": "sql"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_lang", type=int, default=20000, help="max files per language")
    p.add_argument("--max_minutes", type=float, default=60, help="stop streaming after this long")
    p.add_argument("--max_chars", type=int, default=20000, help="skip files longer than this")
    p.add_argument("--val_fraction", type=float, default=0.01)
    a = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ds = load_dataset("parquet", split="train", streaming=True,
                      data_files={"train": "hf://datasets/codeparrot/github-code-clean/data/*.parquet"})
    counts = {t: 0 for t in LANGS.values()}
    train_f = open(os.path.join(out_dir, "train.bin"), "wb")
    val_f = open(os.path.join(out_dir, "val.bin"), "wb")
    n_train = n_val = seen = 0
    start = time.time()
    for row in ds:
        seen += 1
        tag = LANGS.get(row.get("language"))
        if tag and counts[tag] < a.per_lang:
            code = (row.get("code") or "").replace("\x00", "")
            if 40 <= len(code) <= a.max_chars:
                text = f"<|lang:{tag}|>\n{code}"
                ids = enc.encode_ordinary(text) + [eot]
                arr = np.array(ids, dtype=np.uint16)
                # every 100th file goes to validation
                if sum(counts.values()) % int(1 / a.val_fraction) == 0:
                    arr.tofile(val_f); n_val += len(arr)
                else:
                    arr.tofile(train_f); n_train += len(arr)
                counts[tag] += 1
        if seen % 20000 == 0:
            print(f"scanned {seen:,} files | kept {sum(counts.values()):,} | tokens {n_train+n_val:,}", flush=True)
        if all(c >= a.per_lang for c in counts.values()) or time.time() - start > a.max_minutes * 60:
            break
    train_f.close(); val_f.close()
    if n_val == 0 or n_train == 0:
        raise SystemExit("Not enough data collected. Increase --max_minutes.")
    meta = {"vocab_size": 50304, "tokenizer": "gpt2", "languages": counts,
            "train_tokens": n_train, "val_tokens": n_val, "prompt_format": "<|lang:TAG|>\\n<comment with task>\\n"}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print("files per language:", counts)
    print(f"train tokens: {n_train:,}  val tokens: {n_val:,}")

if __name__ == "__main__":
    main()
