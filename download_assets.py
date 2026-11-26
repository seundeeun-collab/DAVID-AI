"""Fetch the five requested GPT-2 tokenizer/model metadata files from HF."""
from pathlib import Path
import requests

OUT=Path(__file__).resolve().parent/"files"
REPO="https://huggingface.co/gpt2/resolve/main/"
FILES=("tokenizer.json","vocab.json","merges.txt","config.json","generation_config.json")
for name in FILES:
    response=requests.get(REPO+name,timeout=(15,120)); response.raise_for_status()
    target=OUT/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(response.content)
    if target.stat().st_size < 10: raise RuntimeError(f"Downloaded file looks empty: {target}")
    print(f"Downloaded {target.name}: {target.stat().st_size:,} bytes")
print("Verified files:", ", ".join(p.name for p in OUT.iterdir() if p.is_file()))
