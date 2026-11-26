"""Stream public code datasets and save cleaned Python files without full mirrors."""
import argparse, hashlib, re
from pathlib import Path
from datasets import load_dataset

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=10000); p.add_argument("--source",choices=("github-code","the-stack","both"),default="github-code"); a=p.parse_args()
    out=ROOT/"files"/"data"; out.mkdir(parents=True,exist_ok=True); seen=set(); count=0
    for source in (["github-code","the-stack"] if a.source=="both" else [a.source]):
        if count>=a.count: break
        # Both sources are streamed so the entire multi-terabyte corpus is never downloaded.
        if source=="github-code":
            # The Hub repo retains a legacy Python loader script, which current
            # `datasets` correctly refuses to execute. Read its public Parquet
            # shards with the built-in safe Parquet loader instead.
            files="hf://datasets/codeparrot/github-code/data/*.parquet"
        else:
            files="hf://datasets/bigcode/the-stack/data/python/*.parquet"
        ds=load_dataset("parquet",data_files={"train":files},split="train",streaming=True)
        for row in ds:
            text=row.get("content") or row.get("code") or ""
            lang=(row.get("language") or row.get("lang") or "Python").lower()
            if not isinstance(text,str) or len(text.strip())<40 or "python" not in lang: continue
            text=text.replace("\x00","")
            digest=hashlib.sha256(text.encode("utf-8",errors="ignore")).hexdigest()
            if digest in seen: continue
            seen.add(digest); target=out/f"python_{count+1:06d}_{digest[:10]}.py"
            target.write_text(text,encoding="utf-8",errors="ignore"); count+=1
            if count%100==0: print(f"saved {count}/{a.count}",flush=True)
            if count>=a.count: break
    print(f"Finished: {count} unique files in {out}")
    if count<a.count: raise RuntimeError(f"Only found {count} Python files; dataset access/configuration may limit streaming.")
if __name__=="__main__": main()
