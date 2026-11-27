"""Generate code with your trained model.

python generate.py --lang python --task "Write a function to add two numbers"
Languages: python javascript typescript html css java cpp go rust php sql
"""
import argparse, os
import torch, tiktoken
from model import GPTConfig, GPT

COMMENT = {"python": "# {}", "javascript": "// {}", "typescript": "// {}", "html": "<!-- {} -->",
           "css": "/* {} */", "java": "// {}", "cpp": "// {}", "go": "// {}", "rust": "// {}",
           "php": "<?php\n// {}", "sql": "-- {}"}

def build_prompt(lang, task):
    return f"<|lang:{lang}|>\n" + COMMENT[lang].format(task) + "\n"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="python", choices=sorted(COMMENT))
    p.add_argument("--task", default="Write a function to add two numbers")
    p.add_argument("--out_dir", default="out-code")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_k", type=int, default=40)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(a.out_dir, "ckpt.pt"), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
    model.eval().to(device)
    enc = tiktoken.get_encoding("gpt2")
    prompt = build_prompt(a.lang, a.task)
    x = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    with torch.no_grad():
        y = model.generate(x, a.max_new_tokens, temperature=a.temperature, top_k=a.top_k)
    text = enc.decode(y[0].tolist())
    print(text.split("<|endoftext|>")[0].split("<|lang:")[1].split("|>\n", 1)[1])
