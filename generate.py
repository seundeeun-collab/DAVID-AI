"""Sample code from a locally trained model."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from model import DecoderOnlyTransformer, ModelConfig

ROOT = Path(__file__).resolve().parent
_cache = {}


def _load(checkpoint_path=None, config_path=None):
    """Load tokenizer + model once and reuse them until the weights file changes."""
    weights = Path(checkpoint_path) if checkpoint_path else ROOT / "model.safetensors"
    cfgpath = Path(config_path) if config_path else ROOT / "model_config.json"
    if not weights.exists(): raise FileNotFoundError("No model.safetensors yet. Train with python train.py first.")
    key = (str(weights), weights.stat().st_mtime, str(cfgpath))
    if _cache.get("key") != key:
        tok = Tokenizer.from_file(str(ROOT / "files" / "tokenizer.json"))
        cfg = ModelConfig(**json.loads(cfgpath.read_text())) if cfgpath.exists() else ModelConfig(vocab_size=tok.get_vocab_size())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DecoderOnlyTransformer(cfg).to(device)
        model.load_state_dict(load_file(str(weights), device=str(device))); model.eval()
        _cache.update(key=key, tok=tok, cfg=cfg, device=device, model=model)
    return _cache["tok"], _cache["cfg"], _cache["device"], _cache["model"]


@torch.no_grad()
def sample(ids, max_new_tokens=200, temperature=.7, top_p=.9, top_k=50, stop_ids=(), stop_text=(), checkpoint_path=None, config_path=None):
    """Continue a list of token ids; return only the new ids. Stops at stop_ids or stop_text."""
    tok, cfg, device, model = _load(checkpoint_path, config_path)
    x = torch.tensor([ids], device=device); new = []
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -cfg.max_seq_len:]); scores = logits[:, -1, :].float() / max(temperature, 1e-5)
        if top_k:
            cutoff = torch.topk(scores, min(top_k, scores.size(-1))).values[:, -1, None]
            scores = scores.masked_fill(scores < cutoff, -float("inf"))
        probs = torch.softmax(scores, dim=-1); sorted_p, idx = torch.sort(probs, descending=True)
        remove = sorted_p.cumsum(-1) - sorted_p > top_p; sorted_p[remove] = 0
        probs = torch.zeros_like(probs).scatter(1, idx, sorted_p); probs = probs / probs.sum(-1, keepdim=True)
        nxt = torch.multinomial(probs, 1)
        if nxt.item() in stop_ids: break
        new.append(nxt.item()); x = torch.cat((x, nxt), dim=1)
        if stop_text and any(s in tok.decode(new[-12:]) for s in stop_text): break
    return new


def generate(prompt="def add(a, b):", max_new_tokens=200, temperature=.7, top_p=.9, top_k=50, checkpoint_path=None, config_path=None):
    """Return the prompt plus the model's continuation (stops when the model ends the file)."""
    tok = _load(checkpoint_path, config_path)[0]
    eot = tok.token_to_id("<|endoftext|>")
    ids = tok.encode(prompt).ids
    new = sample(ids, max_new_tokens, temperature, top_p, top_k, stop_ids={eot} if eot is not None else (), checkpoint_path=checkpoint_path, config_path=config_path)
    return tok.decode(ids + new, skip_special_tokens=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("prompt", nargs="?", default="def add(a, b):"); p.add_argument("--tokens", type=int, default=200); a = p.parse_args()
    code = generate(a.prompt, a.tokens); print(code); (ROOT / "output.py").write_text(code, encoding="utf-8")
