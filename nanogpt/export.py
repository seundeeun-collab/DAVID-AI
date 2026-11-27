"""Turn a nanoGPT checkpoint into files the website uses.

python export.py --out_dir out-code
Writes into export/: model.safetensors, config.json, tokenizer.json, model.onnx
"""
import argparse, json, os
import torch
from safetensors.torch import save_file
from model import GPTConfig, GPT

p = argparse.ArgumentParser()
p.add_argument("--out_dir", default="out-code")
p.add_argument("--dest", default="export")
a = p.parse_args()
os.makedirs(a.dest, exist_ok=True)

ckpt = torch.load(os.path.join(a.out_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
args = ckpt["model_args"]
model = GPT(GPTConfig(**args))
state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(state)
model.eval()

# 1) safetensors (lm_head shares weights with the embedding, so save it once)
tensors = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
save_file(tensors, os.path.join(a.dest, "model.safetensors"))

# 2) config.json
cfg = dict(args, iter_num=int(ckpt.get("iter_num", 0)), best_val_loss=float(ckpt.get("best_val_loss", 0)),
           tokenizer="gpt2", prompt_format="<|lang:TAG|>\n<comment with task>\n")
json.dump(cfg, open(os.path.join(a.dest, "config.json"), "w"), indent=2)

# 3) tokenizer.json (same GPT-2 BPE that tiktoken uses in training)
from transformers import GPT2TokenizerFast
GPT2TokenizerFast.from_pretrained("gpt2").backend_tokenizer.save(os.path.join(a.dest, "tokenizer.json"))

# 4) model.onnx for running in the browser: input ids -> logits of the last token
class LastLogits(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, idx): return self.m(idx)[0][:, -1, :]
dummy = torch.randint(0, 50257, (1, 16))
torch.onnx.export(LastLogits(model), (dummy,), os.path.join(a.dest, "model.onnx"),
                  input_names=["input_ids"], output_names=["logits"],
                  dynamic_axes={"input_ids": {1: "seq"}}, opset_version=17, dynamo=False)

# check the ONNX file gives the same answer as PyTorch
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession(os.path.join(a.dest, "model.onnx"))
test = torch.randint(0, 50257, (1, 40))
want = LastLogits(model)(test).detach().numpy()
got = sess.run(None, {"input_ids": test.numpy()})[0]
diff = float(np.abs(want - got).max())
print(f"ONNX check: max difference {diff:.6f}", "OK" if diff < 1e-3 else "MISMATCH")
# 5) smaller int8 copy for phones/browsers (about 4x smaller)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(os.path.join(a.dest, "model.onnx"), os.path.join(a.dest, "model.int8.onnx"), weight_type=QuantType.QInt8)
sess8 = ort.InferenceSession(os.path.join(a.dest, "model.int8.onnx"))
got8 = sess8.run(None, {"input_ids": test.numpy()})[0]
same_top = int((np.argsort(-want[0])[:5] == np.argsort(-got8[0])[:5]).sum())
print(f"int8 check: {same_top}/5 top next-token guesses match full model")
for f in sorted(os.listdir(a.dest)):
    print(f"{f}: {os.path.getsize(os.path.join(a.dest, f))/1e6:.1f} MB")
