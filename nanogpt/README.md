# DAVID-AI on nanoGPT (code in 11 languages)

These files plug into [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) to train a code model on a free GPU.

| File | Goes to (inside nanoGPT) | What it does |
|---|---|---|
| `colab_train.ipynb` | open in Google Colab | Whole pipeline, step by step |
| `prepare_code.py` | `data/code/prepare.py` | Streams real code (11 languages) from codeparrot/github-code-clean, writes train.bin/val.bin |
| `train_code.py` | `config/train_code.py` | ~51M parameter model sized for a 16GB T4 |
| `train_with_dashboard.py` | repo root | Runs train.py and sends loss/step/GPU to the website |
| `generate.py` | repo root | `python generate.py --lang go --task "Write a function to add two numbers"` |
| `export.py` | repo root | ckpt.pt → model.safetensors, config.json, tokenizer.json, model.onnx, model.int8.onnx |

Prompt format the model learns: `<|lang:python|>` then a comment with the task, then code.
Language tags: python javascript typescript html css java cpp go rust php sql.
