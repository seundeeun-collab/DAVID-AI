# DAVID-AI: a from-scratch Transformer that learns Python, then English, then chat

This project contains a decoder-only Transformer written with PyTorch. It predicts the next GPT-2 token repeatedly. Flask serves a small website; `/generate` calls only `generate.py` and the local `model.safetensors`, with no ChatGPT, Ollama, or hosted inference fallback. The model starts with random weights. It cannot write useful code until it is trained, and three epochs on a home CPU can take days or longer. The default model has 45,171,200 parameters (width 512, six blocks); 25.7M of them are the 50,257-token embedding table, which is shared with the output layer. (124M is GPT-2 small: width 768, 12 blocks.)

## Folder contents

`model.py` contains learned token and position embeddings, six causal decoder blocks, eight-head attention, 4x GELU feed-forward layers, residual paths, layer normalization, dropout, and tied input/output token weights. `train.py` reads `.py` files, shuffles and splits them 90/10, trains with AdamW, cosine decay/warmup, clipping, GPU mixed precision when available, checkpoints, validation, JSONL logs, and early stopping. `generate.py` samples with temperature/top-p/top-k. `app.py`, `templates/`, and `static/` make the local web UI. Extra data/asset download scripts and `test.py` are included.

## Install and prepare (PowerShell, from this folder)

1. Install Python 3.11 from <https://www.python.org/downloads/release/python-3119/> and select “Add python.exe to PATH”. Check with `py -3.11 --version`. Python 3.12 also works for this code if the installer is unavailable.
2. Create and activate a clean environment: `py -3.11 -m venv .venv` then `\.venv\Scripts\Activate.ps1`. If script activation is blocked, use `\.venv\Scripts\python.exe` instead of `python` in the commands.
3. Install CPU PyTorch first: `python -m pip install --upgrade pip`; `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`; then check `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`. CUDA needs an NVIDIA GPU/driver. Intel UHD graphics does not run CUDA; do not install the CUDA toolkit for this computer.
4. Install the remaining packages: `python -m pip install -r requirements.txt`. PyTorch, torchvision, and torchaudio are listed there too; if the CPU install above differs from the pinned version, either install a matching PyTorch trio or update those three pins to the installed versions. Check imports with `python -c "import torch, torchvision, torchaudio, transformers, datasets, tokenizers, accelerate, safetensors, huggingface_hub, flask, flask_cors, numpy, tqdm, dotenv, requests; print(torch.__version__)"`.
5. Download tokenizer metadata: `python download_assets.py`; check `dir files`. The GPT-2 tokenizer is a vocabulary/encoding aid, not pretrained model weights.
6. Download 10,000 examples: `python download_data.py --count 10000 --source github-code`; check `dir files\data`. Dataset streaming needs internet and the Hugging Face `datasets` package. The Stack dataset can be access-controlled and requires accepting its Hugging Face terms/login; use `python -m pip install -U huggingface_hub`, then `hf auth login` if required, and try `python download_data.py --source the-stack --count 10000`.
7. Check code wiring: `python -m py_compile model.py train.py generate.py app.py test.py download_assets.py download_data.py`; then `python test.py`.

### What the installed packages do

| Package | Why it is here |
|---|---|
| Python 3.11 / pip | Runs the project; pip installs its libraries. |
| PyTorch (`torch`) | Tensor math, neural-network layers, gradients, and training. This computer uses the CPU build. |
| `torchvision` / `torchaudio` | Companion PyTorch packages for image/audio models; installed because requested, though this text-only model does not use them. |
| `transformers` | Hugging Face model utilities; installed for the requested toolkit, while this model architecture is implemented directly in `model.py`. |
| `datasets` | Streams Hugging Face datasets without mirroring the full corpus. |
| `tokenizers` | Fast Rust-backed GPT-2 tokenization used by the training and generation scripts. |
| `accelerate` | Common device/mixed-precision training helpers; available for future scaling. |
| `safetensors` | Saves and loads model weights in a tensor-only file format. |
| `huggingface_hub` | Connects to Hugging Face dataset/model file storage. |
| `Flask` / `Flask-Cors` | Serves the local website and controls which local origins may call its API. |
| `numpy` | Array support used across the data and ML ecosystem. |
| `tqdm` | Progress bars for longer data/training jobs. |
| `python-dotenv` | Lets future configuration load local `.env` settings without hard-coding secrets. |
| `requests` | Downloads tokenizer files from Hugging Face. |
| `pyarrow` (installed as a `datasets` dependency) | Reads the dataset's Parquet shards while streaming. |

## Train and use

`python train.py` uses the requested default three epochs, batch size 16, 512-token context, width 512, six blocks, learning rate 0.0005, and accumulation of four batches. This computer has only 3.8 GB of RAM and no CUDA GPU, so the requested full-size batch may not fit; a full-size model can be attempted with `python train.py --batch-size 1 --context 128 --accumulation 16`, at the cost of much slower training. For a quick pipeline-only diagnostic run use `python train.py --epochs 1 --batch-size 1 --context 128 --accumulation 1 --dim 64 --layers 2 --max-steps 5`. That tiny model is only a smoke test and is not interchangeable with the default architecture. On the first run, `train.py` tokenizes `files/data` one file at a time into `files/tokens/train.bin` and `val.bin` (memory-mapped, uint16), so RAM use stays flat even with a million files; later runs reuse the cache unless the file list changes or you pass `--rebuild-cache`. Training writes `model.safetensors`, `model_config.json`, `training.jsonl`, and `checkpoint-<stage>-latest.safetensors` (overwritten every 500 steps, so a cut-off run never loses more than that). Check files with `dir *.safetensors`.

The fixed GPT-2 tokenizer itself limits each token id to the original GPT-2 vocabulary. The model config uses the actual tokenizer size. Evaluation reports validation cross-entropy (“loss”) and `exp(loss)` (“perplexity”); it does not claim a meaningful code-completion accuracy. Run `python generate.py "def add(a, b):"`; the script prints the result and writes `output.py`. Start the UI with `python app.py`, then visit <http://localhost:5000>. Type `Write a function to calculate total price`, press Generate Code, then use Copy or Run Code. The model learned only from raw `.py` files, so the website turns a plain-English request into a Python comment (`# Write a function...`) before generating. You get better results by starting the code yourself, for example `# calculate total price` on one line and `def total_price(items):` on the next. Generated code quality depends on the data and training time.

The web runner permits a small AST-checked subset and built-in functions for this educational local demo. It is not an operating-system security sandbox. Keep the server bound to localhost and do not use it to run arbitrary untrusted code or expose it to the network.

## Teach it English and chat (Stages 1 and 2)

The base model only ever saw Python files, so it cannot hold a conversation. Two more training stages fix that. Each stage continues from the last `model.safetensors`, so nothing learned is thrown away.

| Stage | Command | Data | What it learns |
|---|---|---|---|
| 0 (optional) | `python train.py` | Your Python files | Python code |
| 1 | `python train.py --stage english` | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (CDLA-Sharing-1.0), plus 10% Python | Fluent, simple English |
| 2 | `python train.py --stage chat` | [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (Apache-2.0), plus 10% English and 10% Python | Answering as an assistant |

Download and tokenize the data first: `python prepare_data.py --english 600000 --chat 150000` (add `--code` to mix in your Python files). It streams and writes token files under `files/tokens/`, so RAM stays small.

**How chat training works:** each conversation is written as `<|user|>` / `<|assistant|>` turns (see `chat_format.py`). The model is only graded on the assistant's words and the end token after each reply, never on the user's lines. That is what teaches it to answer instead of repeating the question, and to stop when its reply is done. Chat uses the full 1,024-token window, and conversations longer than that are skipped so each is seen whole.

**Where to train:** a home CPU would take days. Use the free Kaggle GPU: create a new Kaggle notebook and use File → Import Notebook to upload `kaggle_train.ipynb`, turn on **GPU T4 x2** and **Internet**, then **Save Version → Save & Run All**. It runs both stages within one 12-hour session, saving as it goes, and leaves `david-ai-trained.zip` in the Output tab. Unzip `model.safetensors` and `model_config.json` into this folder on your PC.

**Use it:** `python chat.py` chats in the terminal; `python app.py` adds a Chat box to the website (POST `/chat` with `{"messages": [{"role": "user", "content": "Hi"}]}`). `python eval.py` measures validation loss on every dataset and asks the same English, chat and coding questions each time, appending results to `eval_results.json` so you can compare runs.

**Useful flags:** `--max-hours 6` stops cleanly and saves; `--resume file.safetensors` continues from a specific file; `--from-scratch` ignores existing weights; `--mix-code 0` / `--mix-english 0` turn the mixing off; `--eval-every 1000` keeps the best weights during long single-epoch runs. Each stage also saves a copy as `model-english.safetensors` / `model-chat.safetensors` so you can roll back.

**Honest expectations:** the smallest model Hugging Face trained on smol-smoltalk had 135M parameters; this one has 45M. Expect short, simple answers in clear English that are often wrong on facts. It is a real chatbot you trained yourself, not a ChatGPT replacement.

## Getting more training data

`download_data.py --count 10000 --source github-code` streams examples and stores one cleaned `.py` file per unique source. For a million files use `python download_data.py --count 1000000 --source github-code`; expect substantial bandwidth, disk (often many tens of GB), and hours/days. Check available storage first with `Get-PSDrive C`. Better for production: download to a dedicated large disk, filter licenses and duplicates, use a managed dataset snapshot/cache, and train from sharded token files rather than keeping millions of individual files. `bigcode/the-stack` may be gated, so authenticate with an account that has accepted its terms. Check source licenses before redistribution.

## Reading the training numbers (like you're 10)

| Word | Plain meaning |
|---|---|
| Token | A piece of text (sometimes a whole word, sometimes part of one). The model predicts one token at a time. |
| Vocab size | How many different token pieces it knows; GPT-2 has 50,257. |
| Batch | A handful of examples learned together before adjusting weights. |
| Step | One weight update after the batch or accumulated batches. |
| Epoch | One pass through the training examples. |
| Gradient | A hint telling each weight which direction might reduce mistakes. |
| Learning rate | How big a weight adjustment to try. Too high can jump past a good answer; too low learns slowly. |
| Loss | How surprised/wrong the next-token guesses were. Lower is better on held-out examples too. |
| Perplexity | Roughly how many choices the model feels unsure between; lower is better. It is `exp(loss)`. |
| Accuracy | Fraction exactly right. Token accuracy is a crude measure for code; this project reports loss/perplexity instead. |
| Overfitting | Memorizing training examples while getting worse at new ones. Watch validation loss. |

Loss numbers are only rough intuition, not promises: `4.0` = bad guessing; `3.0` = starting; `2.0` = okay simple functions; `1.0` = good real code; `0.5` = very good; `0.1` = maybe overfitting. These thresholds depend on tokenization and dataset. In the terminal, training loss should generally go down. Validation loss should also trend down. If training falls while validation rises for multiple evaluations, overfitting may be starting; stop, add clean varied data, or regularize. Stop when validation no longer improves or resources/time run out. The script's early stopping waits for two non-improving epochs.

## Make it better

First get clean, licensed data and a working small run. Train for more steps/epochs if validation loss keeps improving. A million files helps only if the data is varied, relevant, and deduplicated. Larger models need much more memory and compute: 12 blocks and width 768 are not a simple quality switch on a home CPU. Fine-tune a pretrained code model if the goal is useful code rather than learning Transformer internals; this project deliberately starts from random weights. For this from-scratch model, the full defaults may require a GPU and days/weeks of training.

## Recent fixes

- Training loss is now averaged over every micro-batch since the last log line. Before, with `--accumulation 4` the printed loss was about 4x too high.
- The last, smaller gradient-accumulation group of each epoch is scaled by its real size.
- `model.safetensors` is always saved, even when the validation set is too small to evaluate (before, it was never written in that case).
- Data is tokenized to a memory-mapped cache instead of one big in-RAM tensor.
- Attention uses `is_causal=True`, so PyTorch can pick its fastest kernel without storing a 1024x1024 mask.
- `generate.py` loads the model once and reuses it, and stops when the model emits end-of-file.
- New `--heads` and `--log-every` training flags.
- Staged training (`--stage code|english|chat`), chat data with a train mask, `--max-hours`, best-weights checks during an epoch, and tokens/sec in the log.
- Loss ignores masked (-100) targets; a window with none left gives 0 instead of NaN.

## Common problems

- `torch not found` / `ModuleNotFoundError`: activate `.venv`, then rerun `python -m pip install -r requirements.txt` and check the import command above.
- CUDA out of memory: this computer has Intel graphics, so CUDA is unavailable. On NVIDIA hardware, lower batch/context/dim/layers, or use gradient accumulation; install the correct PyTorch CUDA wheel for the installed driver.
- Tokenizer missing: run `python download_assets.py`, then `dir files`.
- No training files / model file: run the dataset download, check `dir files\data`, then train. Generation and the website explain when weights are absent.
- Port 5000 occupied: close the other local server or change the `port=5000` value at the bottom of `app.py` and visit that port.
- A package is missing: activate the environment and install it with `python -m pip install PACKAGE`; add/pin it in requirements.txt for repeatability.
