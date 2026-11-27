# Train a ~51M parameter code model on 11 languages. Fits a free Colab/Kaggle T4 (16GB).
# Run:      python train.py config/train_code.py
# Resume:   python train.py config/train_code.py --init_from=resume
out_dir = 'out-code'
eval_interval = 250        # check val loss + save checkpoint every 250 steps
eval_iters = 100
log_interval = 10
always_save_checkpoint = True   # Colab can disconnect: always keep the latest
wandb_log = False
dataset = 'code'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 512           # how many tokens the model sees at once
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 6e-5
warmup_iters = 500
beta2 = 0.95
dtype = 'float16'          # T4 has no fast bfloat16
compile = False           # safest on Colab T4; set True for extra speed if it works
