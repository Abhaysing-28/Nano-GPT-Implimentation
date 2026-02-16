"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
print("\n" + "="*60)
print("INITIALIZING TRAINING ENVIRONMENT")
print("="*60)

try:
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        print("\n[DDP] Initializing distributed data parallel training...")
        try:
            init_process_group(backend=backend)
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0
            seed_offset = ddp_rank
            print(f"[DDP] Rank: {ddp_rank}/{ddp_world_size}, Local Rank: {ddp_local_rank}, Device: {device}")
            
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            if gradient_accumulation_steps % ddp_world_size != 0:
                raise ValueError(f"gradient_accumulation_steps ({gradient_accumulation_steps}) must be divisible by ddp_world_size ({ddp_world_size})")
            gradient_accumulation_steps //= ddp_world_size
            print(f"[DDP] Adjusted gradient_accumulation_steps: {gradient_accumulation_steps}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize DDP: {e}")
            raise
    else:
        print("[INFO] Running on single GPU/CPU (no DDP)")
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"[INFO] Tokens per iteration: {tokens_per_iter:,}")
    print(f"[INFO] Device: {device}")
except Exception as e:
    print(f"[ERROR] Failed to initialize training environment: {e}")
    raise

if master_process:
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[INFO] Output directory: {out_dir}")
    except OSError as e:
        print(f"[ERROR] Failed to create output directory {out_dir}: {e}")
        raise

try:
    print("\n[SETUP] Configuring PyTorch settings...")
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(dtype)
    if ptdtype is None:
        raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: 'float32', 'bfloat16', 'float16'")
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    print(f"[SETUP] Device type: {device_type}, Dtype: {dtype}")
except Exception as e:
    print(f"[ERROR] Failed to configure PyTorch: {e}")
    raise

# poor man's data loader
print("\n[DATA] Setting up data loader...")
data_dir = os.path.join('E:\\', 'nanoGPT', 'data', 'shakespeare_char')
print(f"[DATA] Data directory: {data_dir}")

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

def get_batch(split):
    """Load a batch of data from memory-mapped files."""
    try:
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data_file = os.path.join(data_dir, 'train.bin')
        elif split == 'val':
            data_file = os.path.join(data_dir, 'val.bin')
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data = np.memmap(data_file, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    except Exception as e:
        print(f"[ERROR] Failed to load batch for split '{split}': {e}")
        raise

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
print("\n[METADATA] Loading vocabulary metadata...")
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
try:
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"[METADATA] Found vocab_size = {meta_vocab_size} (from {meta_path})")
    else:
        print(f"[METADATA] No metadata file found at {meta_path}")
except Exception as e:
    print(f"[WARNING] Failed to load metadata: {e}")

# model init
print("\n[MODEL] Initializing model...")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

try:
    if init_from == 'scratch':
        print("[MODEL] Creating new model from scratch...")
        if meta_vocab_size is None:
            print(f"[MODEL] No vocab size found, defaulting to GPT-2: 50304")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        print(f"[MODEL] Model created with config: {model_args}")
        
    elif init_from == 'resume':
        print(f"[MODEL] Resuming training from checkpoint in {out_dir}...")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        print(f"[MODEL] Loaded checkpoint from iteration {checkpoint.get('iter_num', 'unknown')}")
        
        # force these config attributes to be equal otherwise we can't even resume training
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"[MODEL] Resumed from iteration {iter_num}, best val loss: {best_val_loss:.4f}")
        
    elif init_from.startswith('gpt2'):
        print(f"[MODEL] Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
        print(f"[MODEL] Loaded pretrained GPT-2 model")
    else:
        raise ValueError(f"Invalid init_from value: {init_from}. Must be 'scratch', 'resume', or 'gpt2*'")
        
except Exception as e:
    print(f"[ERROR] Failed to initialize model: {e}")
    raise

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    print(f"\n[MODEL] Cropping model block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

try:
    model.to(device)
    print(f"[MODEL] Model moved to device: {device}")
except Exception as e:
    print(f"[ERROR] Failed to move model to device: {e}")
    raise

# initialize a GradScaler. If enabled=False scaler is a no-op
print("\n[TRAINING] Initializing training components...")
try:
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    print(f"[TRAINING] GradScaler enabled: {dtype == 'float16'}")
except Exception as e:
    print(f"[ERROR] Failed to initialize GradScaler: {e}")
    raise

# optimizer
try:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    print(f"[TRAINING] Optimizer configured: Adam - LR: {learning_rate}, WD: {weight_decay}")
    
    if init_from == 'resume':
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("[TRAINING] Optimizer state loaded from checkpoint")
        except Exception as e:
            print(f"[WARNING] Failed to load optimizer state: {e}")
except Exception as e:
    print(f"[ERROR] Failed to configure optimizer: {e}")
    raise

checkpoint = None # free up memory

# compile the model
if compile:
    try:
        print("\n[TRAINING] Compiling model with PyTorch 2.0 (this may take ~1 minute)...")
        unoptimized_model = model
        model = torch.compile(model)
        print("[TRAINING] Model compilation successful")
    except Exception as e:
        print(f"[WARNING] Model compilation failed, continuing without compilation: {e}")
        compile = False
else:
    print("\n[TRAINING] Model compilation disabled")

# wrap model into DDP container
if ddp:
    try:
        print("[TRAINING] Wrapping model in DistributedDataParallel...")
        model = DDP(model, device_ids=[ddp_local_rank])
        print("[TRAINING] DDP wrapping successful")
    except Exception as e:
        print(f"[ERROR] Failed to wrap model in DDP: {e}")
        raise

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets."""
    try:
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    except Exception as e:
        print(f"[ERROR] Failed to estimate loss: {e}")
        model.train()
        raise

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    """Calculate learning rate with cosine decay and warmup."""
    try:
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        if not (0 <= decay_ratio <= 1):
            raise ValueError(f"Invalid decay_ratio: {decay_ratio}")
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    except Exception as e:
        print(f"[ERROR] Failed to calculate learning rate at iteration {it}: {e}")
        raise

# logging
if wandb_log and master_process:
    try:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
        print("\n[LOGGING] Weights & Biases logging enabled")
    except Exception as e:
        print(f"[WARNING] Failed to initialize W&B logging: {e}")
        wandb_log = False
else:
    print("\n[LOGGING] W&B logging disabled")

# training loop
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Max iterations: {max_iters:,}")
print(f"Eval interval: {eval_interval:,}")
print(f"Log interval: {log_interval:,}\n")

try:
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    
    while True:
        # determine and set the learning rate for this iteration
        try:
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        except Exception as e:
            print(f"[ERROR] Failed to set learning rate at iteration {iter_num}: {e}")
            raise

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            try:
                losses = estimate_loss()
                print(f"[EVAL] step {iter_num:6d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                if wandb_log:
                    try:
                        wandb.log({
                            "iter": iter_num,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "lr": lr,
                            "mfu": running_mfu*100,
                        })
                    except Exception as e:
                        print(f"[WARNING] Failed to log to W&B: {e}")
                
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        try:
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'model_args': model_args,
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': config,
                            }
                            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                            torch.save(checkpoint, ckpt_path)
                            print(f"[CHECKPOINT] Saved to {ckpt_path} (val_loss: {best_val_loss:.4f})")
                        except Exception as e:
                            print(f"[ERROR] Failed to save checkpoint: {e}")
                            raise
            except Exception as e:
                print(f"[ERROR] Evaluation failed at iteration {iter_num}: {e}")
                raise
        
        if iter_num == 0 and eval_only:
            print("\n[INFO] eval_only=True, exiting after first evaluation")
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        try:
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"[ERROR] Training step failed at iteration {iter_num}: {e}")
            raise

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"[TRAIN] iter {iter_num:6d}: loss {lossf:.4f}, time {dt*1000:6.2f}ms, mfu {running_mfu*100:5.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            print(f"\n[INFO] Reached max iterations ({max_iters:,}), stopping training")
            break

except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user")
    if master_process and iter_num > 0:
        try:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            ckpt_path = os.path.join(out_dir, 'ckpt_interrupted.pt')
            torch.save(checkpoint, ckpt_path)
            print(f"[CHECKPOINT] Saved interrupted checkpoint to {ckpt_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save interrupted checkpoint: {e}")
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    raise
finally:
    print("\n" + "="*60)
    print("TRAINING COMPLETED/STOPPED")
    print("="*60)
    
    if ddp:
        try:
            destroy_process_group()
            print("[DDP] Process group destroyed")
        except Exception as e:
            print(f"[WARNING] Failed to destroy process group: {e}")
