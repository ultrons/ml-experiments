from dataclasses import dataclass
import tiktoken
import random

import time    
import os
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
import math
from hellaswag import render_example, iterate_examples
from model import GPTConfig, GPT

# DataLoader
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        # get the shard filenames
        dataroot = "edu_fineweb10B"
        shards = os.listdir(dataroot)
        shards = [s for s in shards if split in s]  
        shards = sorted(shards) 
        shards = [os.path.join(dataroot, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found in {dataroot} with split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += self.num_processes * B * T
        if self.current_position + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            random.shuffle(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y



def get_most_likely_row(tokens, mask, logits):
    shift_logits  = (logits[..., :-1, :]).contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[...,1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# Run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1) != -1)
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

enc = tiktoken.get_encoding('gpt2')
total_batch_size = 524288
B = 32
T = 1024
assert total_batch_size % (B * T * ddp_world_size)  == 0, "Batch size must be a multiple of B*T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Global Batch Size: {total_batch_size}, Microbatch size: {B}, Gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split='train')
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split='val')


torch.set_float32_matmul_precision('high')


model = GPT(GPTConfig(vocab_size=50304))
model = model.to(device)
raw_model = model
# raw_model = torch.compile(model)
if ddp:
    model = DDP(raw_model, device_ids=[ddp_local_rank])

# Learning Rate Schedule
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 19073 * 3

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it >= max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)

logdir = 'log'
os.makedirs(logdir, exist_ok=True)
log_file = os.path.join(logdir, f"train_log_{ddp_rank}.txt")
with open(log_file, 'w') as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    
    if step % 2000 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_acc = 0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_acc += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_acc, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {val_loss_acc:.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val {val_loss_acc.item():.4f}\n")
                if step > 0 and (step % 4000 == 0 or last_step):
                    checkpoint_path = os.path.join(logdir, f"checkpoint_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_acc.item(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(checkpoint, checkpoint_path)
                    
        ## Hellaswag Eval
        if step > 0:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
                
            if ddp:
                num_total = torch.tensor(num_total, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
                
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"Hellaswag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")
                
        #---- Sampling
        if step > 0: 
            model.eval()
            num_return_sequences = 4
            max_length = 32
                    
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            x = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(47 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(x)
                    logits = logits [:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
                    
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

            
    # One step of optimization
    model.train()
    optimizer.zero_grad()
    loss_acc = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_acc += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # ms
    tokens_per_step = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_per_step / (t1 - t0)
    if master_process: 
        print(f"Step {step:4d}| loss: {loss_acc.item():.6f}| lr: {lr:.4e}| norm: {norm:.4f}| dt={dt:.2f}ms| tokens/sec={tokens_per_sec:.2f}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_acc.item():.6f}\n")

        
    

        
if ddp:
    destroy_process_group()
