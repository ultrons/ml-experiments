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


#---- Model Definition

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.RESIDUAL = True
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # QKV projection
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.RESIDUAL = True 

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )
        
    def forward(self, x):
        
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2) #[B, nh, T, hs]
        q = q.view(B, T, self.n_head, -1).transpose(1, 2) 
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        
        # Using Flash Attention 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        #att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        #att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1)
        #y = att @ v #[B, nh, T, T], [B, nh, T, hs] -> [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
        
        
        

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList(
                [Block(config) for _ in range(config.n_layer)]
            ),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # Weights are shared between the embedding layer and the linear head
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL'):
                std *= 2 * self.config.n_layer ** -0.5
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Shape T
        pos_emb = self.transformer.wpe(pos) # T, n_embed
        tok = self.transformer.wte(idx) # B, T, n_embed
        
        x = pos_emb + tok
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
        

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >=2 ]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decay params: {num_decay_params}, No decay params: {num_nodecay_params}")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")
        
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768), #124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024), #350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280), #774M
            'gpt2-xl': dict(n_layer=48, n_head=35, n_embed=1600), #1558M
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
            ]
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        return model
    
    
#--- Training

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
