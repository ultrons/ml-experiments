import math
import logging

logger = logging.getLogger(__name__)

"""
PyTorch Implementation of a minimal GPT model
It's inspired from everything transformer and karpathy/minGPT
It's intended just as a base implementation to experiment with
other architectures and tasks
"""


class GPTConfig:
    """
    GPT is decoder only class of transformer model
    It can be defined in terms of the following config parameters

    d_model:
    Imagine that model input has shape [batch, length, embed]
    We will call embed dimension as the d_model

    num_heads:
    Number of attention heads

    head_dim:
    The dimension of each of the projection
    We choose head_dim such that head_dim * num_head = d_model

    mlp_dim:
    Dimension of the mlp layer after the self-attention

    num_layers:
    Number of transformer layers
    """
    d_model = 768
    num_head = 12
    head_dim = d_model // num_head
    num_layer = 12
    mlp_dim = 4 * 768


    dropout = 0.1

    # max sequence len
    seq_len = 512
    vocab_size = 32768

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """"""
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        # Create QKV projection layer
        self.qkv = nn.Linear(config.d_model, 3*config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout
        # Causal mask
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len))
        mask = mask.view(1, 1, config.seq_len, -1)
        self.register_buffer("mask", mask)
        self.num_heads = config.num_heads


    def forward(self, x):
        B, T, D = x.size()
        qkv_proj = self.qkv(x).view(3, B, T, D)
        query = qkv_proj[0].view(B, T, self.num_heads, -1)
        key = qkv_proj[1].view(B, T, self.num_heads, -1)
        value = qkv_proj[2].view(B, T, self.num_heads, -1)

        outer = torch.einsum('bshq,bthq->bsth', query, key)
        scale = math.sqrt(key.size(-1))
        alpha = F.softmax(outer/scale, dim=-2)
        inner = torch.einsum('bsth,bthq->bshq', alpha, value)

        inner = inner.contiguous().view(B,T,D)
        y = self.out(inner)
        return y


class TransformerLayer(nn.Module):
    """"""
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.ln1 = nn.Layernorm(config.d_model)
        self.ln2 = nn.Layernorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.wi = nn.Linear(config.d_model, config.mlp_dim)
        self.wo = nn.Linear(config.mlp_dim, config.d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        y = self.wi(self.ln2(x))
        y = self.activation(y)
        y = self.wo(y)
        y = y + x
        return y




class GPT(nn.Module):
    """"""
    def __init__(self, config):
        super(GPT, self).__init__()
        self.input_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, confg.seq_len, config.d_model))
        self.attn_stack = nn.Sequential(
            *[TransformerLayer(config) for _ in range(config.num_layers)])
        self.ln = nn.Layernorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        self.seq_len = config.seq_len

        self.apply(self._init_weights)

        logger.info(f"Number of parameters:{sum(p.numel for p in self.parameters())}")


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.2)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Layernorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.Layernorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in self.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full_param_name)
                if param_name.endswith and isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_param_name)

        # Special case for the position embedding parameter
        # not decayed in the GPT module
        no_decay.add('pos_emb')

        # Validate that all the params are covered
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "found common parameters in the decay and no decay list"
        assert len(param_dict.keys() - union_params) == 0, "not all parameters present in in the union of decay and no decay list"

        optim_groups = [

            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},

        ]
        optimizer = torch.optim.AdamW(optim_groups, lf=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, target=None):
        B, T = x.size()
        assert T <= self.seq_len, f"Sequence length {T} exceeds Max Sequence Length {self.seq_len} "
        token_emb = self.input_emb(x)
        position_emb = self.pos_emb[:,:T, :]
        x = token_emb + pos_emb
        x = self.attn_stack(x)
        x = self.ln(x)
        logits = self.output_head(x)
        B, T, V = logits.size()
        loss = None

        if target is not None:
            loss = nn.cross_entropy(logits.view(-1, V), targets.view(-1))

        return logits, loss










