#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:24:02 2023

@author: sho
"""
import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from einops_exts import repeat_many, rearrange_many

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # calculate query / key similarities
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Net(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(
            cfg.embed.token_size, cfg.dim
        )
        
        self.pe_embedding = nn.Linear(cfg.embed.pe_dim, cfg.dim)
        
        self.last_layer = nn.Sequential(
            nn.Linear(cfg.dim, int(cfg.dim/2)),
            nn.ReLU(),
            nn.Linear(int(cfg.dim/2), 1),
            nn.Sigmoid()
        )
        
        #self.attn = Attention(dim = cfg.dim)
        #self.fft_1 = FeedForward(dim = cfg.dim, mult = cfg.encoder.ff_mult)
        #self.cross_attn = CrossAttention(dim = cfg.dim)
        #self.fft_2 = FeedForward(dim = cfg.dim, mult = cfg.encoder.ff_mult)
        
        depth = cfg.encoder.depth
        self.attn = nn.ModuleList([])
        for _ in range(depth):
            self.attn.append(nn.ModuleList([
                Attention(dim = cfg.dim),
                FeedForward(dim = cfg.dim, mult = cfg.encoder.ff_mult)
            ]))
        
    def forward(self, token, pe):
        x = self.token_embedding(token) + self.pe_embedding(pe)
        
        for attn, ff in self.attn:
            x = x + attn(x)
            x = x + ff(x)
            
        x, _ = torch.max(x, dim = 1)
        out = self.last_layer(x)
        
        return out
            
        
        
        
        
        
        













