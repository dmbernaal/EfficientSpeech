import torch
import yaml
import time
import numpy as np
import validators

import torch.nn as nn
from pathlib import Path
import os
import argparse
from typing import List, Tuple , Dict, Union, String
"""
python3 demo.py --checkpoint https://github.com/roatienza/efficientspeech/releases/download/pytorch2.0.1/tiny_eng_266k.ckpt \
  --infer-device cpu --text "the quick brown fox jumps over the lazy dog" --wav-filename fox.wav
"""
# UTILS
from symbols import symbols

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dims must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.proj = nn.Linear(dim * num_heads, dim)

    def forward(self, x, mask=None, pool=1):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C).permute(2, 0, 3, 1, 4) # dim => [3, B, num_heads, T, C]
        q, k, v = qkv.unbind(0) 
    
class MixFFN(nn.Module):
    def __init__(self):
        raise NotImplementedError

class Encoder(nn.Module):
    # Phoneme encoder
    def __init__(self, depth=2, embed_dim=128, kernel_size=3, expansion=1, reduction=4, head=1):
        super().__init__()
        small_embed_dim = embed_dim // reduction
        dim_ins = [embed_dim] + [small_embed_dim*(2**i) for i in range(depth-1)]
        heads = [head*(i+1) for i in range(depth)]
        kernels = [kernel_size-(2 if i > 0 else 0) for i in range(depth)]
        padding = [k//2 for k in kernels]
        strides = [1] + [2 for _ in range(depth-1)]
        
        self.embeds = nn.Embedding(len(symbols) + 1, embed_dim, padding_idx=0)  
        self.dim_outs = [small_embed_dim*(2**i) for i in range(depth)]

        self.attn_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(dim_in, dim_in, kernel_size=kernel, stride=stride, padding=padding, bias=False),
                nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=False),
                SelfAttention(dim_out, num_heads=head),
                MixFFN(dim_out, expansion),
                nn.LayerNorm(dim_out),
                nn.LayerNorm(dim_out)
            ])
            for dim_in, dim_out, head, kernel, stride, padding in zip(dim_ins, self.dim_outs, heads, kernels, strides, padding)
        ])


class PhonemeEncoder(nn.Module):
    # encodes phenomes to accoustic features
    def __init__(self, pitch_stats=None, energy_stats=None, depth=2, reduction=4, head=1, embed_dim=128, kernel_size=3, expansion=1):
        super().__init__()

class EfficientSpeech(nn.Module):
    # EfficientSpeech model
    def __init__(self, depth=2, n_blocks=2, block_depth=2, reduction=4, head=1, embed_dim=128, kernel_size=3, decoder_kernel_size=3, expansion=1):
        super().__init__()

if __name__ == '__main__':
    print('Inference')