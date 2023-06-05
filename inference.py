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

class PhonemeEncoder:
    # encodes phenomes to accoustic features
    def __init__(self, pitch_stats=None, energy_stats=None, depth=2, reduction=4, head=1, embed_dim=128, kernel_size=3, expansion=1):
        raise NotImplementedError

class EfficientSpeech:
    # EfficientSpeech model
    def __init__(self, depth=2, n_blocks=2, block_depth=2, reduction=4, head=1, embed_dim=128, kernel_size=3, decoder_kernel_size=3, expansion=1):
        raise NotImplementedError


if __name__ == '__main__':
    print('Inference')