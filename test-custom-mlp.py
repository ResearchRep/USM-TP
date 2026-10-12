#!/bin/env python3

#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

# Simple tests for split Qwen3 MLP implementation

import split_matmuls
import force_usm
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
import torch
from time import perf_counter_ns as rn

bf16_xpu = dict (device = "xpu", dtype = torch.bfloat16)

c = AutoConfig.from_pretrained ("Qwen/Qwen3-8B")
mlp = Qwen3MLP (c).to (**bf16_xpu)
smlp = split_matmuls.SplitQwen3MLP (mlp, int (c.intermediate_size * 0.25), b"")
x = torch.rand (1, 1, c.hidden_size, **bf16_xpu)

print (smlp)

def maxerr (a, b):
    res = (a - b).flatten ().abs ().max ()
    return float (res)

with torch.inference_mode ():
    ref = mlp (x)
    mys = smlp (x)
    torch.xpu.synchronize ()
    print (f"Max. error: {maxerr (ref, mys):.0e}")
