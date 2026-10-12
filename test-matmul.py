#!/bin/env python3

#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

M = 1; N = 10240

from argparse import ArgumentParser
ap = ArgumentParser ()
ap.add_argument ("--cpu-N", type = int, default = N // 2)
ap.add_argument ("--flags", default = b"",
    type = lambda s: bytes (s, encoding = "ASCII"))
args = ap.parse_args ()
print (args)

import torch
import split_matmuls
import force_usm
from time import perf_counter_ns as rn

bf16_xpu = dict (dtype = torch.bfloat16, device = "xpu")
layer = torch.nn.Linear (N, N, bias = False, **bf16_xpu)
src = torch.rand (1, M, N, **bf16_xpu)
wt = layer.weight
wt_cpu = wt.cpu ()

def split_linear (*p, **k):
    return split_matmuls.split_linear (*p, **k, cpu_N = args.cpu_N, flags = args.flags, wt_cpu = wt_cpu)

with torch.inference_mode ():
    power = 3

    # Compute the reference output
    ref = src
    for _ in range (power):
        ref = layer (ref)
    torch.xpu.synchronize ()

    # Enqueue a LOT of work, discard outputs
    for _ in range (500):
        split_linear (src, wt)

    # Compute output, should equal reference output
    # This will have to wait for the previous work
    res = src
    for _ in range (power):
        res = split_linear (res, wt)
    torch.xpu.synchronize ()

    max_err = (res - ref).flatten ().abs ().max ()
    print (f"Max. error: {max_err:.0e}")
    #print (ref); print (res)
