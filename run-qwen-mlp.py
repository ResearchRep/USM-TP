#!/bin/env python3

#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser as AP
ap = AP (description = "Run a Qwen3 MLP workload",
         epilog = "See the `split-qwen3_mlp_fwd` function from split_matmuls.cpp for more info on FLAGS")
ap.add_argument ("--variant", default = "8B",
                 help = "Use MLP from Qwen3-VARIANT")
ap.add_argument ("--cpu-frac", type = float, default = 0.0,
                 help = "CPU work fraction")
ap.add_argument ("--flags", default = "",
                 type = lambda s: bytes (s, encoding = "ASCII"),
                 help = "Flags to pass to `split_qwen3_mlp_fwd`")
ap.add_argument ("--targets", default = ["gate_proj"],
                 type = lambda s: s.split (","),
                 help = "Linear layers to replace "
                        "(only for --impl=gemv-split)")
ap.add_argument ("--impl", default = "gemv-split",
                 choices = ("mlp-split", "gemv-split"),
                 help = "Implementation to use")
args = ap.parse_args ()
print (args)

from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from transformers import AutoConfig
import split_matmuls
from time import perf_counter_ns as rn
import torch
import os

import force_usm

bf16_xpu = dict (device = "xpu", dtype = torch.bfloat16)

config = AutoConfig.from_pretrained (f"Qwen/Qwen3-{args.variant}")

def replace_linear (model, crit):
    #flags = bytes (args.flags, encoding = "ASCII")
    for name, mod in model.named_modules ():
        cpu_N = crit (name, mod)
        if cpu_N is not None:
            model.set_submodule (name,
                split_matmuls.SplitLinear (mod, cpu_N = cpu_N, flags = args.flags))

def timer (work, *args):
    s = rn ()
    res = work (*args)
    T = rn () - s

    return T, res

def make_mlp ():
    ref = Qwen3MLP (config).to (**bf16_xpu)
    if args.impl == "gemv-split":
        def c (n, m):
            if n in args.targets:
                return int (args.cpu_frac * m.weight.shape [-2])
        replace_linear (ref, crit = c)
        return ref
    elif args.impl == "mlp-split":
        res = split_matmuls.SplitQwen3MLP (
            ref, int (args.cpu_frac * config.intermediate_size), args.flags)
        return res
    assert False, f"Invalid impl: {args.impl}"

def main ():
    mlp = make_mlp ()
    print (mlp)
    x = torch.rand (1, 1, config.hidden_size, **bf16_xpu)

    def w ():
        for _ in range (50):
        #while True:
            mlp (x)
        torch.xpu.synchronize ()

    # Warmup
    w ()

    os.environ ["PTI_ENABLE_COLLECTION"] = "1"
    T, res = timer (w)
    os.environ ["PTI_ENABLE_COLLECTION"] = "0"
    print (f"{T:.2e}")

with torch.inference_mode (): main ()
