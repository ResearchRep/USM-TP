#!/bin/env python3

#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser as AP
ap = AP (description = "Run a split matmul workload",
         epilog = "See the `split_matmul` function from split_matmul.cpp "
                  "for more info on FLAGS")
ap.add_argument ("--cpu-frac", type = float, default = 0.5,
                 help = "CPU work fraction")
ap.add_argument ("--reps", type = int, default = 10,
                 help = "Number of matmuls to run")
ap.add_argument ("--flags", default = b"",
                 type = lambda s: bytes (s, encoding = "ASCII"),
                 help = "Flags to pass to `split_matmuls.split_linear`")
ap.add_argument ("--warmup", action = "store_true",
                 help = "Whether to run a warmup pass")
ap.add_argument ("--use-pt", action = "store_true",
                 help = "Use `torch.nn.functional.linear` "
                        "instead of `split_matmuls.split_linear`")
ap.add_argument ("--M", type = int, default = 1)
ap.add_argument ("--K", type = int, default = 4096)
ap.add_argument ("--N", type = int, default = 10240,
                 help = "Matrix dimensions; "
                        "the matmul is of size [MxK] x [KxN]")
args = ap.parse_args ()
print (args)

import split_matmuls
import torch
import force_usm
from os import environ
from time import perf_counter_ns as rn

cpu_N = int (float (args.cpu_frac) * args.N)

xpu_bf16 = dict (device = "xpu", dtype = torch.bfloat16)
a = torch.rand (1, args.M, args.K, **xpu_bf16)
b = torch.rand (args.N, args.K, **xpu_bf16)
b_cpu = b.cpu ()

print ("Shapes:", a.shape, b.shape)
print ("cpu_N, flags:", cpu_N, args.flags)

if args.use_pt:
    matmul_line = "torch.nn.functional.linear (a, b)"
else:
    matmul_line = "split_matmuls.split_linear (a, b, cpu_N = cpu_N, flags = args.flags, wt_cpu = b_cpu)"

work = f"""
for _ in range (args.reps):
    {matmul_line}
torch.xpu.synchronize ()
"""

if args.warmup:
    exec (work, globals (), locals ())

torch.xpu.synchronize ()
print (f"Running {args.reps} reps")

environ ["PTI_ENABLE_COLLECTION"] = "1"
T = rn ()
exec (work, globals (), locals ())
T = rn () - T
environ ["PTI_ENABLE_COLLECTION"] = "0"
print (f"Total: {T / 1e6:.4f} ms")
print (f"Average: {T / args.reps / 1e6:.4f} ms")
