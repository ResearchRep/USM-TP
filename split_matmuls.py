#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

import ctypes
from ctypes import (
    CDLL,
    c_int, c_void_p, c_char_p, c_ulong,
)
import atexit
import torch

bf16_xpu = dict (dtype = torch.bfloat16, device = "xpu")

# Load the shared library
this_dir = __file__.rsplit ("/", maxsplit = 1)[0]
libsplit_matmuls = CDLL (f"{this_dir}/libsplit_matmuls.so")

# Function signatures
libsplit_matmuls.split_matmul.argtypes = [
    # M, K, N
    c_int, c_int, c_int,
    # src, wt, bias, dst
    c_void_p, c_void_p, c_void_p, c_void_p,
    # cpu_N, flags, wt_cpu
    c_int, c_char_p, c_void_p
]
libsplit_matmuls.split_qwen3_mlp_fwd.argtypes = [
    # Hidden, intermediate sizes and cpuI
    c_int, c_int, c_int,
    # src, gate_wt, up_wt, down_wt
    c_void_p, c_void_p, c_void_p, c_void_p,
    # out, flags
    c_void_p, c_char_p
]
libsplit_matmuls.enqueue_empty_kernel.argtypes = []
libsplit_matmuls.benchmark_gemv.argtypes = [
    c_int, c_int,
    c_void_p, c_void_p, c_void_p,
    c_char_p, c_int,
]
libsplit_matmuls.benchmark_gemv.rettype = c_ulong

# Initialization
libsplit_matmuls.setup (torch.xpu.current_stream ())
atexit.register (libsplit_matmuls.setdown)

def split_linear (src, wt, *, cpu_N = None, flags = b"", wt_cpu = None):
    "Thin wrapper around `split_matmul`, for use with PyTorch"
    _one, M, K = src.shape
    assert _one == 1
    N, _K = wt.shape
    assert _K == K, _K

    wt_cpu_ptr = None if wt_cpu is None else wt_cpu.data_ptr ()

    res = torch.empty (1, M, N, dtype = src.dtype, device = src.device)
    if cpu_N is None: cpu_N = N // 2
    libsplit_matmuls.split_matmul (
        M, K, N,
        src.data_ptr (),
        wt.data_ptr (),
        None,
        res.data_ptr (),
        cpu_N, flags, wt_cpu_ptr
    )
    return res

class SplitLinear (torch.nn.Module):
    "Replacement for torch.nn.Linear, using split_linear"
    def __init__ (self, inner, cpu_N, flags):
        super ().__init__ ()
        self.inner = inner
        self.cpu_N = cpu_N
        self.flags = flags

    def forward (self, x):
        return split_linear (
            x, self.inner.weight, cpu_N = self.cpu_N, flags = self.flags)

    def extra_repr (self):
        return f"cpu_N={self.cpu_N}, flags={self.flags},"

def replace_linear (model, targets):
    """
    Replace specific Linear layers in a model with SplitLinear layers.
    `targets` is a mapping from (weight matrix size) -> (cpu_N).
    """
    for name, mod in model.named_modules ():
        if not isinstance (mod, torch.nn.Linear):
            continue

        shape = tuple (mod.weight.shape)[::-1]
        if shape not in targets:
            continue
        cpu_N = targets [shape]

        model.set_submodule (name, SplitLinear (mod, cpu_N, b""))

    with open ("model-hacked.txt", "w") as f:
        print (model, file = f)

class SplitQwen3MLP (torch.nn.Module):
    """
    Replacement for Qwen3MLP from HF Transformers, using `split_qwen3_mlp_fwd`
    """
    def __init__ (self, inner, cpuI, flags):
        super ().__init__ ()
        self.inner = inner
        self.cpuI = cpuI
        self.flags = flags

    def extra_repr (self):
        return f"cpuI={self.cpuI}, flags={self.flags},"

    def forward (self, x):
        _b, M, K = x.shape
        assert _b == 1

        if M > 1:
            print (f"WARNING: Got M>1 for SplitQwen3MLP; not splitting")
            return self.inner (x)

        m = self.inner
        H, I = m.hidden_size, m.intermediate_size

        res = torch.empty (1, 1, K, device = "xpu", dtype = torch.bfloat16)
        libsplit_matmuls.split_qwen3_mlp_fwd (
            H, I, self.cpuI,
            x.data_ptr (),
            m.gate_proj.weight.data_ptr (),
            m.up_proj.weight.data_ptr (),
            m.down_proj.weight.data_ptr (),
            res.data_ptr (), self.flags,
        )
        return res

empty_kernel = libsplit_matmuls.enqueue_empty_kernel

def benchmark_gemv (K, N, variant, reps, verify = False):
    src = torch.rand (1, K, **bf16_xpu)
    wt = torch.rand (N, K, **bf16_xpu)
    dst = torch.rand (1, N, **bf16_xpu)

    args = (
        K, N,
        src.data_ptr (), wt.data_ptr (), dst.data_ptr (),
        bytes (variant, encoding = "ASCII"), reps
    )
    if verify:
        libsplit_matmuls.benchmark_gemv (*args)
        ref = torch.nn.functional.linear (src, wt)
        err = (ref - dst).flatten ().abs ().max ()
        print (ref, "\n", dst)
        print (f"Max err: {err:.4f}")

    return libsplit_matmuls.benchmark_gemv (*args)
