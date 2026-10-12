#!/bin/env python3

#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
ap = ArgumentParser ()
ap.add_argument ("--model", default = "./Qwen3-4B")
ap.add_argument ("--ntoks", type = int, default = 5)
ap.add_argument ("--device", default = "xpu")
ap.add_argument ("--hack-linear", action = "store_true")
ap.add_argument ("--hack-mlp", action = "store_true")
args = ap.parse_args ()
print (args)

from time import time
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedConfig,
    CompileConfig,
)

import split_matmuls
from split_matmuls import replace_linear
import force_usm

## Utilities
def timer (F):
    "Measure the execution time of `F`"
    tot = time ()
    res = F () 
    tot = time () - tot
    return tot, res

def profile (enable: bool):
    """
    Enable or disable profiling.
    Use with e.g. unitrace's conditional collection feature.
    """
    os.environ ["PTI_ENABLE_COLLECTION"] = (enable and "1") or "0"

def main ():
    # Model load
    prompt = "Hi bro,"
    config = AutoConfig.from_pretrained (args.model)
    model = AutoModelForCausalLM.from_pretrained (
        # Note: we don't use `device_map` here, since that will cause a
        # call to `torch.xpu.get_mem_info ()`; this isn't supported
        # on our platform, and thus will result in an error.
        # https://github.com/huggingface/transformers/blob/cc832f9055ba11c8c55f918ab4bda9472b910d48/src/transformers/modeling_utils.py#L4949
        args.model,
        torch_dtype = "auto",
    )
    model = model.to (args.device)
    tokzer = AutoTokenizer.from_pretrained (args.model)
    inp = tokzer.encode (prompt, return_tensors = "pt").to (model.device)

    linear_repl = {
        # (in, out) -> cpu_N
    }
    if args.hack_linear:
        replace_linear (model, linear_repl)

    if args.hack_mlp:
        for n, m in model.named_modules ():
            if n.endswith (".mlp"):
                print (n)
                cpuI = int (0.2 * config.intermediate_size)
                repl = split_matmuls.SplitQwen3MLP (m, cpuI, b"")
                model.set_submodule (n, repl)
        print (model)

    generate_cfg = dict (
        use_cache = True,
        cache_implementation = "static",
        compile_config = CompileConfig (
            fullgraph = True,
            mode = "reduce-overhead",
        ),
        disable_compile = True,
    )
    def gen (n):
        res = model.generate (inp, **generate_cfg, max_new_tokens = n)
        torch.xpu.synchronize ()
        return res

    # Warmup
    timer (lambda: gen (args.ntoks))

    # Run
    profile (True)
    t, res = timer (lambda: gen (args.ntoks))
    profile (False)
    ans = tokzer.batch_decode (res)[0]
    print (f"{ans}\nThroughput: {args.ntoks / t:.2f} tok/s")

with torch.inference_mode ():
    main ()
