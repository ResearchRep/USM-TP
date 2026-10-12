#!/bin/env python3

#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

### Build Configuration Script ###
# Run this after activating a venv that satisfies `requirements.txt`
# You will also need a custom oneDNN build.
# Consult the README for more details.

import ninja_syntax
from argparse import ArgumentParser as AP
import importlib.util
import os
import sys

ap = AP ()
ap.add_argument ("--dnnl-prefix", required = True)
ap.add_argument ("--oneapi-prefix", default = "/opt/intel/oneapi")
args = ap.parse_args ()

# Find the Python venv, so we can use SYCL/OMP/etc runtime libs from there
venv_dir = sys.prefix
VENV_DIR = f"{venv_dir}"
assert os.path.exists (f"{VENV_DIR}")
print (f"Using libs from venv: {VENV_DIR}")

# Find the PyTorch installation, so we can use its libtorch_xpu.so
torch_spec = importlib.util.find_spec ("torch")
assert torch_spec is not None
torch_dir = torch_spec.origin.rsplit ("/", maxsplit = 1)[0]
TORCH_DIR = f"{torch_dir}"
assert os.path.exists (f"{TORCH_DIR}/lib/libtorch_xpu.so")
print (f"Using Torch libs from: {TORCH_DIR}")

# Use the user's oneDNN build
DNNL_DIR = f"{args.dnnl_prefix}"
assert os.path.exists (f"{DNNL_DIR}/lib/libdnnl-omp.so")
print (f"Using OneDNN libs from: {DNNL_DIR}")

# Use this specific compiler version
COMPILER = f"{args.oneapi_prefix}/2025.3/bin/icpx"
assert os.path.exists (COMPILER)
print (f"Using compiler: {COMPILER}")

# Construct a link line for a library
def mk_link_line (name, prefix):
    return f"-L{prefix}/lib -I{prefix}/include -l{name} -Wl,-rpath={prefix}/lib "

# Build rules
with open ("build.ninja", "w") as f:
    w = ninja_syntax.Writer (f)
    w.variable ("icpx", COMPILER)
    w.variable ("torch_d", TORCH_DIR)
    w.variable ("dnnl_d", DNNL_DIR)
    w.variable ("venv", VENV_DIR)
    w.newline ()

    w.rule (
        "sycl_lib",
        "$icpx -fsycl -fPIC -shared $flags $in -o $out"
    )
    w.build (
        "libsplit_matmuls.so",
        "sycl_lib",
        ["split_matmuls.cpp"],
        variables = {
            "flags": "-ltbb "
                   + mk_link_line ("torch_xpu", "$torch_d")
                   + mk_link_line ("dnnl-omp", "$dnnl_d")
        }
    )
    w.newline ()

    w.rule (
        "plaintext",
        "echo '$content' > $out"
    )
    w.build (
        ".libpath",
        "plaintext",
        variables = {
            "content": r"export LD_LIBRARY_PATH=$venv/lib"
        }
    )
    w.newline ()

    w.rule (
        "cc",
        "cc $in -o $out $flags"
    )
    w.build (
        "_force_usm.so",
        "cc",
        "force_usm.c",
        variables = {"flags": "-lze_loader -shared -fPIC"}
    )

    w.default ([".libpath", "libsplit_matmuls.so", "_force_usm.so"])
