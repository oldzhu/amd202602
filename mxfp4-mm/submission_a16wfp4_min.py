"""
MXFP4-MM: Ultra-minimal a16wfp4 — single Triton kernel, zero overhead.
"""

import torch
from task import input_t, output_t

import aiter  # noqa: F401 - registers custom dtypes (fp4x2, e8m0)
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4


def custom_kernel(data: input_t) -> output_t:
    A, B, Bs, _, _ = data
    return gemm_a16wfp4(x=A, w=B, w_scales=Bs, dtype=torch.bfloat16)
