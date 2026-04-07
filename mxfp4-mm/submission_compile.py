"""
MXFP4-MM: torch.compile variant for reduced dispatch overhead.

Uses torch.compile(mode='reduce-overhead') which may use CUDAGraph internally
to amortize kernel launch costs.
"""

import torch
from task import input_t, output_t

import aiter  # noqa: F401
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

_Y = {}


@torch.compile(mode="reduce-overhead")
def _gemm(A, B, Bs, y):
    return gemm_a16wfp4(x=A, w=B, w_scales=Bs, dtype=torch.bfloat16, y=y)


def custom_kernel(data: input_t) -> output_t:
    A, B, Bs, _, _ = data
    m = A.shape[0]
    n = B.shape[0]

    key = (m, n)
    y = _Y.get(key)
    if y is None:
        y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
        _Y[key] = y

    return _gemm(A, B, Bs, y)
