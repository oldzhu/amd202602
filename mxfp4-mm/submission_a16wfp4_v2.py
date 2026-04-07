"""
MXFP4-MM: a16wfp4 with correct fp4 B_q tensor and shuffled scales.

Uses data[2]=B_q (fp4x2 [N,K//2]) as w and data[4]=B_scale_sh as w_scales.
Pre-allocates output buffer y.
"""

import torch
from task import input_t, output_t

import aiter  # noqa: F401 - registers custom dtypes
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

_Y: dict[tuple[int, int], torch.Tensor] = {}


def custom_kernel(data: input_t) -> output_t:
    A, _, B_q, _, B_scale_sh = data
    m = A.shape[0]
    n = B_q.shape[0]

    key = (m, n)
    y = _Y.get(key)
    if y is None:
        y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
        _Y[key] = y

    return gemm_a16wfp4(x=A, w=B_q, w_scales=B_scale_sh, dtype=torch.bfloat16, y=y)
