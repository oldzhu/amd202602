"""
MXFP4-MM: Fused BF16→MXFP4 quant + GEMM via gemm_a16wfp4
Eliminates separate quant kernel + e8m0_shuffle + MXFP4 allocation.
Uses Triton kernel instead of ASM, so kernel may be slower but total path may be faster.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes

# Try preshuffle variant first (weights are pre-shuffled in task input)
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    _USE_PRESHUFFLE = True
except ImportError:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    _USE_PRESHUFFLE = False

_OUT_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    if not A.is_contiguous():
        A = A.contiguous()

    m = A.shape[0]
    n = B_shuffle.shape[0]

    out_key = (m, n)
    out = _OUT_CACHE.get(out_key)
    if out is None:
        out = torch.empty((m, n), dtype=dtypes.bf16, device=A.device)
        _OUT_CACHE[out_key] = out

    if _USE_PRESHUFFLE:
        result = gemm_a16wfp4_preshuffle(
            A, B_shuffle, B_scale_sh,
            dtype=dtypes.bf16,
            y=out,
        )
    else:
        result = gemm_a16wfp4(
            A, B_shuffle, B_scale_sh,
            dtype=dtypes.bf16,
            y=out,
        )

    return result
