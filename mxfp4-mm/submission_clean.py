"""
MXFP4-MM: Optimized Implementation
AMD GPU MODE Hackathon - Phase 1

Optimizations:
- Make both A and B contiguous for optimal GEMM performance
- Use e8m0_shuffle for activation scales (required for gemm_a4w4)
- Match reference implementation exactly
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle


def _quant_mxfp4(x: torch.Tensor, shuffle: bool = True):
    """
    Quantize a bf16 tensor to MXFP4 format.

    Args:
        x: [M, K] bfloat16 tensor (K must be divisible by 32)
        shuffle: Whether to shuffle scales for gemm_a4w4 compatibility

    Returns:
        (fp4_data, scale_e8m0) - quantized tensor and scales
    """
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
    if shuffle:
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
    return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)


def custom_kernel(data: input_t) -> output_t:
    """
    MXFP4 Matrix Multiplication: C = A @ B.T

    Uses AITER's gemm_a4w4 for 4-bit quantized matmul.
    """
    A, _, _, B_shuffle, B_scale_sh = data

    if not A.is_contiguous():
        A = A.contiguous()

    A_q, A_scale_sh = _quant_mxfp4(A, shuffle=True)

    output = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )

    return output
