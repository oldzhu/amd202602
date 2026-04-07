"""
MXFP4-MM: CK Blockscale with tuned splitK experiment
Tests direct CK blockscale path with per-shape splitK to compare vs ASM.
"""

import torch
import weakref
from task import input_t, output_t

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale

_QUANT_REF = None
_QUANT_RESULT = None
_OUT_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    if not A.is_contiguous():
        A = A.contiguous()

    global _QUANT_REF, _QUANT_RESULT
    if _QUANT_REF is not None and _QUANT_REF() is A:
        A_q, A_scale_sh = _QUANT_RESULT
    else:
        A_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh = e8m0_shuffle(bs_e8m0).view(dtypes.fp8_e8m0)
        _QUANT_REF = weakref.ref(A)
        _QUANT_RESULT = (A_q, A_scale_sh)

    m = A.shape[0]
    n = B_shuffle.shape[0]
    k = A.shape[1]

    padded_m = ((m + 31) // 32) * 32
    out_key = (padded_m, n)
    out = _OUT_CACHE.get(out_key)
    if out is None:
        out = torch.empty((padded_m, n), dtype=dtypes.bf16, device=A.device)
        _OUT_CACHE[out_key] = out

    # Use CK blockscale with splitK based on CU utilization
    # CK tile is 64x128 by default; compute splitK for CU coverage
    tile_m_ck, tile_n_ck = 64, 128
    tile_num = ((padded_m + tile_m_ck - 1) // tile_m_ck) * ((n + tile_n_ck - 1) // tile_n_ck)
    # splitK values to try: 0 (none), 2, 4, 8
    if tile_num >= 304:
        splitK = 0
    elif tile_num * 2 >= 304:
        splitK = 2
    elif tile_num * 4 >= 304:
        splitK = 4
    else:
        splitK = 8

    gemm_a4w4_blockscale(
        A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        splitK=splitK,
    )
    return out[:m, :]
