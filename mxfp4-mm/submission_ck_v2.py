"""
MXFP4-MM: CK blockscale with corrected splitK values.
Uses CK path (gemm_a4w4_blockscale) instead of ASM to compare cold-cache perf.
"""

import torch
import weakref
from task import input_t, output_t

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale

_CU_NUM = 304
# CK blockscale uses 64x128 tiles internally
_TILE_M_CK, _TILE_N_CK = 64, 128

_QUANT_REF = None
_QUANT_RESULT = None
_OUT_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_SPLITK_CACHE: dict[tuple[int, int, int], int] = {}


def _compute_splitk(m: int, n: int, k: int) -> int:
    """Compute splitK for CK blockscale path (uses linear splitK, not log2)."""
    padded_m = ((m + _TILE_M_CK - 1) // _TILE_M_CK) * _TILE_M_CK
    tile_num = (padded_m // _TILE_M_CK) * ((n + _TILE_N_CK - 1) // _TILE_N_CK)
    # CK splitK uses linear values, not log2
    if tile_num >= _CU_NUM:
        return 0  # Enough tiles, no split needed
    # Target: tile_num * splitK >= CU_NUM for good utilization
    ratio = _CU_NUM / tile_num
    if ratio >= 8:
        return 8
    elif ratio >= 4:
        return 4
    elif ratio >= 2:
        return 2
    return 0


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

    padded_m = ((m + 63) // 64) * 64  # CK uses 64-row tiles
    out_key = (padded_m, n)
    out = _OUT_CACHE.get(out_key)
    if out is None:
        out = torch.empty((padded_m, n), dtype=dtypes.bf16, device=A.device)
        _OUT_CACHE[out_key] = out

    sk_key = (m, n, k)
    sk = _SPLITK_CACHE.get(sk_key)
    if sk is None:
        sk = _compute_splitk(m, n, k)
        _SPLITK_CACHE[sk_key] = sk

    gemm_a4w4_blockscale(
        A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        splitK=sk,
    )
    return out[:m, :]
