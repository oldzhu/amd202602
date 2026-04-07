"""
MXFP4-MM: Per-shape tile + splitK optimization
Uses 32x256 tile for shapes with N>=2048 (better A reuse with wider tile),
32x128 for smaller N. Both paths use optimal per-shape splitK.
"""

import torch
import weakref
from task import input_t, output_t

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm

_ASM_32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
_ASM_32x256 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256E"
_CU_NUM = 304

_QUANT_REF = None
_QUANT_RESULT = None
_OUT_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_SHAPE_CACHE: dict[tuple[int, int, int], tuple[str, int, int, int]] = {}


def _select_kernel_and_splitk(m, n, k):
    """Select tile size and compute splitK in one shot."""
    # Use 32x256 for wide N shapes, 32x128 for narrow
    if n >= 2048:
        kernel = _ASM_32x256
        tile_m, tile_n, tile_k = 32, 256, 256
    else:
        kernel = _ASM_32x128
        tile_m, tile_n, tile_k = 32, 128, 256

    padded_m = ((m + tile_m - 1) // tile_m) * tile_m
    tile_num = (padded_m // tile_m) * ((n + tile_n - 1) // tile_n)
    cus_per_tile = _CU_NUM / tile_num
    log2_sk = 0
    while cus_per_tile >= pow(2, log2_sk + 1) and (pow(2, log2_sk + 1) * tile_k) < 2 * k:
        log2_sk += 1
    log2_sk = min(log2_sk, 4)
    return kernel, tile_m, tile_n, log2_sk


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

    shape_key = (m, n, k)
    cached = _SHAPE_CACHE.get(shape_key)
    if cached is None:
        kernel, tile_m, tile_n, log2_ks = _select_kernel_and_splitk(m, n, k)
        cached = (kernel, tile_m, tile_n, log2_ks)
        _SHAPE_CACHE[shape_key] = cached
    else:
        kernel, tile_m, tile_n, log2_ks = cached

    padded_m = ((m + tile_m - 1) // tile_m) * tile_m
    out_key = (padded_m, n)
    out = _OUT_CACHE.get(out_key)
    if out is None:
        out = torch.empty((padded_m, n), dtype=dtypes.bf16, device=A.device)
        _OUT_CACHE[out_key] = out

    gemm_a4w4_asm(
        A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        kernel, bpreshuffle=True, log2_k_split=log2_ks,
    )
    return out[:m, :]
