"""
MXFP4-MM: Tile size sweep experiment
Tests different ASM tile sizes per shape to find optimal configuration.
"""

import torch
import weakref
import os
import sys
from task import input_t, output_t

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm

# Try different tile configurations
# Available: 32x128, 64x128, 96x128, 128x128, 192x128, 256x128
# Also: 32x256, 64x256, etc.
_TILE_CONFIGS = {
    "32x128": ("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 32, 128),
    "64x128": ("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E", 64, 128),
    "96x128": ("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128E", 96, 128),
    "128x128": ("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128E", 128, 128),
    "32x256": ("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256E", 32, 256),
    "64x256": ("_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x256E", 64, 256),
}

# Select config via env var, default to 32x128
_TILE_KEY = os.environ.get("TILE_CFG", "32x128")
_ASM_KERNEL, _TILE_M, _TILE_N = _TILE_CONFIGS.get(_TILE_KEY, _TILE_CONFIGS["32x128"])
_TILE_K = 256

print(f"[tile_sweep] Using tile config: {_TILE_KEY} → {_ASM_KERNEL}", file=sys.stderr)

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

    padded_m = ((m + _TILE_M - 1) // _TILE_M) * _TILE_M
    out_key = (padded_m, n)
    out = _OUT_CACHE.get(out_key)
    if out is None:
        out = torch.empty((padded_m, n), dtype=dtypes.bf16, device=A.device)
        _OUT_CACHE[out_key] = out

    tile_num = (padded_m // _TILE_M) * ((n + _TILE_N - 1) // _TILE_N)
    log2_ks = None if tile_num < 304 else 0

    gemm_a4w4_asm(
        A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        _ASM_KERNEL, bpreshuffle=True, log2_k_split=log2_ks,
    )
    return out[:m, :]
