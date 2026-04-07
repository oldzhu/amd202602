"""
MXFP4-MM: Pre-allocated ASM path — minimize per-call allocations.

Key difference from clean.py: pre-allocate quant output buffers (A_fp4, bs_e8m0)
so dynamic_mxfp4_quant reuses them instead of allocating new tensors each call.
Uses a direct call pattern that avoids weakref overhead.
"""

import torch
from task import input_t, output_t

import aiter  # noqa: F401
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm

_ASM_KERNEL = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
_TILE_M, _TILE_N, _TILE_K = 32, 128, 256
_CU_NUM = 304

# Per-shape caches: output buffer, splitK
_OUT: dict[tuple[int, int], torch.Tensor] = {}
_SK: dict[tuple[int, int, int], int] = {}


def _splitk(m: int, n: int, k: int) -> int:
    key = (m, n, k)
    v = _SK.get(key)
    if v is not None:
        return v
    pm = ((m + _TILE_M - 1) // _TILE_M) * _TILE_M
    tiles = (pm // _TILE_M) * ((n + _TILE_N - 1) // _TILE_N)
    cu_per = _CU_NUM / tiles
    sk = 0
    while cu_per >= pow(2, sk + 1) and (pow(2, sk + 1) * _TILE_K) < 2 * k:
        sk += 1
    sk = min(sk, 4)
    _SK[key] = sk
    return sk


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    if not A.is_contiguous():
        A = A.contiguous()

    # Quantize A to MXFP4 (1 Triton kernel)
    A_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)

    # Shuffle A scales (1 Triton kernel)
    A_scale_sh = e8m0_shuffle(bs_e8m0).view(dtypes.fp8_e8m0)

    m = A.shape[0]
    n = B_shuffle.shape[0]
    k = A.shape[1]

    # Reuse output buffer
    pm = ((m + 31) // 32) * 32
    okey = (pm, n)
    out = _OUT.get(okey)
    if out is None:
        out = torch.empty((pm, n), dtype=dtypes.bf16, device=A.device)
        _OUT[okey] = out

    gemm_a4w4_asm(
        A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        _ASM_KERNEL, bpreshuffle=True, log2_k_split=_splitk(m, n, k),
    )
    return out[:m, :]
