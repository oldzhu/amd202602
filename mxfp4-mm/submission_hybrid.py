"""
MXFP4-MM Hybrid: a16wfp4 fast-path → BF16+ASM fallback.

Strategy:
  Tier 0 (warmup probe): gemm_a16wfp4 single-kernel on fp4-capable runners.
  Tier 1 (fallback):
    - K ≤ 1024  → torch.mm  (single BLAS call, lower launch overhead)
    - K > 1024  → ASM MXFP4 (3 launches, but 4x less memory traffic for B)
"""

import sys
import torch
from task import input_t, output_t

# ---- Register Triton dtype placeholders (prevent KeyError during JIT) ------
try:
    from triton._utils import type_canonicalisation_dict as _tcd
    for _name in ('float4_e2m1fn_x2', 'float8_e8m0fnu'):
        if _name not in _tcd:
            _tcd[_name] = 'u8'
except (ImportError, AttributeError):
    pass

import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm

_A16WFP4 = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    _A16WFP4 = gemm_a16wfp4
except Exception:
    pass

# ---- Constants & caches ---------------------------------------------------
_buf: dict = {}
_strategy = 0   # 0 = try a16wfp4, 1 = hybrid BF16+ASM

_ASM_KERNEL = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
_TILE_M, _TILE_N, _TILE_K = 32, 128, 256
_CU_NUM = 304
_SK: dict = {}


def _splitk(m: int, n: int, k: int) -> int:
    v = _SK.get((m, n, k))
    if v is not None:
        return v
    pm = ((m + _TILE_M - 1) // _TILE_M) * _TILE_M
    tiles = (pm // _TILE_M) * ((n + _TILE_N - 1) // _TILE_N)
    cu = _CU_NUM / tiles
    sk = 0
    while cu >= pow(2, sk + 1) and (pow(2, sk + 1) * _TILE_K) < 2 * k:
        sk += 1
    _SK[(m, n, k)] = min(sk, 4)
    return _SK[(m, n, k)]


def _unshuffle_e8m0(scale_sh: torch.Tensor) -> torch.Tensor:
    """Reverse e8m0_shuffle: permute(0,3,5,2,4,1) → inverse permute(0,5,3,1,4,2)."""
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    return (s.view(sm // 32, sn // 8, 4, 16, 2, 2)
             .permute(0, 5, 3, 1, 4, 2)
             .contiguous()
             .view(sm, sn))


def custom_kernel(data: input_t) -> output_t:
    global _strategy
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # ---- Tier 0: a16wfp4 single-kernel (fp4-capable runners) ----
    if _strategy == 0:
        if _A16WFP4 is None:
            _strategy = 1
        else:
            try:
                w_scales = _unshuffle_e8m0(B_scale_sh)[:n, :k // 32]
                key = (m, n)
                out = _buf.get(key)
                if out is None:
                    out = torch.empty(key, dtype=torch.bfloat16, device=A.device)
                    _buf[key] = out
                return _A16WFP4(
                    x=A, w=B_q,
                    w_scales=w_scales.view(dtypes.fp8_e8m0),
                    dtype=torch.bfloat16, y=out,
                )
            except Exception as e:
                print(f"[a16wfp4 fail] {e}", file=sys.stderr)
                _strategy = 1

    # ---- Tier 1: Hybrid BF16 + ASM ----
    if k <= 1024:
        # Small K → single BLAS torch.mm (launch overhead dominates)
        key = ('mm', m, n)
        out = _buf.get(key)
        if out is None:
            out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
            _buf[key] = out
        torch.mm(A, B.t(), out=out)
        return out

    # Large K → ASM MXFP4 (memory traffic dominates, 4x compression wins)
    if not A.is_contiguous():
        A = A.contiguous()
    A_fp4, bs = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_sc = e8m0_shuffle(bs).view(dtypes.fp8_e8m0)
    pm = ((m + 31) // 32) * 32
    key = ('asm', pm, n)
    out = _buf.get(key)
    if out is None:
        out = torch.empty((pm, n), dtype=dtypes.bf16, device=A.device)
        _buf[key] = out
    gemm_a4w4_asm(
        A_q, B_shuffle, A_sc, B_scale_sh, out,
        _ASM_KERNEL, bpreshuffle=True, log2_k_split=_splitk(m, n, k),
    )
    return out[:m, :]
