"""
MXFP4-MM: Robust gemm_a16wfp4 with Triton dtype fix + ASM fallback.

Registers fp4/e8m0 dtypes with Triton's canonicalization dict to fix
KeyError 'float4_e2m1fn_x2' on runners with older Triton builds.
Pre-allocates output buffer y per shape.
Falls back to ASM path if a16wfp4 kernel fails.
"""

import sys
import torch
from task import input_t, output_t

import aiter  # noqa: F401
from aiter import dtypes

# ---- Fix Triton dtype canonicalization for FP4/E8M0 ----
try:
    from triton._utils import type_canonicalisation_dict as _tcd
    # Dump known types for diagnostics
    print(f"[diag] Triton known dtypes: {sorted(_tcd.keys())}", file=sys.stderr)

    # Try to find correct fp4 canonical name from existing dict
    _fp4_candidates = [v for k, v in _tcd.items() if 'fp4' in k.lower() or 'e2m1' in k.lower()]
    _fp8_candidates = [v for k, v in _tcd.items() if 'e8m0' in k.lower()]

    if 'float4_e2m1fn_x2' not in _tcd:
        if _fp4_candidates:
            _tcd['float4_e2m1fn_x2'] = _fp4_candidates[0]
            print(f"[diag] Registered float4_e2m1fn_x2 -> {_fp4_candidates[0]}", file=sys.stderr)
        else:
            # Last resort: map to u8 (same byte size, may affect MFMA instruction selection)
            _tcd['float4_e2m1fn_x2'] = 'u8'
            print("[diag] Registered float4_e2m1fn_x2 -> u8 (fallback)", file=sys.stderr)
    else:
        print(f"[diag] float4_e2m1fn_x2 already -> {_tcd['float4_e2m1fn_x2']}", file=sys.stderr)

    if 'float8_e8m0fnu' not in _tcd:
        if _fp8_candidates:
            _tcd['float8_e8m0fnu'] = _fp8_candidates[0]
        else:
            _tcd['float8_e8m0fnu'] = 'u8'
        print(f"[diag] Registered float8_e8m0fnu -> {_tcd.get('float8_e8m0fnu')}", file=sys.stderr)
except (ImportError, AttributeError) as e:
    print(f"[diag] Could not access Triton type dict: {e}", file=sys.stderr)

# ---- Import kernels ----
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm

_A16WFP4_FN = None
try:
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    _A16WFP4_FN = gemm_a16wfp4
    print("[a16wfp4] loaded OK", file=sys.stderr)
except ImportError:
    print("[a16wfp4] NOT available", file=sys.stderr)

_A16WFP4_ALIVE = True
_Y: dict[tuple[int, int], torch.Tensor] = {}

# ASM config
_ASM_KERNEL = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
_TILE_M, _TILE_N, _TILE_K = 32, 128, 256
_CU_NUM = 304
_SK: dict[tuple[int, int, int], int] = {}


def _splitk(m, n, k):
    key = (m, n, k)
    v = _SK.get(key)
    if v is not None:
        return v
    pm = ((m + _TILE_M - 1) // _TILE_M) * _TILE_M
    tiles = (pm // _TILE_M) * ((n + _TILE_N - 1) // _TILE_N)
    cu = _CU_NUM / tiles
    sk = 0
    while cu >= pow(2, sk + 1) and (pow(2, sk + 1) * _TILE_K) < 2 * k:
        sk += 1
    sk = min(sk, 4)
    _SK[key] = sk
    return sk


def custom_kernel(data: input_t) -> output_t:
    global _A16WFP4_ALIVE
    A, B, Bs, B_shuffle, B_scale_sh = data

    # Try a16wfp4 (single Triton kernel launch)
    if _A16WFP4_FN is not None and _A16WFP4_ALIVE:
        try:
            m = A.shape[0]
            n = B.shape[0]
            key = (m, n)
            y = _Y.get(key)
            if y is None:
                y = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
                _Y[key] = y
            return _A16WFP4_FN(x=A, w=B, w_scales=Bs, dtype=torch.bfloat16, y=y)
        except Exception as e:
            _A16WFP4_ALIVE = False
            print(f"[a16wfp4] FAILED: {e}", file=sys.stderr)

    # Fallback: ASM path (3 kernel launches)
    if not A.is_contiguous():
        A = A.contiguous()
    A_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(bs_e8m0).view(dtypes.fp8_e8m0)
    m, k = A.shape
    n = B_shuffle.shape[0]
    pm = ((m + 31) // 32) * 32
    okey = (pm, n)
    out = _Y.get(okey)
    if out is None:
        out = torch.empty((pm, n), dtype=dtypes.bf16, device=A.device)
        _Y[okey] = out
    gemm_a4w4_asm(A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        _ASM_KERNEL, bpreshuffle=True, log2_k_split=_splitk(m, n, k))
    return out[:m, :]
