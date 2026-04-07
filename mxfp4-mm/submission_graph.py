"""
MXFP4-MM: CUDAGraph (hipGraph) approach to minimize kernel launch overhead.

Profile analysis showed 3 kernel launches (quant, e8m0_shuffle, GEMM ASM)
costing ~20μs in launch overhead alone, with GPU compute < 2μs.
By capturing all 3 kernels into a single graph, replay should cost ~1-2μs
instead of ~20μs per call.
"""

import sys
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

_GRAPH_CACHE = {}
_SPLITK_CACHE = {}
_OUT_FALLBACK = {}


def _compute_log2_splitk(m: int, n: int, k: int) -> int:
    key = (m, n, k)
    cached = _SPLITK_CACHE.get(key)
    if cached is not None:
        return cached
    padded_m = ((m + _TILE_M - 1) // _TILE_M) * _TILE_M
    tile_num = (padded_m // _TILE_M) * ((n + _TILE_N - 1) // _TILE_N)
    cus_per_tile = _CU_NUM / tile_num
    log2_sk = 0
    while cus_per_tile >= pow(2, log2_sk + 1) and (pow(2, log2_sk + 1) * _TILE_K) < 2 * k:
        log2_sk += 1
    result = min(log2_sk, 4)
    _SPLITK_CACHE[key] = result
    return result


def _pipeline(A, B_shuffle, B_scale_sh, out, log2_ks):
    """Full 3-kernel pipeline: quant → shuffle → GEMM."""
    A_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(bs_e8m0).view(dtypes.fp8_e8m0)
    gemm_a4w4_asm(
        A_q, B_shuffle, A_scale_sh, B_scale_sh, out,
        _ASM_KERNEL, bpreshuffle=True, log2_k_split=log2_ks,
    )


def custom_kernel(data: input_t) -> output_t:
    A, _, _, B_shuffle, B_scale_sh = data

    if not A.is_contiguous():
        A = A.contiguous()

    m, k = A.shape
    n = B_shuffle.shape[0]
    padded_m = ((m + 31) // 32) * 32

    # Key by tensor identity + shape (data_ptr can be reused after dealloc)
    cache_key = (A.data_ptr(), B_shuffle.data_ptr(), m, n, k)

    if cache_key in _GRAPH_CACHE:
        entry = _GRAPH_CACHE[cache_key]
        if entry is not None:
            # Fast path: replay captured graph (~1-2μs vs ~20μs)
            entry['graph'].replay()
            return entry['out'][:m, :]
        # Graph failed: direct fallback
        log2_ks = _compute_log2_splitk(m, n, k)
        fk = (padded_m, n)
        out = _OUT_FALLBACK.get(fk)
        if out is None:
            out = torch.empty((padded_m, n), dtype=dtypes.bf16, device=A.device)
            _OUT_FALLBACK[fk] = out
        _pipeline(A, B_shuffle, B_scale_sh, out, log2_ks)
        return out[:m, :]

    # First call for this tensor set: warm-up + capture
    log2_ks = _compute_log2_splitk(m, n, k)
    out = torch.empty((padded_m, n), dtype=dtypes.bf16, device=A.device)

    # Warm-up: also produces correct result for return
    _pipeline(A, B_shuffle, B_scale_sh, out, log2_ks)

    # Attempt graph capture
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _pipeline(A, B_shuffle, B_scale_sh, out, log2_ks)
        _GRAPH_CACHE[cache_key] = {'graph': g, 'out': out}
        print(f"[graph] captured M={m} N={n} K={k}", file=sys.stderr)
    except Exception as e:
        _GRAPH_CACHE[cache_key] = None
        print(f"[graph] capture FAILED M={m} N={n} K={k}: {e}", file=sys.stderr)

    return out[:m, :]
