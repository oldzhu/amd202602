"""
MXFP4-MM: Lean gemm_a16wfp4 with unshuffled scales.
Single-kernel MXFP4 GEMM with on-the-fly A quantization.
Uses uint8 view trick to avoid Triton dtype canonicalization issues.
"""

import torch
from task import input_t, output_t

# ---- Register Triton dtypes early ----
try:
    from triton._utils import type_canonicalisation_dict as _tcd
    for name in ['float4_e2m1fn_x2', 'float8_e8m0fnu']:
        if name not in _tcd:
            short = name.split('_')[0].replace('float', 'fp')
            cands = [v for k, v in _tcd.items() if short in k.lower()]
            _tcd[name] = cands[0] if cands else 'u8'
except (ImportError, AttributeError):
    pass

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

_buf: dict[tuple[int, int], torch.Tensor] = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    return (s.view(sm // 32, sn // 8, 4, 16, 2, 2)
             .permute(0, 5, 3, 1, 4, 2)
             .contiguous()
             .view(sm, sn))


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    w_scales = _unshuffle_e8m0(B_scale_sh)[:n, :k // 32]

    key = (m, n)
    out = _buf.get(key)
    if out is None:
        out = torch.empty(key, dtype=torch.bfloat16, device=A.device)
        _buf[key] = out

    return gemm_a16wfp4(
        x=A, w=B_q.view(torch.uint8),
        w_scales=w_scales,
        dtype=torch.bfloat16, y=out,
    )
