"""
MoE-MXFP4: Pre-allocate quantization buffers via correct monkey-patching.

Key insight from top entries (guojun21 122μs, ruichenggu-wq 125μs):
Pre-allocate the fp4 output and blockscale tensors inside
fused_dynamic_mxfp4_quant_moe_sort to avoid per-call torch.empty overhead.

Critical: must patch aiter.fused_moe module (not the source module)
because fused_moe.py uses 'from ... import fused_dynamic_mxfp4_quant_moe_sort'
which creates a module-local binding.
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _afm
import aiter.ops.triton.quant.fused_mxfp4_quant as _fmq
import triton

# Caches for pre-allocated quant output buffers
_QUANT_FP4_CACHE = {}
_QUANT_SCALE_CACHE = {}


def _prealloc_fused_dynamic_mxfp4_quant_moe_sort(
    x, sorted_ids, num_valid_ids, token_num, topk, block_size=32, scaling_mode="even"
):
    """Drop-in replacement that reuses cached output buffers."""
    M, N = x.shape
    MXFP4_QUANT_BLOCK_SIZE = 32

    key = (M, N, sorted_ids.shape[0], block_size)

    fp4_cached = _QUANT_FP4_CACHE.get(key)
    if fp4_cached is None:
        fp4_cached = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
        _QUANT_FP4_CACHE[key] = fp4_cached
    x_fp4 = fp4_cached

    scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    scaleN = scaleN_valid

    BLOCK_SIZE_Mx = 32 if M <= 32 else 128

    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    N_i = scaleN
    M_o, N_o = sorted_ids.shape[0], N_i

    scale_key = (M_o, N_o, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32)
    scale_cached = _QUANT_SCALE_CACHE.get(scale_key)
    if scale_cached is None:
        scale_cached = torch.empty(
            (
                triton.cdiv(M_o, BLOCK_SIZE_M),
                triton.cdiv(N_o, BLOCK_SIZE_N),
                BLOCK_SIZE_N_u32,
                BLOCK_SIZE_M_u32,
                4,
            ),
            dtype=torch.uint8,
            device=x.device,
        )
        _QUANT_SCALE_CACHE[scale_key] = scale_cached
    blockscale_e8m0_sorted = scale_cached

    num_pid = triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN + triton.cdiv(
        M_o, BLOCK_SIZE_M
    ) * triton.cdiv(N_i, BLOCK_SIZE_N)

    _fmq._fused_dynamic_mxfp4_quant_moe_sort_kernel[(num_pid,)](
        x, x_fp4, sorted_ids, num_valid_ids, blockscale_e8m0_sorted,
        M, N, scaleN,
        *x.stride(), *x_fp4.stride(), *blockscale_e8m0_sorted.stride(),
        token_num=token_num, N_i=N_i,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        BLOCK_SIZE_Mx=BLOCK_SIZE_Mx,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )

    return (
        x_fp4.view(aiter_dtypes.fp4x2),
        blockscale_e8m0_sorted.view(aiter_dtypes.fp8_e8m0).view(-1, N_o),
    )


# Monkey-patch on the CORRECT module (aiter.fused_moe, not the source module)
_afm.fused_dynamic_mxfp4_quant_moe_sort = _prealloc_fused_dynamic_mxfp4_quant_moe_sort


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        _,
        _,
        _,
        _,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = topk_ids.shape[0]
    block_size_m = 32 if M <= 256 else None

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        block_size_M=block_size_m,
    )
