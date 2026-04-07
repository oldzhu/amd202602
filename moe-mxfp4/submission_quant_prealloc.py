"""
MoE-MXFP4: Optimized with pre-alloc sorting + quant buffers + per-shape tuning.
Combines:
1. Pre-allocated sorting buffers
2. Monkey-patched fused_dynamic_mxfp4_quant_moe_sort for quant buffer reuse
3. Per-shape block_m tuning
4. k_split=2 for E=33 shapes via env var
"""

import os
import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe_2stages
import triton

# Try to import and monkey-patch the quant function
try:
    import aiter.ops.triton.quant.fused_mxfp4_quant as _fmq
    _HAS_QUANT_PATCH = True
except ImportError:
    _HAS_QUANT_PATCH = False

_SORT_CACHE = {}
_QUANT_FP4_CACHE = {}
_QUANT_SCALE_CACHE = {}


def _get_block_m(M, topk, num_experts):
    """Per-shape block_m: always use 32 (best for all tested shapes)."""
    return 32


def _get_sort_buffers(M, topk, num_experts, model_dim, block_size, device):
    key = (M, topk, num_experts, model_dim, block_size)
    cached = _SORT_CACHE.get(key)
    if cached is not None:
        return cached

    max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=torch.bfloat16, device=device)

    cached = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    _SORT_CACHE[key] = cached
    return cached


if _HAS_QUANT_PATCH:
    _orig_quant = _fmq.fused_dynamic_mxfp4_quant_moe_sort

    def _patched_quant(x, sorted_ids, num_valid_ids, token_num, topk,
                       block_size=32, scaling_mode="even"):
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

        if M <= 32:
            BLOCK_SIZE_Mx = 32
        else:
            BLOCK_SIZE_Mx = 128

        BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
        BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

        N_i = scaleN
        M_o, N_o = sorted_ids.shape[0], N_i

        scale_key = (M_o, N_o, BLOCK_SIZE_M, BLOCK_SIZE_N,
                     BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32)
        scale_cached = _QUANT_SCALE_CACHE.get(scale_key)
        if scale_cached is None:
            scale_cached = torch.empty(
                (triton.cdiv(M_o, BLOCK_SIZE_M),
                 triton.cdiv(N_o, BLOCK_SIZE_N),
                 BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32, 4),
                dtype=torch.uint8, device=x.device,
            )
            _QUANT_SCALE_CACHE[scale_key] = scale_cached
        blockscale_e8m0_sorted = scale_cached

        num_pid = (triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN +
                   triton.cdiv(M_o, BLOCK_SIZE_M) * triton.cdiv(N_i, BLOCK_SIZE_N))

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

    _fmq.fused_dynamic_mxfp4_quant_moe_sort = _patched_quant


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        _, _, _, _,
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
    topk = topk_ids.shape[1]
    num_experts = gate_up_weight_shuffled.shape[0]
    model_dim = hidden_states.shape[1]
    block_size_m = _get_block_m(M, topk, num_experts)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = \
        _get_sort_buffers(M, topk, num_experts, model_dim, block_size_m, hidden_states.device)

    aiter.moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_size_m,
        None, None, 0,
    )

    return fused_moe_2stages(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        True,  # isG1U1
        block_size_m,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=aiter_dtypes.fp4x2,
        q_dtype_w=gate_up_weight_shuffled.dtype,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )
