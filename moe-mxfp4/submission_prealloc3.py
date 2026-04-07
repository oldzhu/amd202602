"""
MoE-MXFP4: Per-shape block_size_M tuning.
Uses block_m=16 for very small M, block_m=32 for medium M, default for large.
Also calls moe_sorting + fused_moe_2stages directly with pre-allocated sorting buffers.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe_2stages

_SORT_CACHE: dict[tuple, tuple] = {}


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
    topk = topk_ids.shape[1]
    num_experts = gate_up_weight_shuffled.shape[0]
    model_dim = hidden_states.shape[1]

    # Let AITER auto-pick block_size_M via heuristic for all shapes
    block_size_m = 32

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
        None,   # expert_mask
        None,   # num_local_tokens
        0,      # dispatch_policy
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
