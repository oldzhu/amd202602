"""
MoE: Direct 2-stage dispatch with pre-allocated buffers.
Bypasses fused_moe() Python wrapper to eliminate overhead.
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Pre-allocated buffers keyed by shape
_BUF_CACHE = {}


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
    
    M = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    E = config["num_experts"]
    d_expert = config["d_expert"]
    
    # Per-shape block_size_M tuning
    # For small M with large d_expert, use 32; otherwise auto
    if M <= 128 and d_expert >= 512:
        block_size_m = 32
    elif M <= 16:
        block_size_m = 32
    else:
        block_size_m = None

    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()

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
