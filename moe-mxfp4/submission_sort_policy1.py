"""
MoE: Try moe_sorting_dispatch_policy=1 (single-phase) + per-shape block_size_M tuning.
Also test if removing .contiguous() calls helps (inputs may already be contiguous).
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


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
    E = config["n_experts"]
    d_expert = config["d_expert"]

    # Per-shape block_size_M tuning based on competition shapes
    if M <= 16:
        block_size_m = 32
    elif M <= 128 and d_expert >= 512:
        block_size_m = 32
    else:
        block_size_m = None  # Let AITER auto-select

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
        moe_sorting_dispatch_policy=1,  # single-phase sorting
    )
