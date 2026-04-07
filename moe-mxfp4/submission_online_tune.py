"""
MoE-MXFP4: Online Tuning Experiment
AMD GPU MODE Hackathon - Phase 1

Uses AITER's fused_moe kernel with AITER_ONLINE_TUNE=1 and AITER_BYPASS_TUNE_CONFIG=1.
"""

import os
os.environ["AITER_ONLINE_TUNE"] = "1"
os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def custom_kernel(data: input_t) -> output_t:
    """
    MoE Layer with MXFP4 quantized weights using AITER fused_moe.
    
    Input:
        hidden_states:                  [M, d_hidden] bf16
        gate_up_weight_shuffled:        [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
        down_weight_shuffled:           [E, d_hidden_pad, d_expert_pad//2] fp4x2
        gate_up_weight_scale_shuffled:  [padded, flat] e8m0
        down_weight_scale_shuffled:     [padded, flat] e8m0
        topk_weights:                   [M, total_top_k] float32
        topk_ids:                       [M, total_top_k] int32
        config:                         dict
    
    Output:
        [M, d_hidden] bf16
    """
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

    block_size_m = 32 if topk_ids.shape[0] <= 128 and config["d_expert"] >= 512 else None

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
        activation=ActivationType.Silu,      # SwiGLU: silu(gate) * up
        quant_type=QuantType.per_1x32,       # MXFP4 block quantization
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        block_size_M=block_size_m,
    )
