"""
MoE-MXFP4: Per-shape block_size_M tuning experiment
Tests block_size_M = 64 for larger batches, 32 for smallest.
Also sets AITER_KSPLIT=2 for potential splitK benefit on decode shapes.
"""

import os
os.environ["AITER_KSPLIT"] = "2"

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _, _, _, _,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = topk_ids.shape[0]
    E = config["n_routed_experts"] + config["n_shared_experts"]
    topk = topk_ids.shape[1]

    # Per-shape block_size_M: balance CU utilization
    if M <= 16:
        block_size_m = 32
    elif M <= 128:
        block_size_m = 64 if (M * topk) > E else 32
    else:
        block_size_m = 128

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
