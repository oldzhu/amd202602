"""
MoE-MXFP4: Enable CKTile via is_shuffled + fused_moe wrapper.
Sets is_shuffled=True on weights to trigger CKTile dispatch for shapes
where ksplit > 1, which skips FP4 activation quantization.
Uses fused_moe wrapper for proper config/heuristic selection.
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_SHUFFLED_SET = False


def custom_kernel(data: input_t) -> output_t:
    global _SHUFFLED_SET
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

    # Enable CKTile dispatch path — weights are pre-shuffled by the task
    if not _SHUFFLED_SET:
        gate_up_weight_shuffled.is_shuffled = True
        down_weight_shuffled.is_shuffled = True
        _SHUFFLED_SET = True

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
