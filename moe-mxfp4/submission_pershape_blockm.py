"""
MoE-MXFP4: Per-shape optimal block_size_M.
Uses heuristic-derived optimal block_m for each (E, d, M) combination.
No CSV override - just the right block_size_M parameter.
"""

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Optimal block_size_M based on CU-rounding heuristic analysis:
# Key: (E_total, d_expert, M) -> block_size_M
# E=257, d=256: always 32
# E=33, d=512: 32 for M<=16, 64 for M>=128
# E=33, d=2048: 128 for M>=512

def _get_block_m(E, d_expert, M):
    if E > 64:  # E=257 case (many experts, small d)
        return 32
    elif d_expert >= 2048:  # E=33, d=2048 (EP-on)
        if M >= 512:
            return 128
        elif M >= 128:
            return 64
        return 32
    else:  # E=33, d=512 (TP=4)
        if M >= 128:
            return 64
        return 32


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _, _, _, _,
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
    E = gate_up_weight_shuffled.shape[0]
    d_expert = config["d_expert"]
    block_size_m = _get_block_m(E, d_expert, M)

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
