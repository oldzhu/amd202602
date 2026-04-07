"""
MoE: Call torch.ops.aiter.fused_moe_ directly (C++ op) to skip Python wrapper overhead.
This bypasses the fused_moe() -> fused_moe_() Python path.
"""

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType


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
    block_size_m = 32 if M <= 256 else None

    # Call the C++ registered op directly
    return torch.ops.aiter.fused_moe_(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        None,                               # expert_mask
        int(ActivationType.Silu),           # activation
        int(QuantType.per_1x32),            # quant_type
        False,                               # doweight_stage1
        gate_up_weight_scale_shuffled,       # w1_scale
        down_weight_scale_shuffled,          # w2_scale
        None,                               # a1_scale
        None,                               # a2_scale
        block_size_m if block_size_m else 0, # block_size_M (0 for None)
        0,                                   # num_local_tokens
        0,                                   # moe_sorting_dispatch_policy
        None,                               # dtype
        hidden_pad,                          # hidden_pad
        intermediate_pad,                    # intermediate_pad
        None,                               # bias1
        None,                               # bias2
        0,                                   # splitk
    )
