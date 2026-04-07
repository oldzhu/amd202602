"""
MoE-MXFP4: Per-shape tuned block_size_M and options.
Based on benchmark shapes analysis:
- TP=8 (E=257, d_expert=256): bs=16,128,512
- TP=4 (E=33, d_expert=512): bs=16,128,512
- EP (E=33, d_expert=2048): bs=512
"""

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
    E = config["n_routed_experts"] + config.get("n_shared_experts", 1)
    topk = topk_ids.shape[1]
    d_expert = config["d_expert"]

    # Estimated tokens per expert
    tokens_per_expert = M * topk / E

    # Per-shape block_size_M tuning
    if tokens_per_expert <= 1:
        # Very sparse: bs=16 with E=257 → ~0.56 tok/expert
        block_m = 32
    elif tokens_per_expert <= 8:
        # Sparse: bs=128 with E=257 → ~4.5
        # or bs=16 with E=33 → ~4.4
        block_m = 32
    elif tokens_per_expert <= 40:
        # Medium: bs=128 with E=33 → ~35
        # or bs=512 with E=257 → ~18
        block_m = 32
    else:
        # Dense: bs=512 with E=33 → ~140
        block_m = 64

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
        block_size_M=block_m,
    )
