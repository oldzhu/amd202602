"""
Probe: Discover what CK kernel names and configs are used for actual benchmark shapes.
Also tests various block_size_M values to find optimal per-shape.
"""

import os
import sys
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_call_count = 0

def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _call_count += 1

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
    E = gate_up_weight_shuffled.shape[0]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    topk = topk_ids.shape[1]

    print(f"[PROBE-{_call_count}] M={M}, E={E}, d_hidden={d_hidden}, d_expert={d_expert}, topk={topk}", file=sys.stderr)
    print(f"  hidden_pad={hidden_pad}, intermediate_pad={intermediate_pad}", file=sys.stderr)
    print(f"  gate_up_weight shape={gate_up_weight_shuffled.shape}", file=sys.stderr)
    print(f"  down_weight shape={down_weight_shuffled.shape}", file=sys.stderr)

    # For the first call of each shape, test with default (no block_size_M override)
    block_size_m = 32 if M <= 256 else None

    result = fused_moe(
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

    return result
