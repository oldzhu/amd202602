"""
Probe: Exact benchmark shape discovery.
This probe prints shape info for each call but passes block_size_M=None
so we see the default/CSV-driven block_m for each actual benchmark shape.
"""

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
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    topk = topk_ids.shape[1]

    print(f"[SHAPE-{_call_count}] M={M}, E={E}, d_hidden={d_hidden}, d_expert={d_expert}, topk={topk}, "
          f"hidden_pad={hidden_pad}, intermediate_pad={intermediate_pad}", file=sys.stderr)

    # Let CSV/heuristic fully decide (block_size_M=None → -1 default)
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
    )
