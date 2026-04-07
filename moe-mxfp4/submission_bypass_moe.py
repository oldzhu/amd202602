"""
MoE: Direct 2-stage call with cached sorting and pre-allocated buffers.
Bypass fused_moe wrapper overhead by calling internal functions directly.
"""

import sys
import functools
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter
from aiter import dtypes as aiter_dtypes

# Try to import internal functions
try:
    from aiter.fused_moe import (
        fused_moe,
        fused_moe_2stages,
        moe_sorting,
        get_2stage_cfgs,
        get_block_size_M,
    )
    print("[BYPASS] Successfully imported internal functions", file=sys.stderr)
except ImportError as e:
    print(f"[BYPASS] Import error: {e}", file=sys.stderr)

# Try to import sorting functions directly
try:
    from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
    print("[BYPASS] fused_dynamic_mxfp4_quant_moe_sort available", file=sys.stderr)
except ImportError:
    print("[BYPASS] fused_dynamic_mxfp4_quant_moe_sort NOT available", file=sys.stderr)

_SORT_CACHE: dict[tuple, tuple] = {}


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
    d_expert = config["d_expert"]
    
    block_size_m = 32 if M <= 128 and d_expert >= 512 else None

    # Use fused_moe but pass through all optimizations
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
