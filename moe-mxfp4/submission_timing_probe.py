"""
MoE probe: Time internal stages of fused_moe to identify optimization targets.
Uses monkey-patching to add timing around internal functions.
"""

import sys
import time
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Patch fused_moe_2stages to add timing
import aiter.fused_moe as _fmoe_mod

_orig_2stages = None
_orig_sorting = None
_call_count = 0
_timing_data = []


def _patched_sorting(topk_ids, topk_weights, n_experts, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_sorting(topk_ids, topk_weights, n_experts, *args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    _timing_data.append(("sort", (t1 - t0) * 1e6))
    return result


def _patched_2stages(hidden_states, w1, w2, topk_ids, topk_weights, *args, **kwargs):
    global _call_count
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig_2stages(hidden_states, w1, w2, topk_ids, topk_weights, *args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_us = (t1 - t0) * 1e6
    M = hidden_states.shape[0]
    d = hidden_states.shape[1]
    E = w1.shape[0]
    _call_count += 1
    
    sort_us = sum(t for name, t in _timing_data if name == "sort")
    _timing_data.clear()
    
    print(f"[TIMING] call={_call_count} M={M} d={d} E={E} topk={topk_ids.shape[1]} "
          f"total={total_us:.1f}us sort={sort_us:.1f}us compute={total_us-sort_us:.1f}us",
          file=sys.stderr)
    return result


# Apply patches
if hasattr(_fmoe_mod, 'fused_moe_2stages'):
    _orig_2stages = _fmoe_mod.fused_moe_2stages
    _fmoe_mod.fused_moe_2stages = _patched_2stages
    print("[PROBE] Patched fused_moe_2stages", file=sys.stderr)

if hasattr(_fmoe_mod, 'moe_sorting'):
    _orig_sorting = _fmoe_mod.moe_sorting
    _fmoe_mod.moe_sorting = _patched_sorting
    print("[PROBE] Patched moe_sorting", file=sys.stderr)


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
