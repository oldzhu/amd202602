"""
MoE: Deep bypass - call internal 2-stage pipeline directly.
Cache sorting buffers and metadata per shape.
Also probe fused_moe_ and get_2stage_cfgs internals.
"""

import sys
import functools
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter
from aiter import dtypes as aiter_dtypes

# Probe the internal fused_moe_ function's source to understand dispatch
try:
    import aiter.fused_moe as fmoe_mod
    import inspect
    
    # Get fused_moe_ source (the internal version)
    if hasattr(fmoe_mod, 'fused_moe_'):
        src = inspect.getsource(fmoe_mod.fused_moe_)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if any(kw in line for kw in ['block_size_M', 'moe_sorting', 'sorted', 'expert_ids', 
                                          'sorting_dispatch', 'policy', 'buf', 'alloc', 'empty',
                                          'torch.empty', 'output', 'fused_moe_2stages', 'fused_moe_1stage',
                                          'metadata', 'get_2stage']):
                print(f"[FM_] L{i}: {line}", file=sys.stderr)
        print(f"[FM_] Total lines: {len(lines)}", file=sys.stderr)
    
    # Get fused_moe_2stages source
    if hasattr(fmoe_mod, 'fused_moe_2stages'):
        src = inspect.getsource(fmoe_mod.fused_moe_2stages)
        lines = src.split('\n')
        print(f"\n[FM_2S] fused_moe_2stages total lines: {len(lines)}", file=sys.stderr)
        for i, line in enumerate(lines):
            if any(kw in line for kw in ['alloc', 'empty', 'zeros', 'output', 'contiguous',
                                          'a1', 'a2', 'scale', 'quant', 'sort',
                                          'fused_dynamic', 'moe_mxfp4', 'token_num',
                                          'return', 'stage1', 'stage2']):
                print(f"[FM_2S] L{i}: {line}", file=sys.stderr)
    
    # Get moe_sorting source
    if hasattr(fmoe_mod, 'moe_sorting'):
        src = inspect.getsource(fmoe_mod.moe_sorting)
        lines = src.split('\n')
        print(f"\n[MOE_SORT] moe_sorting total lines: {len(lines)}", file=sys.stderr)
        for i, line in enumerate(lines):
            print(f"[MOE_SORT] L{i}: {line}", file=sys.stderr)
    
except Exception as e:
    print(f"[PROBE] Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

from aiter.fused_moe import fused_moe


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
    M = hidden_states.shape[0]
    d_expert = config["d_expert"]
    block_size_m = 32 if M <= 128 and d_expert >= 512 else None

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
