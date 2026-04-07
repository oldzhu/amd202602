"""
MoE probe: Dump the full source of fused_moe and fused_moe_ functions.
"""

import sys
import inspect
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

import aiter.fused_moe as _fmoe_mod


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

    # On first call, dump sources
    if not hasattr(custom_kernel, '_dumped'):
        custom_kernel._dumped = True
        
        # Dump fused_moe source
        try:
            src = inspect.getsource(fused_moe)
            print(f"=== fused_moe source ({len(src.splitlines())} lines) ===", file=sys.stderr)
            for i, line in enumerate(src.splitlines()[:80], 1):
                print(f"  L{i}: {line}", file=sys.stderr)
        except Exception as e:
            print(f"Cannot get fused_moe source: {e}", file=sys.stderr)

        # Dump fused_moe_ source
        if hasattr(_fmoe_mod, 'fused_moe_'):
            try:
                src = inspect.getsource(_fmoe_mod.fused_moe_)
                print(f"=== fused_moe_ source ({len(src.splitlines())} lines) ===", file=sys.stderr)
                for i, line in enumerate(src.splitlines()[:40], 1):
                    print(f"  L{i}: {line}", file=sys.stderr)
            except Exception as e:
                print(f"Cannot get fused_moe_ source: {e}", file=sys.stderr)

        # Dump fused_moe_2stages first 60 lines
        if hasattr(_fmoe_mod, 'fused_moe_2stages'):
            try:
                src = inspect.getsource(_fmoe_mod.fused_moe_2stages)
                print(f"=== fused_moe_2stages source ({len(src.splitlines())} lines) ===", file=sys.stderr)
                for i, line in enumerate(src.splitlines()[:100], 1):
                    print(f"  L{i}: {line}", file=sys.stderr)
            except Exception as e:
                print(f"Cannot get fused_moe_2stages source: {e}", file=sys.stderr)
        
        # Dump moe_sorting source
        if hasattr(_fmoe_mod, 'moe_sorting'):
            try:
                src = inspect.getsource(_fmoe_mod.moe_sorting)
                print(f"=== moe_sorting source ({len(src.splitlines())} lines) ===", file=sys.stderr)
                for i, line in enumerate(src.splitlines()[:50], 1):
                    print(f"  L{i}: {line}", file=sys.stderr)
            except Exception as e:
                print(f"Cannot get moe_sorting source: {e}", file=sys.stderr)
        
        # Look for allocation in fused_moe_2stages
        if hasattr(_fmoe_mod, 'fused_moe_2stages'):
            try:
                src = inspect.getsource(_fmoe_mod.fused_moe_2stages)
                print(f"\n=== fused_moe_2stages ALLOCATION LINES ===", file=sys.stderr)
                for i, line in enumerate(src.splitlines(), 1):
                    stripped = line.strip()
                    if any(kw in stripped for kw in ['torch.empty', 'torch.zeros', '.new_', 'torch.full',
                                                      'allocat', '.zero_', '.fill_']):
                        print(f"  L{i}: {line}", file=sys.stderr)
                # Also lines with torch.ops or direct kernel calls
                print(f"\n=== fused_moe_2stages KERNEL CALL LINES ===", file=sys.stderr)
                for i, line in enumerate(src.splitlines(), 1):
                    stripped = line.strip()
                    if any(kw in stripped for kw in ['ck_moe_stage', 'cktile_moe_gemm', 
                                                      'fused_dynamic_mxfp4', 'moe_sorting',
                                                      'quant_moe_sort']):
                        print(f"  L{i}: {line}", file=sys.stderr)
            except Exception as e:
                print(f"Cannot analyze fused_moe_2stages: {e}", file=sys.stderr)

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
        block_size_M=32,
    )
