"""
MoE: Direct FlyDSL dispatch. Uses get_flydsl_stage1_kernels/stage2_kernels 
to find available kernels, then calls them directly via custom CSV or direct API.
"""

import os
import sys

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType

# Probe FlyDSL kernel availability and get kernel names
try:
    from aiter.ops.flydsl.moe_kernels import (
        get_flydsl_stage1_kernels,
        get_flydsl_stage2_kernels,
        get_flydsl_kernel_params,
        flydsl_kernel_name,
    )
    
    # List available FlyDSL stage1 kernels for our config
    s1_kernels = get_flydsl_stage1_kernels()
    s2_kernels = get_flydsl_stage2_kernels()
    print(f"[FLYDSL] Stage1 kernels: {s1_kernels}", file=sys.stderr)
    print(f"[FLYDSL] Stage2 kernels: {s2_kernels}", file=sys.stderr)
    
    # Try to get kernel params
    for k in s1_kernels:
        try:
            params = get_flydsl_kernel_params(k)
            print(f"[FLYDSL] s1 kernel {k} params: {params}", file=sys.stderr)
        except Exception as e:
            print(f"[FLYDSL] s1 kernel {k} params error: {e}", file=sys.stderr)
    for k in s2_kernels:
        try:
            params = get_flydsl_kernel_params(k)
            print(f"[FLYDSL] s2 kernel {k} params: {params}", file=sys.stderr)
        except Exception as e:
            print(f"[FLYDSL] s2 kernel {k} params error: {e}", file=sys.stderr)
            
    FLYDSL_AVAILABLE = True
except ImportError as e:
    print(f"[FLYDSL] Import error: {e}", file=sys.stderr)
    FLYDSL_AVAILABLE = False
except Exception as e:
    print(f"[FLYDSL] Error: {e}", file=sys.stderr)
    FLYDSL_AVAILABLE = False

# Check the DSV3 FP4 tuned CSV content
try:
    dsv3_csv = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
    if os.path.exists(dsv3_csv):
        with open(dsv3_csv) as f:
            for line in f:
                print(f"[DSV3_CSV] {line.rstrip()}", file=sys.stderr)
except Exception as e:
    print(f"[DSV3_CSV] Error: {e}", file=sys.stderr)

# Also check untuned_fmoe.csv
try:
    untuned_csv = "/home/runner/aiter/aiter/configs/untuned_fmoe.csv"
    if os.path.exists(untuned_csv):
        with open(untuned_csv) as f:
            lines = f.readlines()
            print(f"[UNTUNED_CSV] {len(lines)} lines", file=sys.stderr)
            # Print first few lines and any fp4 lines
            for i, line in enumerate(lines):
                if i < 3 or 'fp4' in line.lower() or 'per_1x32' in line or 'flydsl' in line.lower():
                    print(f"[UNTUNED_CSV] {line.rstrip()}", file=sys.stderr)
except Exception as e:
    print(f"[UNTUNED_CSV] Error: {e}", file=sys.stderr)

from aiter.fused_moe import fused_moe


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

    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()

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
