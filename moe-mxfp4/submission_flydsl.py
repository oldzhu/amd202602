"""
MoE: Try FlyDSL kernels via custom tuning CSV override.
Attempts to make fused_moe dispatch to FlyDSL by setting AITER_CONFIG_FMOE.
"""

import os
import sys
import tempfile

# Create a custom tuning CSV that hints FlyDSL kernel names
# The actual kernel names need to match what's compiled on the runner
# We'll try a few common FlyDSL naming patterns

# First, get AITER path to find default CSV
try:
    import aiter
    aiter_dir = os.path.dirname(aiter.__file__)
    config_dir = os.path.join(aiter_dir, 'configs')
    default_csv = os.path.join(config_dir, 'tuned_fmoe.csv')
    
    # Check for DSV3 fp4 model config (most relevant for MXFP4)
    dsv3_csv = os.path.join(config_dir, 'model_configs', 'dsv3_fp4_tuned_fmoe.csv')
    
    if os.path.exists(dsv3_csv):
        # Merge DSV3 FP4 tuned config with default
        os.environ["AITER_CONFIG_FMOE"] = f"{dsv3_csv}:{default_csv}"
        print(f"[FLYDSL] Using DSV3 FP4 config: {dsv3_csv}", file=sys.stderr)
        # Print content to see what kernels are available
        with open(dsv3_csv) as f:
            for line in f:
                print(f"[FLYDSL_CSV] {line.rstrip()}", file=sys.stderr)
    else:
        print(f"[FLYDSL] DSV3 FP4 config not found at {dsv3_csv}", file=sys.stderr)
        # List what model configs exist
        model_dir = os.path.join(config_dir, 'model_configs')
        if os.path.exists(model_dir):
            for name in sorted(os.listdir(model_dir)):
                print(f"[FLYDSL] Available model config: {name}", file=sys.stderr)
            # Try all model configs that mention fp4
            for name in sorted(os.listdir(model_dir)):
                if 'fp4' in name.lower():
                    csv_path = os.path.join(model_dir, name)
                    with open(csv_path) as f:
                        for line in f:
                            print(f"[FLYDSL_{name}] {line.rstrip()}", file=sys.stderr)
except Exception as e:
    print(f"[FLYDSL] Config error: {e}", file=sys.stderr)

# Also enable verbose logging  
os.environ["AITER_LOG_TUNED_CONFIG"] = "1"

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
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
