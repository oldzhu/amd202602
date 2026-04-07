"""Probe: dump the actual CSV content for our benchmark shapes and check FlyDSL dispatch."""
import os
import sys
import importlib
import torch

# First, check FlyDSL
flydsl_spec = importlib.util.find_spec("flydsl")
print(f"[PROBE] flydsl available: {flydsl_spec is not None}", file=sys.stderr)

# Read the CSV files that get merged
csv_files = [
    "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
]
for csv_path in csv_files:
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        print(f"[PROBE] {csv_path}: {len(lines)} lines", file=sys.stderr)
        # Print header
        if lines:
            print(f"[PROBE] Header: {lines[0].strip()}", file=sys.stderr)
        # Print ALL entries with 257 experts or inter_dim=256 or 512 or 2048
        for i, line in enumerate(lines[1:], 1):
            if "257," in line or ",256," in line or ",512," in line or ",2048," in line:
                # Only show FP4 entries
                if "float4" in line or "fp4" in line.lower():
                    print(f"[PROBE] L{i}: {line.strip()}", file=sys.stderr)
    else:
        print(f"[PROBE] NOT FOUND: {csv_path}", file=sys.stderr)

# Also check if flydsl can actually compile by trying a tiny import
try:
    from aiter.ops.flydsl import utils as flydsl_utils
    print(f"[PROBE] flydsl utils available: {hasattr(flydsl_utils, 'is_flydsl_available')}", file=sys.stderr)
    print(f"[PROBE] is_flydsl_available(): {flydsl_utils.is_flydsl_available()}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] flydsl utils error: {e}", file=sys.stderr)

# Check env vars
for key in ["AITER_USE_FLYDSL", "AITER_KSPLIT", "AITER_USE_NT", "AITER_CONFIG_FMOE"]:
    print(f"[PROBE] {key}={os.environ.get(key, 'not set')}", file=sys.stderr)

from task import input_t, output_t
from aiter import ActivationType, QuantType
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
