"""Probe: check if FlyDSL is available on runner and what CSV dispatch decides."""
import os
import sys
import importlib
import torch
from task import input_t, output_t

# Check FlyDSL availability
flydsl_spec = importlib.util.find_spec("flydsl")
print(f"[PROBE] flydsl spec: {flydsl_spec}", file=sys.stderr)
print(f"[PROBE] flydsl available: {flydsl_spec is not None}", file=sys.stderr)

if flydsl_spec is not None:
    try:
        import flydsl
        print(f"[PROBE] flydsl version: {getattr(flydsl, '__version__', 'unknown')}", file=sys.stderr)
    except Exception as e:
        print(f"[PROBE] flydsl import error: {e}", file=sys.stderr)

# Check what CSV is loaded
try:
    from aiter.fused_moe import fused_moe
    csv_path = os.environ.get("AITER_CONFIG_FMOE", "not set")
    print(f"[PROBE] AITER_CONFIG_FMOE env: {csv_path}", file=sys.stderr)
    
    # Try to read the merged CSV
    import glob
    csv_files = glob.glob("/home/runner/aiter/aiter/configs/*.csv") + glob.glob("/home/runner/aiter/aiter/configs/model_configs/*.csv")
    for f in sorted(csv_files):
        print(f"[PROBE] Found CSV: {f}", file=sys.stderr)
    
    # Read the merged CSV content for our shapes
    default_csv = "/tmp/aiter_configs/tuned_fmoe.csv"
    if os.path.exists(default_csv):
        with open(default_csv) as fh:
            lines = fh.readlines()
        print(f"[PROBE] Default CSV has {len(lines)} lines", file=sys.stderr)
        for line in lines[:3]:
            print(f"[PROBE] CSV header/first: {line.strip()}", file=sys.stderr)
        # Find entries for E=257 and E=33
        for line in lines:
            if ",257," in line or ",33," in line:
                print(f"[PROBE] CSV entry: {line.strip()}", file=sys.stderr)
    else:
        print(f"[PROBE] Default CSV not found at {default_csv}", file=sys.stderr)
        # Check /tmp/aiter_configs/
        tmp_csvs = glob.glob("/tmp/aiter_configs/*.csv")
        print(f"[PROBE] /tmp/aiter_configs/ files: {tmp_csvs}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] Error: {e}", file=sys.stderr)

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
