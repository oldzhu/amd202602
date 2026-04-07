"""
MoE diagnostic probe: discover available FlyDSL kernels and current dispatch.
"""

import os
import sys

# Enable verbose logging to see kernel selection
os.environ["AITER_LOG_MORE"] = "2"
os.environ["AITER_LOG_TUNED_CONFIG"] = "1"

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Probe FlyDSL availability
try:
    from aiter.ops.flydsl import moe_kernels
    print(f"[PROBE] FlyDSL moe_kernels available: {dir(moe_kernels)}", file=sys.stderr)
except ImportError as e:
    print(f"[PROBE] FlyDSL moe_kernels NOT available: {e}", file=sys.stderr)

try:
    import aiter.ops.flydsl as flydsl_mod
    print(f"[PROBE] FlyDSL module contents: {dir(flydsl_mod)}", file=sys.stderr)
except ImportError as e:
    print(f"[PROBE] FlyDSL module NOT available: {e}", file=sys.stderr)

# Check for tuning CSV paths
try:
    from aiter.fused_moe import get_tuned_config
    print(f"[PROBE] get_tuned_config available", file=sys.stderr)
except ImportError:
    pass

# Check for moe_sorting  
try:
    from aiter.fused_moe import moe_sorting
    print(f"[PROBE] moe_sorting available", file=sys.stderr)
except ImportError:
    pass

# Check for fused_moe_2stages
try:
    from aiter.fused_moe import fused_moe_2stages
    print(f"[PROBE] fused_moe_2stages available", file=sys.stderr)
except ImportError:
    pass

# List AITER config directory
try:
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aiter', 'configs')
    if not os.path.exists(config_dir):
        import aiter
        aiter_dir = os.path.dirname(aiter.__file__)
        config_dir = os.path.join(aiter_dir, 'configs')
    if os.path.exists(config_dir):
        print(f"[PROBE] Config dir: {config_dir}", file=sys.stderr)
        for f in sorted(os.listdir(config_dir)):
            if 'moe' in f.lower() or 'fmoe' in f.lower():
                print(f"[PROBE]   {f}", file=sys.stderr)
    # Check model_configs too
    model_dir = os.path.join(config_dir, 'model_configs')
    if os.path.exists(model_dir):
        for f in sorted(os.listdir(model_dir)):
            if 'moe' in f.lower() or 'fmoe' in f.lower() or 'fp4' in f.lower():
                print(f"[PROBE]   model_configs/{f}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] Config dir error: {e}", file=sys.stderr)

# List available AITER fused_moe submodules
try:
    import aiter.fused_moe as fm
    print(f"[PROBE] fused_moe module contents: {[x for x in dir(fm) if not x.startswith('_')]}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] fused_moe dir error: {e}", file=sys.stderr)

# Check AITER env vars
for key in sorted(os.environ):
    if 'AITER' in key.upper():
        print(f"[PROBE] ENV {key}={os.environ[key]}", file=sys.stderr)


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
    )
