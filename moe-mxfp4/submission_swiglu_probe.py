"""
MoE: Test ActivationType.Swiglu instead of Silu.
Research suggests Swiglu triggers a fast path that skips inter-stage re-quantization.
"""

import sys
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType

# Probe ActivationType enum members
print(f"[PROBE] ActivationType members: {list(ActivationType.__members__)}", file=sys.stderr)
for name, val in ActivationType.__members__.items():
    print(f"[PROBE]   {name} = {val.value}", file=sys.stderr)

# Check if Swiglu exists
try:
    swiglu = ActivationType.Swiglu
    print(f"[PROBE] ActivationType.Swiglu = {swiglu.value}", file=sys.stderr)
except AttributeError:
    print("[PROBE] ActivationType.Swiglu does NOT exist", file=sys.stderr)

try:
    swiglu = ActivationType.SwiGLU
    print(f"[PROBE] ActivationType.SwiGLU = {swiglu.value}", file=sys.stderr)
except AttributeError:
    print("[PROBE] ActivationType.SwiGLU does NOT exist", file=sys.stderr)

# Check moe_sorting_dispatch_policy options
try:
    from aiter.fused_moe import fused_moe as fm
    import inspect
    sig = inspect.signature(fm)
    print(f"[PROBE] fused_moe signature: {sig}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] fused_moe sig error: {e}", file=sys.stderr)

# Try to check get_2stage_cfgs to see how activation type affects dispatch
try:
    from aiter.fused_moe import get_2stage_cfgs
    print(f"[PROBE] get_2stage_cfgs found", file=sys.stderr)
    import inspect
    src = inspect.getsource(get_2stage_cfgs)
    # Print lines mentioning activation, Swiglu, per_1x32, etc.
    for i, line in enumerate(src.split('\n')):
        if any(kw in line.lower() for kw in ['swiglu', 'silu', 'activation', 'per_1x32', 'flydsl', 'cktile', 'asm_stage', 'run_1stage']):
            print(f"[PROBE_CFG] L{i}: {line}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] get_2stage_cfgs error: {e}", file=sys.stderr)

# Try to see fused_moe_2stages source for inter-stage quantization
try:
    from aiter.fused_moe import fused_moe_2stages
    import inspect
    src = inspect.getsource(fused_moe_2stages)
    for i, line in enumerate(src.split('\n')):
        if any(kw in line.lower() for kw in ['a2_scale', 'quant', 'swiglu', 'silu', 'activation', 'fused_dynamic', 'mxfp4_quant']):
            print(f"[PROBE_2S] L{i}: {line}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] fused_moe_2stages error: {e}", file=sys.stderr)

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
