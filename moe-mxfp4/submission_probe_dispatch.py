"""
MoE: Diagnostic probe to identify available FlyDSL kernels and CK modules.
Logs kernel selection for each benchmark shape then calls fused_moe normally.
"""

import os
import sys

# Log everything so we can see dispatch choices
os.environ["AITER_LOG_MORE"] = "2"

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Probe FlyDSL availability
try:
    from aiter.ops.flydsl.utils import is_flydsl_available
    print(f"[PROBE] FlyDSL available: {is_flydsl_available()}", file=sys.stderr)
except ImportError as e:
    print(f"[PROBE] FlyDSL import failed: {e}", file=sys.stderr)

# Probe available FlyDSL kernels
try:
    from aiter.ops.flydsl.moe_kernels import _KERNEL_PARAMS
    print(f"[PROBE] FlyDSL kernel count: {len(_KERNEL_PARAMS)}", file=sys.stderr)
    # Show some example kernel names
    for i, name in enumerate(sorted(_KERNEL_PARAMS.keys())):
        if i < 50:  # First 50 only
            if 'fp4' in name:
                print(f"[PROBE] FlyDSL kernel: {name}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] FlyDSL kernel probe failed: {e}", file=sys.stderr)

# Probe get_2stage_cfgs for our shapes
try:
    from aiter.fused_moe import get_2stage_cfgs
    from aiter import dtypes as aiter_dtypes
    
    for tok, mdim, idim, exp, topk in [
        (4, 7168, 256, 257, 9),
        (64, 7168, 256, 257, 9),
        (256, 7168, 256, 257, 9),
        (64, 7168, 2048, 33, 9),
        (256, 7168, 2048, 33, 9),
        (1024, 7168, 2048, 33, 9),
    ]:
        try:
            meta = get_2stage_cfgs(
                token=tok, model_dim=mdim, inter_dim=idim,
                expert=exp, topk=topk, activation=ActivationType.Silu,
                dtype=torch.bfloat16, q_dtype_a=aiter_dtypes.fp4x2,
                q_dtype_w=aiter_dtypes.fp4x2, q_type=QuantType.per_1x32,
                use_g1u1=True, doweight_stage1=False, block_size_M=32
            )
            print(f"[PROBE] shape tok={tok} E={exp} d={idim}: "
                  f"block_m={meta.block_m} ksplit={meta.ksplit} "
                  f"stage1={getattr(meta.stage1, '__name__', str(meta.stage1))} "
                  f"stage2={getattr(meta.stage2, '__name__', str(meta.stage2))} "
                  f"fuse_fp4_quant={getattr(meta, 'fuse_fp4_quant', 'N/A')} "
                  f"run_1stage={meta.run_1stage}",
                  file=sys.stderr)
        except Exception as e:
            print(f"[PROBE] shape tok={tok}: ERROR {e}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] get_2stage_cfgs probe failed: {e}", file=sys.stderr)

# Check tuning CSV path
try:
    from aiter.fused_moe import _tuned_fmoe_csv_path
    print(f"[PROBE] CSV path: {_tuned_fmoe_csv_path}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] CSV path probe: {e}", file=sys.stderr)

# Check CU num
try:
    from aiter import get_cu_num
    print(f"[PROBE] CU num: {get_cu_num()}", file=sys.stderr)
except Exception as e:
    print(f"[PROBE] CU num: {e}", file=sys.stderr)


_CALL_COUNT = 0

def custom_kernel(data: input_t) -> output_t:
    global _CALL_COUNT
    _CALL_COUNT += 1
    
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

    M = topk_ids.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    d = config.get("d_expert", config.get("d_expert_pad", 0))
    
    if _CALL_COUNT <= 10:
        print(f"[PROBE] call#{_CALL_COUNT}: M={M} E={E} d_expert={d} "
              f"topk={topk_ids.shape[1]} w1_shape={gate_up_weight_shuffled.shape} "
              f"w1_dtype={gate_up_weight_shuffled.dtype}", file=sys.stderr)

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

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
