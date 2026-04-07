"""
MoE: Custom tuning CSV with explicit kernel names from DSV3 CSV.
Maps our exact benchmark shapes to the best CK kernel names.
Uses both cu_num=256 and cu_num=304 to ensure matching.
"""

import os
import sys
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType

# Actual benchmark shapes (from reference kernel README):
# EP-off: bs=4/64/256, E=257, d_hidden=7168, d_expert=256, topk=9
# EP-on:  bs=64/256/1024, E=33, d_hidden=7168, d_expert=2048, topk=9

# From DSV3 CSV, the optimal kernel names by token count for E=257 d_expert=256:
S1_SMALL = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_MED = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_LARGE = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_SMALL = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_LARGE = "moe_ck2stages_gemm2_64x128x128x128_1x1_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

CSV_HEADER = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag"

def make_line(cu, tok, mdim, idim, exp, topk, bm, s1, s2):
    return f"{cu},{tok},{mdim},{idim},{exp},{topk},ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,{bm},0,50.0,{s1},0.0%,30.0,{s2},1.0%,80.0,0,10.0,5000.0,"

# Build CSV content for benchmark shapes
# DSV3 CSV shows for E=257 d_expert=256:
#   token<=32: block_m=32, S1_SMALL, S2_SMALL
#   token=64,128: block_m=32, S1_MED, S2_SMALL 
#   token=256,512,1024: block_m=32, S1_SMALL, S2_SMALL
#   token=2048: block_m=128, S1_LARGE, S2_LARGE

# For E=33 d_expert=2048, we don't have DSV3 data, but let's try different combos
SHAPES = []
for cu in [256, 304]:  # Try both cu_num values to ensure match
    # EP-off shapes (E=257, d_expert=256)
    SHAPES.append((cu, 4, 7168, 256, 257, 9, 32, S1_SMALL, S2_SMALL))
    SHAPES.append((cu, 64, 7168, 256, 257, 9, 32, S1_MED, S2_SMALL))
    SHAPES.append((cu, 256, 7168, 256, 257, 9, 32, S1_SMALL, S2_SMALL))
    # EP-on shapes (E=33, d_expert=2048) - try S1_SMALL for small batch, S1_LARGE for large
    SHAPES.append((cu, 64, 7168, 2048, 33, 9, 32, S1_SMALL, S2_SMALL))
    SHAPES.append((cu, 256, 7168, 2048, 33, 9, 32, S1_SMALL, S2_SMALL))
    SHAPES.append((cu, 1024, 7168, 2048, 33, 9, 32, S1_SMALL, S2_SMALL))

csv_content = CSV_HEADER + "\n"
for cu, tok, mdim, idim, exp, topk, bm, s1, s2 in SHAPES:
    csv_content += make_line(cu, tok, mdim, idim, exp, topk, bm, s1, s2) + "\n"

print(f"[CUSTOM_CSV] Writing CSV with {len(SHAPES)} entries", file=sys.stderr)
print(f"[CUSTOM_CSV] Content:\n{csv_content}", file=sys.stderr)

csv_path = "/tmp/custom_moe_tuned.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

# Set env var BEFORE importing fused_moe
os.environ["AITER_CONFIG_FMOE"] = csv_path
os.environ["AITER_LOG_MORE"] = "2"

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
        block_size_M=None,  # Let CSV control block_m
    )
