"""
MoE-MXFP4: Fine-tuned CK CSV dispatch per shape.
Uses different CK tile configurations optimized per benchmark shape.
Also experiments with ksplit values and larger tiles for bigger batches.
"""

import os
import sys

CSV_HEADER = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,"
    "block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,"
    "us,run_1stage,tflops,bw,_tag"
)

# CK kernel names from DSV3 tuned CSV:
# Stage1 tiles:
S1_64x32 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256x32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# Stage2 tiles:
S2_64x32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def make_row(cu, tok, mdim, idim, exp, topk, bm, ksplit, s1, s2):
    return (
        f"{cu},{tok},{mdim},{idim},{exp},{topk},"
        f"ActivationType.Silu,torch.bfloat16,"
        f"torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,"
        f"QuantType.per_1x32,1,0,"
        f"{bm},{ksplit},"
        f"50.0,{s1},0.0%,"
        f"30.0,{s2},1.0%,"
        f"80.0,0,10.0,5000.0,"
    )

rows = []

# ── BENCHMARK shapes (from task.yml) ──
# 7 shapes, all cu=256

# Shape 1: bs=16, E=257, d=256 - small batch, few tokens per expert
# Small tile better for sparse dispatch
rows.append(make_row(256, 16, 7168, 256, 257, 9, 32, 0, S1_64x32, S2_64x32))

# Shape 2: bs=128, E=257, d=256 - medium batch
# Larger tile to amortize overhead
rows.append(make_row(256, 128, 7168, 256, 257, 9, 32, 0, S1_256x32, S2_64x32))

# Shape 3: bs=512, E=257, d=256 - large batch
rows.append(make_row(256, 512, 7168, 256, 257, 9, 32, 0, S1_256x32, S2_64x32))

# Shape 4: bs=16, E=33, d=512 - fewer experts but bigger
rows.append(make_row(256, 16, 7168, 512, 33, 9, 32, 0, S1_64x32, S2_64x32))

# Shape 5: bs=128, E=33, d=512
rows.append(make_row(256, 128, 7168, 512, 33, 9, 32, 0, S1_256x32, S2_64x32))

# Shape 6: bs=512, E=33, d=512
rows.append(make_row(256, 512, 7168, 512, 33, 9, 32, 0, S1_256x32, S2_64x32))

# Shape 7: bs=512, E=33, d=2048 - EP-on, large expert dim
rows.append(make_row(256, 512, 7168, 2048, 33, 9, 32, 0, S1_256x32, S2_64x32))

# ── TEST shapes ──
rows.append(make_row(256, 8, 4096, 1024, 257, 9, 32, 0, S1_64x32, S2_64x32))
rows.append(make_row(256, 32, 7168, 2048, 33, 9, 32, 0, S1_64x32, S2_64x32))
rows.append(make_row(256, 128, 4096, 1536, 65, 7, 32, 0, S1_256x32, S2_64x32))

csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"
csv_path = "/tmp/custom_ck_tuned.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

os.environ["AITER_CONFIG_FMOE"] = csv_path
print(f"[CK_TUNED] Wrote {len(rows)} entries to {csv_path}", file=sys.stderr)

import torch
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
