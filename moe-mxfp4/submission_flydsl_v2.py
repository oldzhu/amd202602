"""
MoE-MXFP4: FlyDSL-enabled CSV for E=257 shapes.
FlyDSL is confirmed available on runner (v0.0.1.dev).
Uses FlyDSL kernel names from dsv3_fp4_tuned_fmoe.csv for E=257 shapes.
E=33 shapes use CK fallback since CSV doesn't have them.
"""

import os
import sys

CSV_HEADER = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,"
    "block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,"
    "us,run_1stage,tflops,bw,_tag"
)

# FlyDSL stage1 kernel names (from dsv3_fp4_tuned_fmoe.csv)
FLY_S1_32x32 = "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3"
FLY_S1_32x64 = "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w2"
FLY_S1_64x64 = "flydsl_moe1_afp4_wfp4_bf16_t64x64x256_w4"

# FlyDSL stage2 kernel names
FLY_S2_64x256_red = "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce"
FLY_S2_32x128_red = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce"

# CK fallback stage2 (used by CSV for small tokens)
CK_S2_64x32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
# CK stage1 for test shapes
CK_S1_64x32 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
CK_S1_256x32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"


def make_row(cu, tok, mdim, idim, exp, topk, bm, ksplit, s1, s2, tag=""):
    return (
        f"{cu},{tok},{mdim},{idim},{exp},{topk},"
        f"ActivationType.Silu,torch.bfloat16,"
        f"torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,"
        f"QuantType.per_1x32,1,0,"
        f"{bm},{ksplit},"
        f"50.0,{s1},0.0%,"
        f"30.0,{s2},1.0%,"
        f"80.0,0,10.0,5000.0,{tag}"
    )


rows = []

# === E=257, d=256 (Benchmark shapes) - Use FlyDSL kernels! ===
# token=16 (bs=16): small batch, FlyDSL stage1 32x32, CK stage2
rows.append(make_row(256, 16, 7168, 256, 257, 9, 32, 0, FLY_S1_32x32, CK_S2_64x32))
# token=128 (bs=128): mid batch, FlyDSL stage1 64x64, FlyDSL stage2
rows.append(make_row(256, 128, 7168, 256, 257, 9, 64, 0, FLY_S1_64x64, FLY_S2_64x256_red))
# token=512 (bs=512): large batch, FlyDSL stage1 64x64, FlyDSL stage2
rows.append(make_row(256, 512, 7168, 256, 257, 9, 64, 0, FLY_S1_64x64, FLY_S2_64x256_red))

# === E=33, d=512 (Benchmark shapes) - CK fallback ===
rows.append(make_row(256, 16, 7168, 512, 33, 9, 32, 0, CK_S1_64x32, CK_S2_64x32))
rows.append(make_row(256, 128, 7168, 512, 33, 9, 64, 0, CK_S1_256x32, CK_S2_64x32))
rows.append(make_row(256, 512, 7168, 512, 33, 9, 64, 0, CK_S1_256x32, CK_S2_64x32))

# === E=33, d=2048 (Benchmark shape) - CK fallback ===
rows.append(make_row(256, 512, 7168, 2048, 33, 9, 128, 0, CK_S1_256x32, CK_S2_64x32))

# === Test shapes - CK ===
rows.append(make_row(256, 8, 4096, 1024, 257, 9, 32, 0, CK_S1_64x32, CK_S2_64x32))
rows.append(make_row(256, 32, 7168, 2048, 33, 9, 32, 0, CK_S1_64x32, CK_S2_64x32))
rows.append(make_row(256, 128, 4096, 1536, 65, 7, 64, 0, CK_S1_256x32, CK_S2_64x32))

csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"
csv_path = "/tmp/custom_flydsl_v2.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

os.environ["AITER_CONFIG_FMOE"] = csv_path
print(f"[FLYDSL_V2] Wrote {len(rows)} entries to {csv_path}", file=sys.stderr)

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

    # Don't set block_size_M - let CSV control it
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
