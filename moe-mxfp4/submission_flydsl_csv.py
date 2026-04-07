"""
MoE: Force FlyDSL dispatch via custom tuning CSV.
FlyDSL with _fq (fuse_fp4_quant) fuses activation quantization into stage1 GEMM.
This eliminates the separate dynamic_mxfp4_quant + moe_mxfp4_sort passes.
"""

import os
import sys
import tempfile

# ── Build and install custom CSV BEFORE any aiter imports ───────────────
# CSV lookup key: (cu_num, token, model_dim, inter_dim, expert, topk,
#                  act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)
# MI355X has 304 CUs.

CSV_HEADER = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,"
    "block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,"
    "us,run_1stage,tflops,bw,_tag"
)

# FlyDSL kernel names for a=fp4, w=fp4, out=bf16
# Stage1 with _fq = fuse fp4 quantization into GEMM (eliminates separate quant pass!)
# Stage2 with reduce mode

def make_row(cu, tok, mdim, idim, exp, topk, bm, s1_name, s2_name):
    return (
        f"{cu},{tok},{mdim},{idim},{exp},{topk},"
        f"ActivationType.Silu,torch.bfloat16,"
        f"torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,"
        f"QuantType.per_1x32,1,0,"
        f"{bm},0,"  # block_m, ksplit
        f"50.0,{s1_name},0.0%,"  # us1, kernelName1, err1
        f"30.0,{s2_name},1.0%,"  # us2, kernelName2, err2
        f"80.0,0,10.0,5000.0,"   # us, run_1stage, tflops, bw, _tag
    )

# For fp4 activations x fp4 weights -> bf16 output:
# Stage1: tile_m x tile_n x tile_k
#   Small batch (bs<=64*topk): tile_m=32, tile_n=128, tile_k=256
#   Large batch: tile_m=64, tile_n=128, tile_k=256
# Stage2: tile_m x tile_n x tile_k, mode=reduce
#   Small: tile_m=32, tile_n=128, tile_k=256
#   Large: tile_m=64, tile_n=256, tile_k=256

# Try conservative tile configs first - match what DSV3 CSV uses
# DSV3 used S1=64x32x32x128 for CK, but for FlyDSL tile_k=256 always

# Stage1 options (with _fq for fused quant):
S1_32x128 = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_fq"
S1_32x64  = "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_fq"
S1_64x128 = "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_fq"
S1_16x128 = "flydsl_moe1_afp4_wfp4_bf16_t16x128x256_fq"
S1_16x64  = "flydsl_moe1_afp4_wfp4_bf16_t16x64x256_fq"

# Stage2 options:
S2_32x128_reduce  = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce"
S2_32x256_reduce  = "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_reduce"
S2_64x128_reduce  = "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_reduce"
S2_64x256_reduce  = "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce"
S2_32x128_atomic  = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic"
S2_32x256_atomic  = "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic"

rows = []
# MI355X reports cu_num=256 (confirmed via test logs)
for cu in [256]:
    # ── BENCHMARK shapes (from task.yml) ──
    # TP=8 group: E=257, d_expert=256, topk=9 (bs=16,128,512)
    for bs in [16, 128, 512]:
        s1 = S1_32x128 if bs <= 128 else S1_64x128
        s2 = S2_32x128_reduce if bs <= 128 else S2_64x128_reduce
        rows.append(make_row(cu, bs, 7168, 256, 257, 9, 32, s1, s2))

    # TP=4 group: E=33, d_expert=512, topk=9 (bs=16,128,512)
    for bs in [16, 128, 512]:
        s1 = S1_32x128 if bs <= 128 else S1_64x128
        s2 = S2_32x128_reduce if bs <= 128 else S2_64x128_reduce
        rows.append(make_row(cu, bs, 7168, 512, 33, 9, 32, s1, s2))

    # EP-on group: E=33, d_expert=2048, topk=9 (bs=512 only)
    rows.append(make_row(cu, 512, 7168, 2048, 33, 9, 32, S1_64x128, S2_64x128_reduce))

    # ── TEST shapes (from remote logs) ──
    rows.append(make_row(cu, 8, 4096, 1024, 257, 9, 32, S1_32x128, S2_32x128_reduce))
    rows.append(make_row(cu, 32, 7168, 2048, 33, 9, 32, S1_32x128, S2_32x128_reduce))
    rows.append(make_row(cu, 128, 4096, 1536, 65, 7, 32, S1_32x128, S2_32x128_reduce))

csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"

csv_path = "/tmp/custom_flydsl_moe.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

os.environ["AITER_CONFIG_FMOE"] = csv_path
print(f"[FLYDSL_CSV] Wrote {len(rows)} entries to {csv_path}", file=sys.stderr)
print(f"[FLYDSL_CSV] CSV content:\n{csv_content}", file=sys.stderr)

# ── Now import aiter ───────────────────────────────────────────────────
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

    M = topk_ids.shape[0]
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
