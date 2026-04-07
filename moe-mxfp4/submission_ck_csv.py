"""
MoE: Force best CK kernel configs via custom CSV for cu_num=304.
Maps the DSV3-tuned CK kernel names (originally for cu=256) to our cu=304 shapes.
The CK kernels are already compiled, so this just steers dispatch without new JIT.
Also sets block_size_M=None to let CSV control everything.
"""

import os
import sys

CSV_HEADER = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,"
    "block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,"
    "us,run_1stage,tflops,bw,_tag"
)

# CK kernel names from DSV3 CSV (proven correct at cu=256):
# Stage1 small: tile 64x32x32x128, 1x1 pipe
S1_SMALL = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
# Stage1 medium: tile 256x32x128x128, 1x4 pipe
S1_MED = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
# Stage2 small: tile 64x32x32x128, 1x1 pipe
S2_SMALL = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def make_row(cu, tok, mdim, idim, exp, topk, bm, s1_name, s2_name):
    return (
        f"{cu},{tok},{mdim},{idim},{exp},{topk},"
        f"ActivationType.Silu,torch.bfloat16,"
        f"torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,"
        f"QuantType.per_1x32,1,0,"
        f"{bm},0,"
        f"50.0,{s1_name},0.0%,"
        f"30.0,{s2_name},1.0%,"
        f"80.0,0,10.0,5000.0,"
    )

rows = []
# MI355X reports cu_num=256 (confirmed from test logs)

# ── BENCHMARK shapes (from task.yml) ──
# TP=8: E=257, d_expert=256, topk=9 (bs=16,128,512)
for tok in [16, 128, 512]:
    s1 = S1_MED if tok >= 128 else S1_SMALL
    rows.append(make_row(256, tok, 7168, 256, 257, 9, 32, s1, S2_SMALL))

# TP=4: E=33, d_expert=512, topk=9 (bs=16,128,512)
for tok in [16, 128, 512]:
    s1 = S1_MED if tok >= 128 else S1_SMALL
    rows.append(make_row(256, tok, 7168, 512, 33, 9, 32, s1, S2_SMALL))

# EP-on: E=33, d_expert=2048, topk=9 (bs=512)
rows.append(make_row(256, 512, 7168, 2048, 33, 9, 32, S1_MED, S2_SMALL))

# ── TEST shapes (from remote logs) ──
rows.append(make_row(256, 8, 4096, 1024, 257, 9, 32, S1_SMALL, S2_SMALL))
rows.append(make_row(256, 32, 7168, 2048, 33, 9, 32, S1_SMALL, S2_SMALL))
rows.append(make_row(256, 128, 4096, 1536, 65, 7, 32, S1_MED, S2_SMALL))

csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"

csv_path = "/tmp/custom_ck_moe.csv"
with open(csv_path, "w") as f:
    f.write(csv_content)

os.environ["AITER_CONFIG_FMOE"] = csv_path
print(f"[CK_CSV] Wrote {len(rows)} entries to {csv_path}", file=sys.stderr)
print(f"[CK_CSV] CSV:\n{csv_content}", file=sys.stderr)

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
    # Let the CSV control block_m rather than overriding from Python
    # But block_size_M param to fused_moe can be None (CSV wins)
    # or we can still pass 32 as guidance for the sorting step

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
        block_size_M=None,  # Let CSV decide
    )
